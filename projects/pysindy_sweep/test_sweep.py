"""Scientific invariants: grammar, leakage, scoring, and checkpointing."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from library import (Observations, build_problem, expand_truth, make_factors,
                     multiply_keys, term_key, term_name, differentiate)
from run_sweep import (fit_problem, read_checkpoint, score_fit, settings_from_grid,
                       observations_for, truth_forms, export_results, digest)


class SweepTests(unittest.TestCase):
    def test_paper_library_sizes_and_no_target_leakage(self):
        for dataset, target, size in (("ode_data.npy", "u_tt", 62), ("wave_data.csv", "u_tt", 226)):
            obs = observations_for(dataset, 0, 0, 0)
            factors, derivatives = make_factors(obs)
            problem = build_problem(factors, derivatives, target)
            self.assertEqual(len(problem.keys), size)
            self.assertTrue(all(target not in dict(key) for key in problem.keys))
            self.assertEqual(len(set(problem.keys)), size)
            self.assertTrue(np.isfinite(problem.matrix).all())

    def test_canonical_names_roundtrip(self):
        for text in ("u^2 u_t", "u_t sin(2 t)", "(1/x) u_x", "sin(x) cos(t)",
                     "u_x^2", "x0 x1", "(1/t)^2 u^3", "sin(t)^2", "1"):
            key = term_key(text)
            self.assertEqual(term_key(term_name(key)), key)
        self.assertEqual(multiply_keys(term_key("u^2"), term_key("u")), term_key("u^3"))
        self.assertEqual(multiply_keys(term_key("x"), term_key("(1/x)")), ())

    def test_compound_and_laplacian_truth(self):
        self.assertEqual(expand_truth({"d_x(u u_x)": 1}), expand_truth({"u_x^2": 1, "u u_xx": 1}))
        self.assertEqual(expand_truth({"(u_xx + u_yy)": 0.01}),
                         expand_truth({"u_xx": 0.01, "u_yy": 0.01}))

    def test_all_benchmark_laws_are_in_grammar(self):
        from datasets import TRUE_COEFFICIENTS
        for dataset, targets in TRUE_COEFFICIENTS.items():
            obs = observations_for(dataset, 0, 0, 0)
            factors, derivatives = make_factors(obs)
            for target in targets:
                problem = build_problem(factors, derivatives, target, max_samples=100)
                self.assertTrue(any(set(form) <= set(problem.keys) for form in truth_forms(dataset, target)),
                                (dataset, target))

    def test_fd_derivatives_of_polynomial(self):
        t = np.linspace(0, 1, 101)
        for d, expected in ((1, 3 * t ** 2), (2, 6 * t)):
            actual = differentiate(t ** 3, t, 0, d, fd_order=4)
            np.testing.assert_allclose(actual[5:-5], expected[5:-5], atol=1e-8)

    def test_noise_and_sampling_are_reproducible(self):
        first = observations_for("vdp_data.npy", 1, 0, 42)
        second = observations_for("vdp_data.npy", 1, 0, 42)
        different = observations_for("vdp_data.npy", 1, 1, 42)
        np.testing.assert_array_equal(first.fields["u"], second.fields["u"])
        self.assertFalse(np.array_equal(first.fields["u"], different.fields["u"]))
        factors, derivatives = make_factors(first)
        one = build_problem(factors, derivatives, "u_tt", max_samples=50, seed=42)
        two = build_problem(factors, derivatives, "u_tt", max_samples=50, seed=42)
        np.testing.assert_array_equal(one.row_indices, two.row_indices)

    def test_exact_support_and_alternatives(self):
        record = dict(dataset="pde_compound_data.npy", target="u_t", status="ok",
                      feature_keys=[term_key("u_x^2"), term_key("u u_xx"), term_key("u")],
                      coefficients=[1, 1, 1e-5])
        self.assertFalse(score_fit(record, 0)["structure_success"])
        self.assertTrue(score_fit(record, 1e-4)["structure_success"])
        record["coefficients"] = [4, 2, 0]
        # Structure accuracy is independent of coefficient accuracy.
        self.assertTrue(score_fit(record, 0)["structure_success"])
        self.assertEqual(score_fit(record, 0)["coefficient_error"], 4)
        record["target"] = "u_tt"
        self.assertFalse(score_fit(record, 0)["structure_success"])

    def test_sparse_optimizers_recover_known_support(self):
        from library import Problem
        rng = np.random.default_rng(0)
        matrix, _ = np.linalg.qr(rng.normal(size=(200, 5)))
        response = matrix @ np.array([2, 0, -1, 0, 0])
        problem = Problem("u_t", matrix, response, [term_key(f"x{i}") for i in range(5)],
                          [f"x{i}" for i in range(5)], 200, 200, [], np.arange(200))
        for optimizer in ("STLSQ", "FROLS"):
            setting = dict(optimizer=optimizer, threshold=0.1 if optimizer == "STLSQ" else None,
                           max_iter=100 if optimizer == "STLSQ" else 2, alpha=1e-10,
                           normalize_columns=True, unbias=True)
            coefficients = fit_problem(problem, setting)
            np.testing.assert_allclose(coefficients, [2, 0, -1, 0, 0], atol=1e-10)

    def test_paper_grid_has_2200_scored_settings(self):
        grid = json.loads((Path(__file__).parent / "paper_grid.json").read_text())
        self.assertEqual(len(settings_from_grid(grid)), 200)
        self.assertEqual(len(grid["coefficient_tolerances"]), 11)

    def test_checkpoint_recovers_partial_final_line(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "fits.jsonl"
            path.write_bytes(b'{"fit_id": "one"}\n{"fit_id":')
            self.assertEqual(read_checkpoint(path), [{"fit_id": "one"}])
            self.assertEqual(path.read_bytes(), b'{"fit_id": "one"}\n')

    def test_checkpoint_adds_missing_newline(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "fits.jsonl"
            path.write_bytes(b'{"fit_id": "one"}')
            self.assertEqual(read_checkpoint(path), [{"fit_id": "one"}])
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_system_requires_every_component_and_counts_missing_runs(self):
        import argparse
        import csv
        from datasets import TRUE_COEFFICIENTS
        setting = dict(optimizer="FROLS", threshold=None, max_iter=3, alpha=1e-5,
                       normalize_columns=True, unbias=True)
        records = []
        for target, coefficients in TRUE_COEFFICIENTS["lorenz_data.npy"].items():
            keys = [term_key(name) for name in coefficients]
            values = list(coefficients.values())
            if target == "x2_t":
                keys.append(term_key("u"))
                values.append(1)
            records.append(dict(dataset="lorenz_data.npy", target=target, noise_percent=0,
                                run=0, setting_id=digest(setting), **setting, status="ok",
                                coefficients=values, feature_keys=keys))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            output = Path(directory)
            args = argparse.Namespace(datasets=["lorenz_data.npy"], noise_levels=[0], runs=2)
            export_results(output, records, args, [setting], {"coefficient_tolerances": [0]})
            with (output / "by_dataset.csv").open(encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["correct_count"], "0")
            self.assertEqual(row["completed_runs"], "1")
            self.assertEqual(row["failed_or_missing_runs"], "1")
            with (output / "by_setting.csv").open(encoding="utf-8-sig") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["correct_equation_runs"], "2")
            self.assertEqual(row["planned_equation_runs"], "6")

    def test_reciprocal_singularities_are_masked(self):
        t = np.linspace(0, 1, 50)
        factors, derivatives = make_factors(Observations({"u": np.exp(t)}, [("t", 0, t)]))
        problem = build_problem(factors, derivatives, "u_t", crop=0, max_samples=0)
        self.assertEqual(problem.valid_rows, 49)
        self.assertNotIn(0, problem.row_indices)


if __name__ == "__main__":
    unittest.main()
