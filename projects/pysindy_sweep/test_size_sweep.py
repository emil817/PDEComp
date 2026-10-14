"""Scientific controls for complete, comparable and resumable size experiments."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

from library import build_problem, make_factors, term_key
from make_size_catalog import build_catalog
from run_sweep import (TRUE_COEFFICIENTS, digest, observations_for, seed_for,
                       settings_from_grid, truth_forms, write_csv, write_json)
from run_size_sweep import (benchmark_and_full_ac_forms, job_id, make_plan, run_job, score_sparse)
from summarize_size_sweep import aggregate, read_job
from variable_library import FactorBank, LibrarySpec, contains_term, enumerate_keys


def plan_args(datasets, max_features=6000):
    return argparse.Namespace(datasets=datasets, seed=0, crop=5, max_samples=100,
        max_features=max_features, max_matrix_mb=256., max_working_mb=2048.,
        noise_levels=[0.], runs=1)


class SizeSweepTests(unittest.TestCase):
    def test_reference_reproduces_original_grammar_and_values(self):
        spec = LibrarySpec(total_field_degree=None, time_order=2,
                           derivative_interactions="all", time_scope="all_fields")
        for dataset, target, size in (("ode_data.npy", "u_tt", 62), ("wave_data.csv", "u_tt", 226)):
            obs = observations_for(dataset, 0., 0, 0)
            factors, derivatives = make_factors(obs)
            original = build_problem(factors, derivatives, target, max_samples=100, seed=0)
            keys, count = enumerate_keys(list(obs.fields), [a[0] for a in obs.axes], target, spec)
            bank = FactorBank(obs)
            rows, full, valid = bank.common_rows([spec], [target], 5, 100, 0)
            bank.freeze_rows(rows)
            problem = bank.build(keys, target, rows, full, valid)
            self.assertEqual(count, size)
            self.assertEqual(set(keys), set(original.keys))
            np.testing.assert_array_equal(rows, original.row_indices)
            positions = {key: i for i, key in enumerate(original.keys)}
            np.testing.assert_allclose(problem.matrix, original.matrix[:, [positions[k] for k in keys]], rtol=1e-12, atol=1e-12)

    def test_complete_baseline_covers_every_benchmark_and_full_ac(self):
        base = build_catalog()["libraries"][0]
        grid = json.loads((HERE / "smoke_grid.json").read_text())
        plan, summary = make_plan(plan_args(list(TRUE_COEFFICIENTS)),
                                  dict(libraries=[base]), settings_from_grid(grid), grid)
        self.assertEqual(summary["equations"], 20)
        self.assertTrue(all(any(p["truth_coverage"]) for p in plan))
        self.assertTrue(any(p["full_ac_truth_coverage"] == [True] for p in plan))

    def test_incomplete_libraries_are_rejected(self):
        grid = json.loads((HERE / "smoke_grid.json").read_text())
        for changes in (dict(field_power=2, total_field_degree=2), dict(space_order=2),
                        dict(max_factors=1), dict(trig_frequency=1)):
            bad = dict(id="bad", spec=replace(LibrarySpec(), **changes).as_dict())
            dataset = ("ac_data.npy" if "field_power" in changes else "kdv_data.mat"
                       if "space_order" in changes else "ode_data.npy")
            with self.assertRaisesRegex(ValueError, "Incomplete library"):
                make_plan(plan_args([dataset]), dict(libraries=[bad]), settings_from_grid(grid), grid)

    def test_membership_matches_enumeration_including_squares(self):
        fields, axes, target = ["u"], ["t", "x"], "u_t"
        for spec in (LibrarySpec(), replace(LibrarySpec(), max_factors=3),
                     replace(LibrarySpec(), derivative_interactions="all")):
            keys, _ = enumerate_keys(fields, axes, target, spec)
            for text in ("u_x^2", "u u_xx", "u^3", "u^4", "u_x u_xx", "(1/x) u_x", "sin(x) cos(t)"):
                key = term_key(text)
                self.assertEqual(contains_term(key, fields, axes, target, spec), key in keys)

    def test_degree_is_total_across_fields_and_target_is_always_excluded(self):
        for factors in (2, 3):
            spec = replace(LibrarySpec(), max_factors=factors, time_order=3,
                           time_scope="all_fields", derivative_interactions="all")
            keys, _ = enumerate_keys(["x0", "x1"], ["t"], "x0_t", spec)
            self.assertTrue(all("x0_t" not in dict(key) for key in keys))
            self.assertTrue(all(sum(p for atom, p in key if atom in ("x0", "x1")) <= 3 for key in keys))
            self.assertIn(term_key("x0^2 x1"), keys)
            self.assertNotIn(term_key("x0^2 x1^2"), keys)

    def test_shared_rows_and_response_when_derivative_controls_change(self):
        obs = observations_for("wave_data.csv", 0., 0, 0)
        specs = [LibrarySpec(), replace(LibrarySpec(), space_order=6, time_order=3, max_factors=3)]
        bank = FactorBank(obs)
        rows, full, valid = bank.common_rows(specs, ["u_tt"], 5, 200, 19)
        bank.freeze_rows(rows)
        problems = []
        for spec in specs:
            keys, _ = enumerate_keys(["u"], ["t", "x"], "u_tt", spec)
            problems.append(bank.build(keys, "u_tt", rows, full, valid))
        np.testing.assert_array_equal(problems[0].response, problems[1].response)
        np.testing.assert_array_equal(problems[0].row_indices, problems[1].row_indices)
        self.assertLess(len(problems[0].keys), len(problems[1].keys))

    def test_equivalent_ode_grammars_deduplicate_irrelevant_spatial_order(self):
        one, _ = enumerate_keys(["u"], ["t"], "u_tt", LibrarySpec())
        two, _ = enumerate_keys(["u"], ["t"], "u_tt", replace(LibrarySpec(), space_order=8))
        self.assertEqual(digest(one), digest(two))

    def test_nested_growth_is_nested(self):
        chain = [(g["value"], LibrarySpec(**entry["spec"])) for entry in build_catalog()["libraries"]
                 for g in entry["groups"] if g["family"] == "nested_growth"]
        previous = set()
        for _, spec in sorted(chain, key=lambda item: item[0]):
            keys, _ = enumerate_keys(["u"], ["t", "x"], "u_t", spec)
            self.assertTrue(previous <= set(keys))
            previous = set(keys)

    def test_budget_violation_rejects_catalog_instead_of_skipping(self):
        grid = json.loads((HERE / "smoke_grid.json").read_text())
        base = build_catalog()["libraries"][0]
        with self.assertRaisesRegex(ValueError, "does not skip pairs"):
            make_plan(plan_args(["wave_data.csv"], max_features=5),
                      dict(libraries=[base]), settings_from_grid(grid), grid)

    def test_calibrated_envelope_has_complete_multilevel_bounds_and_headroom(self):
        calibration = json.loads((HERE / "size_bounds.json").read_text())
        bounds = calibration["bounds"]
        for name in ("field_degree", "space_order", "time_order", "trig_frequency", "coordinate_power"):
            self.assertGreaterEqual(len(bounds[name]), 2)
        self.assertEqual(bounds["max_factors"], [2])
        limits, margin = calibration["budgets"], calibration["budget_margin"]
        for row in calibration["checked_envelope"]:
            self.assertLessEqual(row["features"], limits["max_features"] * (1 - margin))
            self.assertLessEqual(row["matrix_mb"], limits["max_matrix_mb"] * (1 - margin))
            self.assertLessEqual(row["working_mb"], limits["max_working_mb"] * (1 - margin))
        catalog = build_catalog()
        expected = np.prod([len(values) for values in bounds.values()])
        factorial = sum(any(g["family"] == "factorial" for g in entry["groups"])
                        for entry in catalog["libraries"])
        self.assertEqual(factorial, expected)

    def test_sparse_checkpoint_and_full_ac_scoring(self):
        keys = [term_key("u"), term_key("u^3"), term_key("u_xx")]
        record = dict(dataset="ac_data.npy", target="u_t", status="ok", error="",
                      nonzero_indices=[0, 1, 2], nonzero_coefficients=[5., -5., 1e-4], warnings=[])
        row = score_sparse(record, keys, 0.)
        self.assertFalse(row["structure_success"])
        self.assertTrue(row["full_ac_structure_success"])
        self.assertTrue(row["full_ac_law_success"])
        row = score_sparse(record, keys, .001)
        self.assertTrue(row["structure_success"])
        self.assertFalse(row["full_ac_structure_success"])

    def test_resuming_job_does_not_refit_and_aggregation_preserves_errors(self):
        from unittest.mock import patch
        args = plan_args(["vdp_data.npy"])
        base = build_catalog()["libraries"][0]
        grid = json.loads((HERE / "smoke_grid.json").read_text())
        settings = settings_from_grid(grid)
        catalog = dict(schema_version=1, libraries=[base])
        plan, summary = make_plan(args, catalog, settings, grid)
        obs = observations_for("vdp_data.npy", 0., 0, 0)
        bank = FactorBank(obs)
        rows, full, valid = bank.common_rows([LibrarySpec()], ["u_tt"], 5, 100, seed_for(0, "rows", "vdp_data.npy", 0))
        keys, _ = enumerate_keys(["u"], ["t"], "u_tt", LibrarySpec())
        problem = bank.build(keys, "u_tt", rows, full, valid)
        effective = plan[0]["effective_library_id"]
        with tempfile.TemporaryDirectory(dir=HERE) as name:
            output = Path(name)
            run_job(output, "vdp_data.npy", "u_tt", effective, 0., 0, problem, plan[0], settings, grid, "same_rows")
            with patch("run_size_sweep.fit_problem", side_effect=AssertionError("Must not refit")):
                self.assertEqual(run_job(output, "vdp_data.npy", "u_tt", effective, 0., 0, problem,
                                         plan[0], settings, grid, "same_rows", resume=True), 0)
            write_json(output / "manifest.json", dict(parameters=vars(args), catalog=catalog, settings=settings, grid=grid))
            write_json(output / "plan.json", dict(libraries=plan, summary=summary))
            targets, common, total = aggregate(output)
            self.assertTrue(total["all_jobs_finished"])
            self.assertEqual(targets[0]["completed_fits"], 4)
            self.assertEqual(targets[0]["completed_scores"], 8)
            self.assertTrue(common[0]["complete"])

    def test_aggregation_uses_fixed_setting_across_runs_and_all_system_components(self):
        args = plan_args(["lotka_data.npy"])
        args.runs = 2
        base = build_catalog()["libraries"][0]
        grid = json.loads((HERE / "smoke_grid.json").read_text())
        settings = settings_from_grid(grid)
        catalog = dict(schema_version=1, libraries=[base])
        plan, summary = make_plan(args, catalog, settings, grid)
        with tempfile.TemporaryDirectory(dir=HERE) as name:
            output = Path(name)
            write_json(output / "manifest.json", dict(parameters=vars(args), catalog=catalog, settings=settings, grid=grid))
            write_json(output / "plan.json", dict(libraries=plan, summary=summary))
            for entry in plan:
                for run in range(2):
                    directory = output / "jobs" / job_id("lotka_data.npy", entry["target"], entry["effective_library_id"], 0., run)
                    directory.mkdir(parents=True)
                    write_json(directory / "library.json", dict(nonzero_columns=entry["library_size"], fitted_rows=100,
                                                               row_indices_sha256=f"rows_{run}"))
                    write_json(directory / "status.json", dict(state="completed", completed_fits=4, fit_errors=0, fit_seconds=1.))
                    # Each component can succeed, but different settings succeed for each run/component.
                    successful_index = run if entry["target"] == "x0_t" else run + 2
                    rows = []
                    for index, setting in enumerate(settings):
                        for tolerance in grid["coefficient_tolerances"]:
                            rows.append(dict(setting_id=digest(setting), coefficient_tolerance=tolerance,
                                             status="ok", structure_success=index == successful_index,
                                             law_success=index == successful_index, full_ac_structure_success=None,
                                             hamming=0 if index == successful_index else 1))
                    write_csv(directory / "scores.csv", rows)
            targets, _, _ = aggregate(output)
            self.assertTrue(all(row["oracle_fixed_setting_success_rate"] == .5 for row in targets))
            import csv
            with (output / "recovery_by_dataset.csv").open(encoding="utf-8-sig") as handle:
                result = next(csv.DictReader(handle))
            self.assertEqual(float(result["oracle_fixed_setting_success_rate"]), 0.)

    def test_partial_checkpoint_is_scored_read_only(self):
        setting = dict(optimizer="FROLS", threshold=None, max_iter=3, alpha=1e-5,
                       normalize_columns=False, unbias=True)
        sid = digest(setting)
        keys = [term_key("u"), term_key("u^3")]
        with tempfile.TemporaryDirectory(dir=HERE) as name:
            output = Path(name)
            directory = output / "jobs" / job_id("ac_data.npy", "u_t", "effective", 0., 0)
            directory.mkdir(parents=True)
            write_json(directory / "library.json", dict(feature_keys=keys, nonzero_columns=2))
            record = dict(dataset="ac_data.npy", target="u_t", status="ok", setting_id=sid,
                          nonzero_indices=[0, 1], nonzero_coefficients=[5., -5.], warnings=[], fit_seconds=1.)
            path = directory / "fits.jsonl"
            data = (json.dumps(record) + '\n{"incomplete":').encode()
            path.write_bytes(data)
            result = read_job(output, "ac_data.npy", "u_t", "effective", 0., 0, {(sid, 0.): 0})
            self.assertEqual(result["state"], "partial")
            self.assertEqual(result["fits"], 1)
            self.assertEqual(int(result["success"][0]), 1)
            self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
