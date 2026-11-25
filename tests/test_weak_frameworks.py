import unittest
from unittest.mock import patch

import numpy as np

from clean_run_metrics import load_framework_module
from run_benchmark import FRAMEWORKS
from utils.protocols import ProtocolUnavailableError
from utils.weak_sources import load_source
from wendy.run import polynomial_feature, prepare_ode, run_wendy
from wsindy.run import canonical_weak_term, coordinate_columns, fit_mstls, native_ns_problem, physical_weak_column


class WeakFrameworkTests(unittest.TestCase):
    def test_frameworks_load_through_common_runner(self):
        for name in ("wsindy", "wendy"):
            self.assertIn(name, FRAMEWORKS)
            self.assertTrue(callable(getattr(load_framework_module(name), f"run_{name}")))

    def test_conservative_terms_preserve_coefficient_scale(self):
        self.assertEqual(canonical_weak_term(["u"], [2], (1, 0), ["x", "t"]), ("u u_x", 2.0))
        self.assertEqual(canonical_weak_term(["u"], [3], (0, 1), ["x", "t"]), ("u^2 u_t", 3.0))
        self.assertEqual(canonical_weak_term(["u"], [2], (2, 0), ["x", "t"]), ("d_x(u u_x)", 2.0))
        # Higher derivatives cannot be relabelled using the first-order chain rule.
        self.assertEqual(canonical_weak_term(["u"], [3], (3, 0), ["x", "t"]), ("d_xxx(u^3)", 1.0))

    def test_wendy_rejects_fixed_and_pde_without_loading_upstream(self):
        with patch("wendy.run.load_source", side_effect=AssertionError("unnecessary upstream load")):
            with self.assertRaisesRegex(ProtocolUnavailableError, "not a sparse optimizer"):
                run_wendy(np.zeros(30), None, None, None, np.arange(30), "lotka_data.npy")
            with self.assertRaisesRegex(ProtocolUnavailableError, "not PDE"):
                run_wendy(np.zeros((30, 20)), np.arange(20), None, None, np.arange(30), "burgers_data.mat", protocol="native")

    def test_wendy_keeps_full_library_and_all_small_coefficients(self):
        class DenseEstimator:
            def __init__(self, features):
                self.features = features

            def fit_IRLS(self, values, times, **kwargs):
                count = sum(len(equation) for equation in self.features)
                self.w_hat = np.full((count, 1), 1e-9)
                self.w_hat_its = [self.w_hat]
                self.pvals_list = [0.5] * 10 + [1e-8]
                self.nits_list = []

        upstream = type("Upstream", (), {"WENDy": DenseEstimator})
        time = np.linspace(0, 2, 50)
        with patch("wendy.run.load_source", return_value=upstream):
            result = run_wendy([np.sin(time), np.cos(time)], None, None, None, time, "lotka_data.npy", protocol="native")
        self.assertEqual(result["library_sizes"], {"x0_t": 14, "x1_t": 14})
        self.assertTrue(np.all(np.asarray(result["coefficients"]) == 1e-9))
        self.assertEqual(result["diagnostics"]["fit_status"], "upstream_residual_test_fallback")
        self.assertIn("fallback", result["selection_policy"])

    def test_wendy_second_order_lift_preserves_clock_and_scores_acceleration(self):
        times = np.linspace(2, 5, 61)
        values, features, labels, scored, diagnostics = prepare_ode(
            [times ** 3], ["u"], [{"variable": "u", "order": 2}], times)
        np.testing.assert_allclose(values[:, 1], 3 * times ** 2, atol=1e-10)
        np.testing.assert_array_equal(values[:, 2], times)
        self.assertEqual(scored, [1])
        self.assertEqual(labels[0], ["u_t"])
        self.assertIn("u^2 u_t", labels[1])
        self.assertIn("u_t sin(2 t)", labels[1])
        i = labels[1].index("u_t sin(2 t)")
        np.testing.assert_allclose(features[1][i](*values.T), values[:, 1] * np.sin(2 * times))
        self.assertIn("correlated", diagnostics["covariance_limitation"])

    def test_wsindy_custom_weak_columns_keep_physical_units(self):
        import torch
        time = torch.linspace(0, 2, 151, dtype=torch.float64)
        space = torch.linspace(1, 3, 151, dtype=torch.float64)
        xx, tt = torch.meshgrid(space, time, indexing="ij")
        values = xx ** 3 * torch.exp(tt / 5)
        cls = load_source("wsindy").WSINDy
        model = cls(values, [(0, 2), (0, 0), (1, 0), (0, 1)], [[0], [1], [2], [3]],
                    [space, time], verbosity=False, rescale=True,
                    test_fcn_support_radii=[15, 15], test_fcn_poly_degrees=[6, 6], subsample_rate=[10, 10])
        columns = dict(coordinate_columns(model, ["x", "t"], {"variable": "u", "order": 2}))
        expected_x = physical_weak_column(model, 3 * xx * torch.exp(tt / 5), (0, 0))
        expected_t = physical_weak_column(model, values / 5 * torch.sin(2 * tt), (0, 0))
        np.testing.assert_allclose(columns["(1/x) u_x"], expected_x, rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(columns["u_t sin(2 t)"], expected_t, rtol=1e-4, atol=1e-8)

    def test_wsindy_spatial_constraint_accepts_coupled_fields(self):
        time = np.linspace(0, 1, 25)
        x = np.linspace(1, 2, 31)
        y = np.linspace(2, 3, 33)
        tt, yy, xx = np.meshgrid(time, y, x, indexing="ij")
        labels, coefficients, diagnostics = native_ns_problem(
            [xx, -yy, 0 * tt], ["u", "v", "p"],
            {"variable": "u", "axis": "x", "order": 1, "feature_tokens": ["v_y"]},
            x, y, None, time, {})
        self.assertEqual(labels, ["v_y"])
        np.testing.assert_allclose(coefficients, [-1], atol=1e-5)
        self.assertGreater(diagnostics["query_count"], 0)

    def test_wendy_features_work_with_numpy_and_symbolic_jacobian(self):
        import sympy

        feature = polynomial_feature((2, 1))
        x, y = sympy.symbols("x y")
        self.assertEqual(sympy.diff(feature(x, y), x), 2 * x * y)
        np.testing.assert_array_equal(feature(np.array([2., 3.]), np.array([4., 5.])), [16., 45.])

    def test_upstream_mstls_recovers_known_sparse_linear_problem(self):
        rng = np.random.RandomState(12)
        matrix = rng.normal(size=(300, 6))
        expected = np.array([0., 1.5, 0., -0.7, 0., 0.])
        coefficients, _ = fit_mstls(matrix, matrix @ expected)
        np.testing.assert_allclose(coefficients, expected, atol=1e-12)

    def test_upstream_weak_time_derivative_sign_and_scaling(self):
        import torch

        cls = load_source("wsindy").WSINDy
        times = torch.linspace(0, 5, 1001, dtype=torch.float64)
        for order, expected_values in ((1, torch.cos(times)), (2, -torch.sin(times))):
            with self.subTest(order=order):
                model = cls(torch.sin(times), [(order,), (0,)], [[1]], [times],
                            equation_type="ode", verbosity=False, rescale=False,
                            test_fcn_support_radii=[50], test_fcn_poly_degrees=[6], subsample_rate=[20])
                model.build_lhs()
                expected = model.compute_weak_poly(expected_values, model.test_fcn_kernels[1], model.spacing)[model.conv_mask]
                np.testing.assert_allclose(model.lhs.numpy(), expected.numpy(), rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
