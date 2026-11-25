"""WENDy parameter estimation with the full candidate library, without oracle support."""

from itertools import product

import numpy as np

from data.config import sindy_params
from utils.protocols import FIXED_PROTOCOL, ProtocolUnavailableError, validate_protocol
from utils.sindy_library import default_targets, default_variable_names, monomial_name, normalize_data_arrays
from utils.weak_sources import load_source


DATASETS = list(sindy_params)


def polynomial_feature(powers):
    # The same callable must work with NumPy arrays and SymPy symbols, because
    # upstream differentiates every feature symbolically for its EIV covariance.
    def feature(*state):
        value = 1
        for variable, power in zip(state, powers):
            if power:
                value = value * variable ** power
        return value + 0 * state[0]
    return feature


def run_wendy(data, x, y, z, t, filename, protocol=FIXED_PROTOCOL, native_options=None):
    validate_protocol(protocol)
    if protocol == FIXED_PROTOCOL:
        raise ProtocolUnavailableError(
            "WENDy fits weak-form ODE models from raw observations; it is not a "
            "sparse optimizer for precomputed fixed derivatives. Use --protocol native."
        )
    arrays = normalize_data_arrays(data)
    config = sindy_params[filename]
    names = config.get("library", {}).get("variable_names", default_variable_names(arrays))
    targets = config.get("targets", default_targets(names))
    if arrays[0].ndim != 1:
        raise ProtocolUnavailableError("Upstream pyWENDy supports ODE parameter estimation, not PDE discovery")
    if any(target.get("axis", "t") != "t" or target.get("order", 1) != 1 for target in targets):
        raise ProtocolUnavailableError("Upstream pyWENDy requires first-order ODE targets; no observed velocity is available for second-order scalar ODEs")
    if len(targets) != len(arrays):
        raise ProtocolUnavailableError("WENDy requires an evolution equation for each observed state")
    times = np.asarray(t, dtype=float) - float(t[0])
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-12):
        raise ProtocolUnavailableError("Upstream pyWENDy requires increasing uniformly sampled time")
    degree = 2 if len(arrays) > 1 else 3
    powers = [power for power in product(range(degree + 1), repeat=len(arrays))
              if sum(item > 0 for item in power) <= 2]
    feature_names = [monomial_name(names, power) for power in powers]
    features = [[polynomial_feature(power) for power in powers] for _ in arrays]
    radius = min(max(8, len(times) // 20), (len(times) - 2) // 2)
    # Limit dense EIV covariance size via test-function spacing, retaining every
    # observation and using the same policy for all datasets.
    gap = max(1, int(np.ceil((len(times) - 2 * radius) / 100)))
    model = load_source("wendy").WENDy(features)
    values = np.column_stack(arrays)
    model.fit_IRLS(values, times, radius=[radius], gap=gap, p=6)
    coefficients = np.asarray(model.w_hat).reshape(len(arrays), len(powers))
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("WENDy returned non-finite coefficients")
    pvalues = [float(value) for value in getattr(model, "pvals_list", [])]
    fit_status = "returned_iterate"
    if len(pvalues) > getattr(model, "check_pval_it", 10) and pvalues[-1] < getattr(model, "pvalmin", 1e-4):
        # The unmodified upstream implementation returns the iterate with the
        # largest residual-normality p-value when this diagnostic rejects IRLS.
        fit_status = "upstream_residual_test_fallback"
    elif not hasattr(model, "nits_list"):
        fit_status = "upstream_initial_guess_returned"
    elif len(model.w_hat_its) >= getattr(model, "max_iter", 100):
        fit_status = "upstream_iteration_limit"
    target_names = [target.get("name", f"{target['variable']}_t") for target in targets]
    return {
        "protocol": protocol, "targets": target_names,
        "features": [feature_names[:] for _ in arrays], "coefficients": coefficients.tolist(),
        "library_sizes": {name: len(feature_names) for name in target_names},
        "selection_policy": "upstream_IRLS_full_library_no_sparsification:" + fit_status,
        "diagnostics": {"radius": radius, "gap": gap, "test_function_degree": 6,
                        "iterations": max(0, len(pvalues) - 1), "fit_status": fit_status,
                        "residual_normality_pvalues": pvalues},
        "model": "\n".join(name + " = " + " + ".join(f"{value:.12g} {feature}" for value, feature in zip(row, feature_names)) for name, row in zip(target_names, coefficients)),
    }
