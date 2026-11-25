"""WENDy parameter estimation with the full candidate library, without oracle support."""

from itertools import product

import numpy as np
from scipy.interpolate import CubicSpline

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


def time_feature(kind, velocity_index=None):
    """Callable for upstream's NumPy evaluation and symbolic Jacobian."""
    def feature(*state):
        import sympy
        clock = state[-1]
        if kind == "t":
            return clock
        frequency = 2 if "2" in kind else 1
        trig = "sin" if "sin" in kind else "cos"
        function = getattr(sympy if isinstance(clock, sympy.Basic) else np, trig)
        value = function(frequency * clock)
        return value if velocity_index is None else state[velocity_index] * value
    return feature


def prepare_ode(arrays, names, targets, times):
    """Lift second-order ODEs and explicit time into a first-order observed state.

    Only requested acceleration/evolution equations are scored. Kinematic and
    clock equations are auxiliary fits, with their coefficients recorded.
    """
    orders = [target.get("order", 1) for target in targets]
    if any(target.get("axis", "t") != "t" for target in targets):
        raise ProtocolUnavailableError("pyWENDy requires time-evolution ODE targets")
    second_order = len(arrays) == 1 and orders == [2]
    if not second_order and (any(order != 1 for order in orders) or len(targets) != len(arrays)):
        raise ProtocolUnavailableError("Supported ODE lifts are first-order systems and scalar second-order equations")
    state = list(arrays)
    state_names = list(names)
    if second_order:
        state.append(CubicSpline(times, arrays[0])(times, 1))
        state_names.append(names[0] + "_t")
    degree = 2 if len(state) > 1 else 3
    powers = [power for power in product(range(degree + 1), repeat=len(state))
              if sum(item > 0 for item in power) <= 2]
    library_names = [monomial_name(state_names, power) for power in powers]
    library = [polynomial_feature(power) for power in powers]
    for name in ("t", "sin(t)", "cos(t)", "sin(2 t)", "cos(2 t)"):
        library_names.append(name)
        library.append(time_feature(name))
    if second_order:
        library_names.append(state_names[1] + " sin(2 t)")
        library.append(time_feature("sin(2 t)", velocity_index=1))
        # u' = velocity is the definition of the augmented state.
        features = [[polynomial_feature((0, 1))], library]
        labels = [[state_names[1]], library_names]
        scored_indexes = [1]
    else:
        features = [library[:] for _ in state]
        labels = [library_names[:] for _ in state]
        scored_indexes = [names.index(target["variable"]) for target in targets]
    # Exact observed clock supports non-autonomous features without replacing
    # upstream's IRLS. Its derivative equation is fitted as an auxiliary row.
    state.append(times)
    state_names.append("clock")
    features.append([polynomial_feature((0,) * len(state))])
    labels.append(["1"])
    diagnostics = {"state_names": state_names, "scored_state_indexes": scored_indexes,
                   "velocity_preprocessing": "scipy.CubicSpline derivative, not-a-knot" if second_order else "none",
                   "covariance_limitation": "Derived velocity errors are correlated with position; upstream uses its diagonal observation-noise model." if second_order else "none"}
    return np.column_stack(state), features, labels, scored_indexes, diagnostics


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
    times = np.asarray(t, dtype=float) - float(t[0])
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-12):
        raise ProtocolUnavailableError("Upstream pyWENDy requires increasing uniformly sampled time")
    values, features, labels, scored_indexes, preprocessing = prepare_ode(arrays, names, targets, np.asarray(t, dtype=float))
    radius = min(max(8, len(times) // 20), (len(times) - 2) // 2)
    # Limit dense EIV covariance size via test-function spacing, retaining every
    # observation and using the same policy for all datasets.
    gap = max(1, int(np.ceil((len(times) - 2 * radius) / 100)))
    model = load_source("wendy").WENDy(features)
    model.fit_IRLS(values, times, radius=[radius], gap=gap, p=6)
    flat_coefficients = np.asarray(model.w_hat).reshape(-1)
    if not np.all(np.isfinite(flat_coefficients)):
        raise ValueError("WENDy returned non-finite coefficients")
    offsets = np.cumsum([0] + [len(equation) for equation in features])
    all_coefficients = [flat_coefficients[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    coefficients = [all_coefficients[index] for index in scored_indexes]
    feature_names = [labels[index] for index in scored_indexes]
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
        "features": feature_names, "coefficients": [row.tolist() for row in coefficients],
        "library_sizes": {name: len(features) for name, features in zip(target_names, feature_names)},
        "selection_policy": "upstream_IRLS_full_library_no_sparsification:" + fit_status,
        "diagnostics": {"radius": radius, "gap": gap, "test_function_degree": 6,
                        "iterations": max(0, len(pvalues) - 1), "fit_status": fit_status,
                        "residual_normality_pvalues": pvalues, **preprocessing,
                        "auxiliary_equations": [{"state": preprocessing["state_names"][i], "features": labels[i], "coefficients": row.tolist()} for i, row in enumerate(all_coefficients) if i not in scored_indexes]},
        "model": "\n".join(name + " = " + " + ".join(f"{value:.12g} {feature}" for value, feature in zip(row, terms)) for name, row, terms in zip(target_names, coefficients, feature_names)),
    }
