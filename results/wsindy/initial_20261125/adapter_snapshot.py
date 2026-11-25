"""Pinned PyWSINDy MSTLS baseline and its raw-data convolutional weak pipeline."""

from itertools import product

import numpy as np

from data.config import sindy_params
from utils.derivatives import compute_derivative_bundle, get_data_axes
from utils.protocols import FIXED_PROTOCOL, NATIVE_PROTOCOL, ProtocolUnavailableError, validate_protocol
from utils.sindy_library import (
    build_crop_slices, build_target_problem, configured_max_deriv_order,
    default_targets, default_variable_names, monomial_name, normalize_data_arrays,
)
from utils.weak_sources import load_source


DATASETS = list(sindy_params)


def fit_mstls(features, target):
    """Execute upstream MSTLS directly, with its stock 50-threshold loss scan."""
    import torch

    cls = load_source("wsindy").WSINDy
    model = cls.__new__(cls)
    model.library = torch.as_tensor(np.asarray(features), dtype=torch.float64)
    model.lhs = torch.as_tensor(np.asarray(target).reshape(-1), dtype=torch.float64)
    model.rescale = False
    coefficients = model.MSTLS().numpy()
    return coefficients, model.Lambda


def canonical_weak_term(variable_names, powers, orders, axis_names):
    """Convert exact conservative identities into the benchmark's term names."""
    name = monomial_name(variable_names, powers)
    derivative_order = sum(orders)
    if derivative_order == 0:
        return name, 1.0
    nonzero_powers = [i for i, power in enumerate(powers) if power]
    suffix = "".join(axis * order for axis, order in zip(axis_names, orders))
    if len(nonzero_powers) == 1:
        index = nonzero_powers[0]
        variable, power = variable_names[index], powers[index]
        if power == 1:
            return f"{variable}_{suffix}", 1.0
        if derivative_order == 1:
            factor = monomial_name([variable], [power - 1])
            return f"{factor} {variable}_{suffix}", float(power)
        if power == 2 and suffix == "xx":
            return f"d_x({variable} {variable}_x)", 2.0
    return f"d_{suffix}({name})", 1.0


def native_problem(arrays, variable_names, target, x, y, z, t, options):
    import torch

    if arrays[0].ndim > 1 and len(arrays) > 1:
        raise ProtocolUnavailableError(
            "The stock conservative WSINDy adapter does not map coupled PDE/advection "
            "terms and the Navier-Stokes divergence constraint to benchmark features."
        )
    axes = get_data_axes(arrays[0], x, y, z, t)
    # Upstream stores spatial axes first and time last.
    permutation = list(range(1, arrays[0].ndim)) + [0]
    axes = [axes[i] for i in permutation]
    grids = [np.asarray(axis[2], dtype=float) for axis in axes]
    for grid in grids:
        steps = np.diff(grid)
        if not len(steps) or np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-12):
            raise ProtocolUnavailableError("WSINDy requires increasing uniformly sampled coordinate grids")
    selected = variable_names.index(target["variable"])
    state_order = [selected] + [i for i in range(len(arrays)) if i != selected]
    states = [torch.tensor(arrays[i].transpose(permutation).copy(), dtype=torch.float64) for i in state_order]
    names = [variable_names[i] for i in state_order]
    axis_names = [axis[0] for axis in axes]
    lhs_orders = tuple(target.get("order", 1) if axis == target.get("axis", "t") else 0 for axis in axis_names)
    identity = (0,) * len(axes)
    derivatives = [lhs_orders, identity]
    # Systems use a polynomial state library; scalar problems also include
    # conservative derivatives up to the common broad complexity limits.
    if len(states) == 1:
        for axis_index, axis in enumerate(axis_names):
            maximum = 2 if axis == "t" else 4
            for order in range(1, maximum + 1):
                orders = tuple(order if i == axis_index else 0 for i in range(len(axes)))
                if orders not in derivatives:
                    derivatives.append(orders)
    degree = 2 if len(states) > 1 else 3
    monomials = [list(powers) for powers in product(range(degree + 1), repeat=len(states))
                 if sum(power > 0 for power in powers) <= 2]
    radii = [min(max(8, len(grid) // 20), (len(grid) - 2) // 2) for grid in grids]
    max_samples = int(options.get("max_samples", 2500))
    per_axis = max(2, int(max_samples ** (1 / len(grids))))
    strides = [max(1, int(np.ceil((len(grid) - 2 * radius) / per_axis)))
               for grid, radius in zip(grids, radii)]
    cls = load_source("wsindy").WSINDy
    model = cls(
        states[0], derivatives, monomials,
        [torch.tensor(grid.copy(), dtype=torch.float64) for grid in grids],
        aux_vars=states[1:], equation_type="ode" if len(axes) == 1 else "pde",
        var_names=names, grid_axis_names=axis_names, verbosity=False,
        rescale=True, subsample_rate=strides, test_fcn_support_radii=radii,
        test_fcn_poly_degrees=[max(6, max(item[i] for item in derivatives) + 2) for i in range(len(axes))],
    )
    model.build_lhs()
    columns, powers, derivs, labels = model.create_default_library()
    model.set_library(columns, powers, derivs, labels)
    coefficients = model.MSTLS().numpy()
    canonical = [canonical_weak_term(names, power, derivatives[deriv], axis_names)
                 for power, deriv in zip(powers, derivs)]
    coefficients *= np.asarray([scale for _, scale in canonical])
    return [name for name, _ in canonical], coefficients, {
        "threshold": model.Lambda, "support_radii": radii,
        "test_function_degrees": model.test_fcn_poly_degrees,
        "query_stride": strides, "query_count": len(model.lhs),
    }


def run_wsindy(data, x, y, z, t, filename, protocol=FIXED_PROTOCOL, native_options=None):
    validate_protocol(protocol)
    config = sindy_params[filename]
    arrays = normalize_data_arrays(data)
    variable_names = config.get("library", {}).get("variable_names", default_variable_names(arrays))
    targets = config.get("targets", default_targets(variable_names))
    options = native_options or {}
    bundle = None
    if protocol == FIXED_PROTOCOL:
        bundle = compute_derivative_bundle(
            data, x, y, z, t, variable_names=variable_names,
            max_orders=configured_max_deriv_order(arrays[0].shape, config),
        )
    result = {"protocol": protocol, "targets": [], "features": [], "coefficients": [],
              "library_sizes": {}, "diagnostics": [], "selection_policy": "upstream_MSTLS_loss"}
    equations = []
    for target in targets:
        name = target.get("name", f"{target['variable']}_{target.get('axis', 't') * target.get('order', 1)}")
        if protocol == NATIVE_PROTOCOL:
            features, coefficients, diagnostics = native_problem(arrays, variable_names, target, x, y, z, t, options)
        else:
            name, matrix, features, values = build_target_problem(
                target, config, bundle, build_crop_slices(arrays[0].shape, config.get("crop", 0)),
                arrays[0].shape, x, t,
            )
            coefficients, threshold = fit_mstls(matrix, values)
            diagnostics = {"threshold": threshold}
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("WSINDy returned non-finite coefficients")
        result["targets"].append(name)
        result["features"].append(features)
        result["coefficients"].append(coefficients.tolist())
        result["library_sizes"][name] = len(features)
        result["diagnostics"].append(diagnostics)
        equations.append(name + " = " + " + ".join(f"{value:.12g} {feature}" for value, feature in zip(coefficients, features) if abs(value) > 1e-12))
    result["model"] = "\n".join(equations)
    return result
