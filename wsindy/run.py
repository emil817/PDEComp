"""Pinned PyWSINDy MSTLS baseline and its raw-data convolutional weak pipeline."""

from itertools import product

import numpy as np
from scipy.interpolate import CubicSpline

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


def physical_weak_column(model, values, orders):
    """Integrate a physical feature in the same scaled units as upstream LHS."""
    index = model.derivatives.index(tuple(orders))
    column = model.compute_weak_poly(values, model.test_fcn_kernels[index], model.spacing)[model.conv_mask]
    if model.rescale:
        scales = np.asarray(model.x_scales + [model.t_scale])
        factor = model.u_scale * np.prod(scales ** (np.asarray(orders) - np.asarray(model.derivatives[0])))
        column *= float(factor * np.prod(scales))
    return column


def coordinate_columns(model, axis_names, target):
    """Custom weak columns; variable coefficients use integration by parts."""
    import torch

    zero = (0,) * len(axis_names)
    coordinates = {}
    for i, name in enumerate(axis_names):
        shape = [1] * len(axis_names)
        shape[i] = len(model.grid[i])
        coordinates[name] = model.grid[i].reshape(shape).expand_as(model.U)
    time = coordinates["t"]
    fields = [("t", time), ("sin(t)", torch.sin(time)), ("cos(t)", torch.cos(time)),
              ("sin(2 t)", torch.sin(2 * time)), ("cos(2 t)", torch.cos(2 * time))]
    if "x" in coordinates:
        space = coordinates["x"]
        fields.extend([("x", space), ("sin(x)", torch.sin(space)),
                       ("sin(x) cos(t)", torch.sin(space) * torch.cos(time)),
                       ("cos(x) sin(t)", torch.cos(space) * torch.sin(time))])
    columns = [(name, physical_weak_column(model, field, zero)) for name, field in fields]
    if target.get("order", 1) > 1:
        dt = tuple(int(axis == "t") for axis in axis_names)
        column = physical_weak_column(model, model.U * torch.sin(2 * time), dt)
        column -= physical_weak_column(model, 2 * model.U * torch.cos(2 * time), zero)
        columns.append(("u_t sin(2 t)", column))
    if "x" in coordinates:
        space = coordinates["x"]
        # A reciprocal coefficient is smooth only on domains avoiding zero.
        if bool(torch.all(space > 0) or torch.all(space < 0)):
            dx = tuple(int(axis == "x") for axis in axis_names)
            columns.append(("(1/x) u", physical_weak_column(model, model.U / space, zero)))
            column = physical_weak_column(model, model.U / space, dx)
            column += physical_weak_column(model, model.U / space ** 2, zero)
            columns.append(("(1/x) u_x", column))
    return columns


def native_ns_problem(arrays, variable_names, target, x, y, z, t, options):
    """Coupled PDE custom library through upstream weak convolution and MSTLS.

    Nonconservative velocity products use CubicSpline spatial derivatives of
    the observations. All linear spatial and target derivatives are weak.
    """
    import torch

    axes = get_data_axes(arrays[0], x, y, z, t)
    permutation = list(range(1, arrays[0].ndim)) + [0]
    axes = [axes[i] for i in permutation]
    grids = [np.asarray(axis[2], dtype=float) for axis in axes]
    for grid in grids:
        steps = np.diff(grid)
        if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-12):
            raise ProtocolUnavailableError("WSINDy requires increasing uniformly sampled coordinate grids")
    axis_names = [axis[0] for axis in axes]
    fields = {name: torch.tensor(array.transpose(permutation).copy(), dtype=torch.float64)
              for name, array in zip(variable_names, arrays)}
    zero = (0,) * len(axes)
    orders = lambda axis, order: tuple(order if name == axis else 0 for name in axis_names)
    lhs = orders(target.get("axis", "t"), target.get("order", 1))
    derivatives = list(dict.fromkeys([lhs, zero, orders("x", 1), orders("y", 1), orders("x", 2), orders("y", 2)]))
    radii = [min(max(8, len(grid) // 20), (len(grid) - 2) // 2) for grid in grids]
    per_axis = max(2, int(int(options.get("max_samples", 2500)) ** (1 / len(grids))))
    strides = [max(1, int(np.ceil((len(grid) - 2 * radius) / per_axis))) for grid, radius in zip(grids, radii)]
    model = load_source("wsindy").WSINDy(fields[target["variable"]], derivatives, [[1]],
        [torch.tensor(grid.copy(), dtype=torch.float64) for grid in grids],
        var_names=[target["variable"]], grid_axis_names=axis_names,
        verbosity=False, rescale=False, subsample_rate=strides,
        test_fcn_support_radii=radii, test_fcn_poly_degrees=[6] * len(grids))
    model.build_lhs()
    if "feature_tokens" in target:
        labels = target["feature_tokens"]
        columns = [physical_weak_column(model, fields[name.split("_")[0]], orders(name.split("_")[1], 1)) for name in labels]
    else:
        labels = ["1", "u", "v"]
        columns = [physical_weak_column(model, field, zero) for field in [torch.ones_like(model.U), fields["u"], fields["v"]]]
        for axis in ("x", "y"):
            for order in (1, 2):
                labels.append("p_" + axis * order)
                columns.append(physical_weak_column(model, fields["p"], orders(axis, order)))
        for velocity, axis in zip(("u", "v"), ("x", "y")):
            index = axis_names.index(axis)
            derivative = CubicSpline(grids[index], model.U.numpy(), axis=index)(grids[index], 1)
            labels.append(f"{velocity} {target['variable']}_{axis}")
            columns.append(physical_weak_column(model, fields[velocity] * torch.from_numpy(derivative), zero))
        for velocity in ("u", "v"):
            labels.append(f"({velocity}_xx + {velocity}_yy)")
            columns.append(sum(physical_weak_column(model, fields[velocity], orders(axis, 2)) for axis in ("x", "y")))
    model.set_library(columns, [None] * len(columns), [None] * len(columns), labels)
    coefficients = model.MSTLS().numpy()
    return labels, coefficients, {"threshold": model.Lambda, "support_radii": radii,
        "query_stride": strides, "query_count": len(model.lhs),
        "advection_preprocessing": "scipy.CubicSpline spatial derivative, not-a-knot; upstream weak integration",
        "library": "coupled velocity/pressure custom columns; no fixed derivative bundle"}


def native_problem(arrays, variable_names, target, x, y, z, t, options):
    import torch

    if arrays[0].ndim > 1 and len(arrays) > 1:
        return native_ns_problem(arrays, variable_names, target, x, y, z, t, options)
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
    extras = coordinate_columns(model, axis_names, target) if len(states) == 1 else []
    for label, column in extras:
        columns.append(column)
        labels.append(label)
        powers.append(None)
        derivs.append(None)
    model.set_library(columns, powers, derivs, labels)
    coefficients = model.MSTLS().numpy()
    canonical = [canonical_weak_term(names, power, derivatives[deriv], axis_names) if power is not None else (label, 1.0)
                 for power, deriv, label in zip(powers, derivs, labels)]
    coefficients *= np.asarray([scale for _, scale in canonical])
    return [name for name, _ in canonical], coefficients, {
        "threshold": model.Lambda, "support_radii": radii,
        "test_function_degrees": model.test_fcn_poly_degrees,
        "query_stride": strides, "query_count": len(model.lhs),
        "coordinate_columns": [name for name, _ in extras],
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
