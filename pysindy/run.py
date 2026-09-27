import json
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pysindy as ps

sys.path.append(str(Path().absolute()))

from data.config import (
    NATIVE_PYSINDY_DATASET_OVERRIDES,
    NATIVE_PYSINDY_DEFAULTS,
    sindy_params,
)
from utils import sindy_library
from utils.dataloader import load_data
from utils.derivatives import (
    compute_derivative_bundle,
    get_data_axes,
)
from utils.protocols import FIXED_PROTOCOL, NATIVE_PROTOCOL, validate_protocol


RESULTS_DIR = Path("results/pysindy")
DATASETS = [
    "ode_data.npy",
    "vdp_data.npy",
    "lorenz_data.npy",
    "lotka_data.npy",
    "burgers_data.mat",
    "ac_data.npy",
    "kdv_data.mat",
    "kdv_periodic_data.npy",
    "wave_data.csv",
    "pde_divide_data.npy",
    "pde_compound_data.npy",
    "ns_data.mat",
    "ks_data.mat",
    "burgers_sln_100_data.csv",
    "ODE_simple_discovery",
]


def save_combined_results(results):
    """Save all dataset results into one timestamped JSON file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as handle:
        json.dump([results], handle, indent=2)


def build_optimizer(opt_config):
    """Create the sparse optimizer described by the dataset config."""

    opt_type = opt_config.get("type", "STLSQ")

    if opt_type == "STLSQ":
        return ps.STLSQ(
            threshold=opt_config.get("threshold", 0.1),
            alpha=opt_config.get("alpha", 0.05),
            normalize_columns=opt_config.get("normalize_columns", False),
        )

    if opt_type == "SR3":
        return ps.SR3(
            threshold=opt_config.get("threshold", 0.1),
            max_iter=opt_config.get("max_iter", 30),
            tol=opt_config.get("tol", 1e-5),
            normalize_columns=opt_config.get("normalize_columns", False),
            thresholder=opt_config.get("thresholder", opt_config.get("regularizer", "L0")),
            nu=opt_config.get("nu", 1.0),
        )

    if opt_type == "FROLS":
        return ps.FROLS(
            max_iter=opt_config.get("max_iter", 10),
            alpha=opt_config.get("alpha", 0.05),
            normalize_columns=opt_config.get("normalize_columns", False),
        )

    raise ValueError(f"Unknown optimizer type: {opt_type}")


def print_equation(target_name, feature_names, coefficients, precision=4):
    """Print an equation using fitted coefficients."""

    active_terms = []
    for coef, feature in zip(np.ravel(coefficients), feature_names):
        if abs(coef) > 1e-12:
            active_terms.append(f"{coef:.{precision}f} {feature}")

    rhs = " + ".join(active_terms).replace("+ -", "- ")
    if not rhs:
        rhs = "0"
    print(f"{target_name} = {rhs}")


def fit_sparse_system(feature_matrix, target_vector, feature_names, target_name, filename, opt_config):
    """Fit one sparse regression problem and return results."""

    optimizer = build_optimizer(opt_config)
    optimizer.fit(feature_matrix, target_vector)

    coefficients = np.asarray(optimizer.coef_)
    coefficient_tol = opt_config.get("coefficient_tol", 0.0)
    relative_tol = opt_config.get("coefficient_relative_tol", 0.0)
    cutoff = float(coefficient_tol)
    if relative_tol > 0 and coefficients.size:
        cutoff = max(cutoff, float(relative_tol) * np.max(np.abs(coefficients)))
    if cutoff > 0:
        coefficients[np.abs(coefficients) < cutoff] = 0.0
    if coefficients.ndim == 1:
        coefficients = coefficients[np.newaxis, :]

    print_equation(target_name, feature_names, coefficients[0])
    print()

    return {
        "dataset": filename.split(".")[0],
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }


def pysindy_derivative(values, axes, axis_name, derivative_order, diff_config):
    """Differentiate one field with PySINDy's public differentiation API."""

    axis_by_name = {name: (axis, grid) for name, axis, grid in axes}
    if axis_name not in axis_by_name:
        raise ValueError(f"Axis {axis_name!r} is unavailable")
    axis, grid = axis_by_name[axis_name]
    differentiator = ps.FiniteDifference(
        order=int(diff_config.get("order", 2)),
        d=int(derivative_order),
        axis=axis,
        periodic=bool(diff_config.get("periodic", False)) and axis_name != "t",
    )
    return np.asarray(differentiator(np.asarray(values, dtype=float), grid), dtype=float)


def native_spatial_grid(axes):
    """Return a PySINDy spatial grid in the same axis order as native data."""

    spatial_axes = [(name, grid) for name, _, grid in axes if name != "t"]
    if not spatial_axes:
        return None, []
    names = [name for name, _ in spatial_axes]
    grids = [np.asarray(grid, dtype=float) for _, grid in spatial_axes]
    if len(grids) == 1:
        return grids[0], names
    mesh = np.meshgrid(*grids, indexing="ij")
    return np.stack(mesh, axis=-1), names


def native_state(
    data_arrays,
    variable_names,
    axes,
    target,
    diff_config,
    include_time,
    include_lower_time_derivatives=True,
):
    """Build raw framework input, adding only metadata-required state variables."""

    time_axis = next(axis for name, axis, _ in axes if name == "t")
    channels = [np.asarray(values, dtype=float) for values in data_arrays]
    names = list(variable_names)

    target_order = int(target.get("order", 1))
    target_axis = target.get("axis", "t")
    if include_lower_time_derivatives and target_axis == "t" and target_order > 1:
        variable_index = variable_names.index(target["variable"])
        values = data_arrays[variable_index]
        for order in range(1, target_order):
            channels.append(pysindy_derivative(values, axes, "t", order, diff_config))
            names.append(f"{target['variable']}_{'t' * order}")

    if include_time:
        time_grid = next(grid for name, _, grid in axes if name == "t")
        shape = [1] * data_arrays[0].ndim
        shape[time_axis] = len(time_grid)
        time_values = np.broadcast_to(
            np.asarray(time_grid, dtype=float).reshape(shape),
            data_arrays[0].shape,
        )
        channels.append(time_values)
        names.append("t")

    moved = [np.moveaxis(values, time_axis, -1) for values in channels]
    return np.stack(moved, axis=-1), names


def native_ode_library(dynamic_variables, config):
    """Construct a compact ODE library without tensoring every feature."""

    polynomial = ps.PolynomialLibrary(
        degree=int(config.get("polynomial_degree", 3)), include_bias=True
    )
    time_functions = ps.CustomLibrary(
        library_functions=[
            lambda value: value,
            lambda value: np.sin(value),
            lambda value: np.cos(value),
        ],
        function_names=[
            lambda name: name,
            lambda name: f"sin({name})",
            lambda name: f"cos({name})",
        ],
    )
    state_linear = ps.PolynomialLibrary(degree=1, include_bias=False)
    if dynamic_variables == 1:
        second_harmonic = ps.CustomLibrary(
            library_functions=[
                lambda value: np.sin(2.0 * value),
                lambda value: np.cos(2.0 * value),
            ],
            function_names=[
                lambda name: f"sin(2 {name})",
                lambda name: f"cos(2 {name})",
            ],
        )
        excluded_libraries = [2]
    else:
        second_harmonic = ps.CustomLibrary(
            library_functions=[lambda value: np.sin(2.0 * value)],
            function_names=[lambda name: f"sin(2 {name})"],
        )
        excluded_libraries = [2, 3]
    time_index = dynamic_variables
    return ps.GeneralizedLibrary(
        [polynomial, time_functions, state_linear, second_harmonic],
        tensor_array=[[0, 0, 1, 1]],
        inputs_per_library=[
            list(range(dynamic_variables)),
            [time_index],
            list(range(dynamic_variables)),
            [time_index],
        ],
        exclude_libraries=excluded_libraries,
    )


def native_system_library(dynamic_variables):
    """Build the fixed-scale system space with standard PySINDy libraries."""

    polynomial = ps.PolynomialLibrary(degree=2, include_bias=True)
    pair_powers = ps.CustomLibrary(
        library_functions=[
            lambda first, second: first ** 2 * second,
            lambda first, second: first * second ** 2,
            lambda first, second: first ** 2 * second ** 2,
        ],
        function_names=[
            lambda first, second: f"{first}^2 {second}",
            lambda first, second: f"{first} {second}^2",
            lambda first, second: f"{first}^2 {second}^2",
        ],
    )
    time = ps.CustomLibrary(
        library_functions=[lambda value: value],
        function_names=[lambda name: name],
    )
    dynamic_inputs = list(range(dynamic_variables))
    return ps.GeneralizedLibrary(
        [polynomial, pair_powers, time],
        inputs_per_library=[
            dynamic_inputs,
            dynamic_inputs,
            [dynamic_variables],
        ],
    )


def native_pde_library(spatial_grid, spatial_order, config):
    """Construct PySINDy's standard strong-form PDE library."""

    function_library = ps.PolynomialLibrary(
        degree=int(config.get("polynomial_degree", 3)), include_bias=False
    )
    return ps.PDELibrary(
        function_library=function_library,
        derivative_order=int(spatial_order),
        spatial_grid=spatial_grid,
        include_bias=True,
        include_interaction=True,
        differentiation_method=ps.FiniteDifference,
        diff_kwargs={"order": int(config.get("differentiation", {}).get("order", 2))},
    )


def broadcast_native_axis(grid, shape, axis):
    reshape = [1] * len(shape)
    reshape[axis] = len(grid)
    return np.broadcast_to(np.asarray(grid, dtype=float).reshape(reshape), shape)


def native_scalar_pde_extras(values, axes, target, diff_config):
    """Return the common coordinate terms used by all scalar PDE tasks."""

    shape = values.shape
    axis_map = {name: (axis, grid) for name, axis, grid in axes}
    time_axis, time_grid = axis_map["t"]
    x_axis, x_grid = axis_map["x"]
    time_values = broadcast_native_axis(time_grid, shape, time_axis)
    x_values = broadcast_native_axis(x_grid, shape, x_axis)
    u_x = pysindy_derivative(values, axes, "x", 1, diff_config)
    compound = pysindy_derivative(values * u_x, axes, "x", 1, diff_config)
    inverse_x_u = np.divide(
        values,
        x_values,
        out=np.zeros_like(values, dtype=float),
        where=np.abs(x_values) > 1e-12,
    )
    inverse_x_u_x = np.divide(
        u_x,
        x_values,
        out=np.zeros_like(values, dtype=float),
        where=np.abs(x_values) > 1e-12,
    )
    specs = [
        ("t", time_values),
        ("x", x_values),
        ("sin(x)", np.sin(x_values)),
        ("cos(t)", np.cos(time_values)),
        ("sin(x) cos(t)", np.sin(x_values) * np.cos(time_values)),
        ("cos(x) sin(t)", np.cos(x_values) * np.sin(time_values)),
        ("(1/x) u", inverse_x_u),
        ("(1/x) u_x", inverse_x_u_x),
        ("d_x(u u_x)", compound),
    ]
    target_order = int(target.get("order", 1))
    temporal_order = 2 if target.get("axis", "t") == "t" and target_order == 1 else 1
    temporal_suffix = "t" * temporal_order
    temporal_derivative = pysindy_derivative(
        values, axes, "t", temporal_order, diff_config
    )
    specs.extend([
        (f"u_{temporal_suffix}", temporal_derivative),
        (f"u u_{temporal_suffix}", values * temporal_derivative),
        (f"u^2 u_{temporal_suffix}", values ** 2 * temporal_derivative),
        (f"u^3 u_{temporal_suffix}", values ** 3 * temporal_derivative),
    ])
    return specs


def compact_navier_stokes_features(
    data_arrays, variable_names, axes, target, diff_config
):
    """Build a compact, equation-agnostic incompressible-flow library."""

    fields = dict(zip(variable_names, data_arrays))
    velocity_names = variable_names[:2]
    pressure_name = variable_names[2] if len(variable_names) > 2 else None
    specs = [("1", np.ones_like(data_arrays[0], dtype=float))]
    for name in velocity_names:
        specs.extend([(name, fields[name]), (f"{name}^2", fields[name] ** 2)])

    available_spatial_axes = {name for name, _, _ in axes if name != "t"}
    spatial_axes = [
        name for name in ("x", "y", "z") if name in available_spatial_axes
    ]
    if target.get("axis", "t") != "t":
        specs = []
        for variable_name in variable_names:
            if variable_name == target["variable"]:
                continue
            for axis_name in spatial_axes:
                specs.append((
                    f"{variable_name}_{axis_name}",
                    pysindy_derivative(
                        fields[variable_name], axes, axis_name, 1, diff_config
                    ),
                ))
        return specs

    target_variable = target["variable"]
    for axis_name in spatial_axes:
        if pressure_name is not None:
            specs.append((
                f"{pressure_name}_{axis_name}",
                pysindy_derivative(
                    fields[pressure_name], axes, axis_name, 1, diff_config
                ),
            ))

    for velocity_name in velocity_names:
        laplacian = sum(
            pysindy_derivative(
                fields[velocity_name], axes, axis_name, 2, diff_config
            )
            for axis_name in spatial_axes
        )
        suffix = " + ".join(
            f"{velocity_name}_{axis_name}{axis_name}"
            for axis_name in spatial_axes
        )
        specs.append((f"({suffix})", laplacian))

    for velocity_name, axis_name in zip(velocity_names, spatial_axes):
        target_derivative = pysindy_derivative(
            fields[target_variable], axes, axis_name, 1, diff_config
        )
        specs.append((
            f"{velocity_name} {target_variable}_{axis_name}",
            fields[velocity_name] * target_derivative,
        ))
    return specs


def native_trim_slices(shape, trim):
    if trim <= 0:
        return tuple(slice(None) for _ in shape)
    return tuple(
        slice(trim, size - trim) if size > 2 * trim else slice(None)
        for size in shape
    )


def canonical_native_feature_name(name, spatial_axes):
    """Translate PySINDy's coordinate-number suffixes to benchmark axis names."""

    result = str(name)
    axis_by_digit = {
        str(index): axis_name for index, axis_name in enumerate(spatial_axes, start=1)
    }

    def derivative_suffix(match):
        digits = match.group(1)
        if not all(digit in axis_by_digit for digit in digits):
            return match.group(0)
        return "_" + "".join(axis_by_digit[digit] for digit in digits)

    result = re.sub(r"_([1-9]+)", derivative_suffix, result)
    result = re.sub(r"^(u(?:\^\d+)?)(u_.+)$", r"\1 \2", result)
    result = result.replace("sin(1 ", "sin(").replace("cos(1 ", "cos(")
    return result


def build_native_problem(
    data_arrays, variable_names, target, x, y, z, t, max_orders, config
):
    """Build one compact problem with PySINDy differentiation and libraries."""

    axes = get_data_axes(data_arrays[0], x, y, z, t)
    diff_config = config.get("differentiation", {})
    time_axis = next(axis for name, axis, _ in axes if name == "t")
    target_values = pysindy_derivative(
        data_arrays[variable_names.index(target["variable"])],
        axes,
        target.get("axis", "t"),
        target.get("order", 1),
        diff_config,
    )

    spatial_grid, spatial_axes = native_spatial_grid(axes)
    if spatial_grid is None:
        is_system = len(data_arrays) > 1
        if is_system:
            state, state_names = native_state(
                data_arrays,
                variable_names,
                axes,
                target,
                diff_config,
                include_time=True,
                include_lower_time_derivatives=False,
            )
            library = native_system_library(len(variable_names))
            transformed = np.asarray(library.fit_transform(state), dtype=float)
            feature_names = library.get_feature_names(state_names)
        else:
            state, state_names = native_state(
                data_arrays,
                variable_names,
                axes,
                target,
                diff_config,
                include_time=True,
            )
            dynamic_variables = len(state_names) - 1
            library = native_ode_library(dynamic_variables, config)
            transformed = np.asarray(library.fit_transform(state), dtype=float)
            feature_names = library.get_feature_names(state_names)
    elif len(data_arrays) > 1:
        feature_specs = compact_navier_stokes_features(
            data_arrays, variable_names, axes, target, diff_config
        )
        raw_features = np.stack(
            [np.moveaxis(values, time_axis, -1) for _, values in feature_specs],
            axis=-1,
        )
        library = ps.IdentityLibrary()
        transformed = np.asarray(library.fit_transform(raw_features), dtype=float)
        feature_names = library.get_feature_names(
            [name for name, _ in feature_specs]
        )
    else:
        state, state_names = native_state(
            data_arrays,
            variable_names,
            axes,
            target,
            diff_config,
            include_time=False,
            include_lower_time_derivatives=False,
        )
        spatial_orders = [
            order for (axis_name, _, _), order in zip(axes, max_orders) if axis_name != "t"
        ]
        extras = native_scalar_pde_extras(data_arrays[0], axes, target, diff_config)
        extra_state = np.stack(
            [np.moveaxis(values, time_axis, -1) for _, values in extras],
            axis=-1,
        )
        combined_state = np.concatenate([state, extra_state], axis=-1)
        pde_library = native_pde_library(
            spatial_grid, max(spatial_orders), config
        )
        library = ps.GeneralizedLibrary(
            [pde_library, ps.IdentityLibrary()],
            inputs_per_library=[
                [0],
                list(range(1, 1 + len(extras))),
            ],
        )
        combined_names = state_names + [name for name, _ in extras]
        transformed = np.asarray(library.fit_transform(combined_state), dtype=float)
        feature_names = library.get_feature_names(combined_names)

    trim = int(config.get("boundary_trim", 0))
    trim_slices = native_trim_slices(transformed.shape[:-1], trim)
    features = transformed[trim_slices].reshape(-1, transformed.shape[-1])
    moved_target = np.moveaxis(np.asarray(target_values, dtype=float), time_axis, -1)
    target_vector = moved_target[trim_slices].reshape(-1)
    feature_names = [
        canonical_native_feature_name(name, spatial_axes)
        for name in feature_names
    ]
    finite = np.isfinite(target_vector) & np.all(np.isfinite(features), axis=1)
    return (
        target.get("name", f"{target['variable']}_t"),
        features[finite],
        feature_names,
        target_vector[finite],
    )


def run_sindy(data, x, y, z, t, filename, protocol=FIXED_PROTOCOL, native_options=None):
    """Run PySINDy under the fixed-library or framework-native protocol."""

    validate_protocol(protocol)
    params = sindy_params[filename]
    
    data_arrays = sindy_library.normalize_data_arrays(data)
    lib_config = params.get("library", {})
    variable_names = lib_config.get("variable_names", sindy_library.default_variable_names(data_arrays))
    targets = params.get("targets", sindy_library.default_targets(variable_names))
    max_orders = sindy_library.configured_max_deriv_order(data_arrays[0].shape, params)
    if protocol == NATIVE_PROTOCOL:
        dataset_options = NATIVE_PYSINDY_DATASET_OVERRIDES.get(filename, {})
        native_config = {
            **NATIVE_PYSINDY_DEFAULTS,
            **dataset_options,
            **(native_options or {}),
        }
        optimizer_config = {
            **NATIVE_PYSINDY_DEFAULTS["optimizer"],
            **dataset_options.get("optimizer", {}),
            **native_config.get("optimizer", {}),
        }
    else:
        derivatives = compute_derivative_bundle(
            data_arrays if len(data_arrays) > 1 else data_arrays[0],
            x=x,
            y=y,
            z=z,
            t=t,
            variable_names=variable_names,
            max_orders=max_orders,
        )
        optimizer_config = params["optimizer"]
        crop_slices = sindy_library.build_crop_slices(
            data_arrays[0].shape, params.get("crop", 0)
        )

    results = []
    feature_names_by_target = []
    for target in targets:
        if protocol == NATIVE_PROTOCOL:
            target_name, features, feature_names, target_values = build_native_problem(
                data_arrays,
                variable_names,
                target,
                x,
                y,
                z,
                t,
                max_orders,
                native_config,
            )
        else:
            target_name, features, feature_names, target_values = sindy_library.build_target_problem(
                target,
                params,
                derivatives,
                crop_slices,
                data_arrays[0].shape,
                x,
                t,
            )
        result = fit_sparse_system(
            features,
            target_values,
            feature_names,
            target_name,
            filename,
            optimizer_config,
        )
        results.append(result)
        feature_names_by_target.append(feature_names)

    return {
        "dataset": filename.split(".")[0],
        "targets": [result["target"] for result in results],
        "coefficients": [result["coefficients"][0] for result in results],
        "features": feature_names_by_target,
        "protocol": protocol,
    }


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            result = run_sindy(data, x, y, z, t, dataset)
            all_results.append(result)
        except Exception as error:
            print(f"Error processing {dataset}: {error}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
