import json
from datetime import datetime
from pathlib import Path
import re
import sys

import numpy as np
import pysindy as ps

sys.path.append(str(Path().absolute()))

from data.config import NATIVE_PYSINDY_DEFAULTS, sindy_params
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
    if coefficient_tol > 0:
        coefficients[np.abs(coefficients) < coefficient_tol] = 0.0
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


def native_state(data_arrays, variable_names, axes, target, diff_config, include_time):
    """Build raw framework input, adding only metadata-required state variables."""

    time_axis = next(axis for name, axis, _ in axes if name == "t")
    channels = [np.asarray(values, dtype=float) for values in data_arrays]
    names = list(variable_names)

    target_order = int(target.get("order", 1))
    target_axis = target.get("axis", "t")
    if target_axis == "t" and target_order > 1:
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


def native_ode_library(config):
    """Construct a generic PySINDy ODE library, including Fourier interactions."""

    polynomial = ps.PolynomialLibrary(
        degree=int(config.get("polynomial_degree", 3)), include_bias=True
    )
    fourier = ps.FourierLibrary(
        n_frequencies=int(config.get("fourier_frequencies", 2))
    )
    return ps.GeneralizedLibrary(
        [polynomial, fourier],
        tensor_array=[[1, 1]],
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
    result = result.replace("sin(1 ", "sin(").replace("cos(1 ", "cos(")
    return result


def build_native_problem(
    data_arrays, variable_names, target, x, y, z, t, max_orders, config
):
    """Build one target problem entirely with PySINDy differentiation/libraries."""

    axes = get_data_axes(data_arrays[0], x, y, z, t)
    diff_config = config.get("differentiation", {})
    target_values = pysindy_derivative(
        data_arrays[variable_names.index(target["variable"])],
        axes,
        target.get("axis", "t"),
        target.get("order", 1),
        diff_config,
    )

    spatial_grid, spatial_axes = native_spatial_grid(axes)
    if spatial_grid is None:
        state, state_names = native_state(
            data_arrays, variable_names, axes, target, diff_config, include_time=True
        )
        library = native_ode_library(config)
    else:
        state, state_names = native_state(
            data_arrays, variable_names, axes, target, diff_config, include_time=False
        )
        spatial_orders = [
            order for (axis_name, _, _), order in zip(axes, max_orders) if axis_name != "t"
        ]
        library = native_pde_library(spatial_grid, max(spatial_orders), config)

    transformed = np.asarray(library.fit_transform(state), dtype=float)
    features = transformed.reshape(-1, transformed.shape[-1])
    target_vector = np.moveaxis(
        target_values, next(axis for name, axis, _ in axes if name == "t"), -1
    ).reshape(-1)
    feature_names = [
        canonical_native_feature_name(name, spatial_axes)
        for name in library.get_feature_names(state_names)
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
        native_config = {
            **NATIVE_PYSINDY_DEFAULTS,
            **(native_options or {}),
        }
        optimizer_config = {
            **NATIVE_PYSINDY_DEFAULTS["optimizer"],
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
