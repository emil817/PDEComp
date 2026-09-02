import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "epde" / "EPDE"))

import epde.globals as epde_globals
from epde.operators.common.sparsity import PhysicsInformedLasso
from epde.operators.common.stability import VaryingCoefSetup

from data.config import VWSR_DEFAULTS, sindy_params, vwsr_params
from utils.dataloader import load_data
from utils.derivatives import compute_derivative_bundle
from utils.sindy_library import (
    build_crop_slices,
    build_target_problem,
    configured_max_deriv_order,
    default_targets,
    default_variable_names,
    normalize_data_arrays,
)


RESULTS_DIR = PROJECT_ROOT / "results" / "vwsr"
DATASETS = list(vwsr_params)
ZERO_TOLERANCE = 1e-12


def save_combined_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def build_run_params(filename):
    if filename not in vwsr_params:
        raise KeyError(f"No VWSR params configured for {filename!r}")
    if filename not in sindy_params:
        raise KeyError(f"No shared library params configured for {filename!r}")

    params = dict(VWSR_DEFAULTS)
    params.update(vwsr_params[filename])
    params["optimizer"] = {
        **VWSR_DEFAULTS.get("optimizer", {}),
        **vwsr_params[filename].get("optimizer", {}),
    }
    params["sindy_config"] = sindy_params[filename]
    params["crop"] = params.get("crop", sindy_params[filename].get("crop", 0))
    return params


def cropped_shape(shape, crop_slices):
    return tuple(
        len(range(*crop_slice.indices(axis_size)))
        for axis_size, crop_slice in zip(shape, crop_slices)
    )


@contextlib.contextmanager
def vwsr_global_options(config):
    previous = {
        "vc_mode_decouple": epde_globals.vc_mode_decouple,
        "anchor_on_residual": epde_globals.anchor_on_residual,
    }
    epde_globals.vc_mode_decouple = config.get("mode_decouple", True)
    epde_globals.anchor_on_residual = config.get("anchor_on_residual", False)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(epde_globals, name, value)


def configured_modes(config, grid_shape):
    modes = config.get("modes")
    if modes is None:
        modes = config.get("k_max", 3)
    if isinstance(modes, int):
        return (modes,) * len(grid_shape)
    if len(modes) != len(grid_shape):
        raise ValueError(
            f"VWSR modes {modes!r} do not match cropped grid shape {grid_shape!r}"
        )
    return tuple(int(value) for value in modes)


def fit_vwsr_sparse_system(
    features,
    target_values,
    feature_names,
    target_name,
    target_variable,
    grid_shape,
    optimizer_config,
):
    features = np.asarray(features, dtype=float)
    target_values = np.asarray(target_values, dtype=float).reshape(-1)

    constant_indexes = [index for index, name in enumerate(feature_names) if name == "1"]
    if len(constant_indexes) > 1:
        raise ValueError("The shared library contains more than one constant term")

    constant_index = constant_indexes[0] if constant_indexes else None
    regression_indexes = [
        index for index in range(len(feature_names)) if index != constant_index
    ]
    regression_features = features[:, regression_indexes]

    gram_setup = VaryingCoefSetup(
        regression_features,
        target_values,
        np.ones(target_values.size, dtype=float),
        grid_shape,
        main_var=target_variable,
        modes=configured_modes(optimizer_config, grid_shape),
        k_max=optimizer_config.get("k_max", 3),
        freq_coef=optimizer_config.get("freq_coef", 0.0),
        fit_intercept=True,
    )
    estimator = PhysicsInformedLasso(
        max_iter=optimizer_config.get("max_iter", 1000),
        tol=optimizer_config.get("tol", 1e-4),
        grid_shape=grid_shape,
        main_var=target_variable,
    )
    with vwsr_global_options(optimizer_config):
        estimator.fit(
            regression_features,
            target_values,
            sample_weights=np.ones(target_values.size, dtype=float),
            gram_setup=gram_setup,
        )

    coefficients = np.zeros(len(feature_names), dtype=float)
    coefficients[regression_indexes] = estimator.coef_
    if constant_index is not None:
        coefficients[constant_index] = estimator.intercept_

    coefficient_tol = optimizer_config.get("coefficient_tol", 0.0)
    coefficients[np.abs(coefficients) < max(coefficient_tol, ZERO_TOLERANCE)] = 0.0

    active_terms = [
        f"{coefficient:.4f} {feature}"
        for coefficient, feature in zip(coefficients, feature_names)
        if coefficient != 0.0
    ]
    rhs = " + ".join(active_terms).replace("+ -", "- ") or "0"
    print(f"{target_name} = {rhs}")
    print()

    return {
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }


def run_vwsr(data, x, y, z, t, filename):
    """Run EPDE's VWSR optimizer on the shared fixed candidate library."""

    params = build_run_params(filename)
    sindy_config = params["sindy_config"]
    data_arrays = normalize_data_arrays(data)
    library_config = sindy_config.get("library", {})
    variable_names = library_config.get(
        "variable_names", default_variable_names(data_arrays)
    )
    targets = sindy_config.get("targets", default_targets(variable_names))
    derivatives = compute_derivative_bundle(
        data_arrays if len(data_arrays) > 1 else data_arrays[0],
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=variable_names,
        max_orders=configured_max_deriv_order(data_arrays[0].shape, sindy_config),
    )
    crop_slices = build_crop_slices(data_arrays[0].shape, params.get("crop", 0))
    grid_shape = cropped_shape(data_arrays[0].shape, crop_slices)

    results = []
    feature_names_by_target = []
    library_sizes = {}
    for target in targets:
        target_name, features, feature_names, target_values = build_target_problem(
            target,
            sindy_config,
            derivatives,
            crop_slices,
            data_arrays[0].shape,
            x,
            t,
        )
        result = fit_vwsr_sparse_system(
            features,
            target_values,
            feature_names,
            target_name,
            target.get("variable"),
            grid_shape,
            params["optimizer"],
        )
        results.append(result)
        feature_names_by_target.append(feature_names)
        library_sizes[target_name] = len(feature_names)

    return {
        "dataset": filename.split(".")[0],
        "targets": [result["target"] for result in results],
        "coefficients": [result["coefficients"] for result in results],
        "features": feature_names_by_target,
        "library_sizes": library_sizes,
        "library_size": sum(library_sizes.values()),
    }


if __name__ == "__main__":
    selected_datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    all_results = []
    for dataset in selected_datasets:
        print(f"\n=== Processing {dataset} ===")
        start = time.perf_counter()
        try:
            data, x, y, z, t = load_data(dataset)
            all_results.append(run_vwsr(data, x, y, z, t, dataset))
        except Exception as error:
            print(f"Error processing {dataset}: {error}")
        finally:
            print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    save_combined_results(all_results)
