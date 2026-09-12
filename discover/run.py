import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISCOVER_ROOT = Path(__file__).resolve().parent
DISCOVER_FORK_ROOT = DISCOVER_ROOT / "discover"
DSO_ROOT = DISCOVER_FORK_ROOT / "dso"
RESULTS_DIR = Path("results/discover")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DSO_ROOT))

from data.config import (
    DISCOVER_DATASETS,
    DISCOVER_DEFAULTS,
    NATIVE_DISCOVER_DEFAULTS,
    discover_params,
    sindy_params,
)
from utils.derivatives import compute_derivative_bundle, get_derivative
from utils.dataloader import load_data
from utils.sindy_library import (
    build_crop_slices,
    build_target_problem,
    configured_max_deriv_order,
    default_targets,
)
from utils.protocols import FIXED_PROTOCOL, NATIVE_PROTOCOL, validate_protocol


DATASETS = DISCOVER_DATASETS


def save_combined_results(results):
    output_file = RESULTS_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump([results], handle, indent=2)


def build_run_params(filename, protocol=FIXED_PROTOCOL, native_options=None):
    if filename not in discover_params:
        raise KeyError(f"No DISCOVER params configured for {filename!r}")
    params = dict(DISCOVER_DEFAULTS)
    params.update(discover_params[filename])
    if protocol == NATIVE_PROTOCOL:
        params.update(NATIVE_DISCOVER_DEFAULTS)
        params.update(native_options or {})
        if native_options and native_options.get("max_iterations") is not None:
            params["n_samples"] = int(native_options["max_iterations"])
        if int(params["n_samples"]) < int(params["batch_size"]):
            raise ValueError(
                "DISCOVER native n_samples must be at least batch_size "
                f"({params['batch_size']})"
            )
    params["sindy_config"] = sindy_params[filename]
    return params


def discover_config_path(params):
    return DSO_ROOT / "dso" / "config" / "MODE1" / params["base_config"]


def scalar_values(data, filename):
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"DISCOVER wrapper supports scalar equations only, got a system in {filename}")
        data = data[0]
    return np.asarray(data, dtype=float)


def discover_time_derivative(values, t, order):
    """Compute a time derivative with DISCOVER's own finite differences."""

    from dso.task.pde.utils_v1 import FiniteDiff, FiniteDiff2

    step = float(np.asarray(t, dtype=float)[1] - np.asarray(t, dtype=float)[0])
    derivative_function = FiniteDiff if order == 1 else FiniteDiff2 if order == 2 else None
    if derivative_function is None:
        raise ValueError(f"DISCOVER native wrapper supports time derivatives up to order 2, got {order}")

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return derivative_function(values, step)
    return np.vstack([derivative_function(row, step) for row in values.T]).T


def native_ode_function_set(function_set, target_order):
    """Exclude the target derivative and higher derivatives from an ODE RHS."""

    derivative_tokens = {1: "diff", 2: "diff2", 3: "diff3", 4: "diff4"}
    return [
        token
        for token in function_set
        if token not in {
            name for order, name in derivative_tokens.items() if order >= target_order
        }
    ]


def build_external_problem(data, x, y, z, t, filename, params, protocol=FIXED_PROTOCOL):
    values = scalar_values(data, filename)
    if values.ndim not in (1, 2):
        raise ValueError(f"DISCOVER wrapper supports scalar ODE and 1D PDE data, got {filename}")
    if values.ndim == 2 and x is None:
        raise ValueError(f"DISCOVER 1D PDE data requires an x grid for {filename}")

    sindy_config = params["sindy_config"]
    if protocol == NATIVE_PROTOCOL:
        target = params["sindy_config"].get("targets", default_targets(["u"]))[0]
        if target.get("axis", "t") != "t":
            raise ValueError("DISCOVER native mode requires a time-derivative target")
        target_name = target.get("name", f"u_{'t' * target.get('order', 1)}")
        target_values = discover_time_derivative(values, t, target.get("order", 1))

        if values.ndim == 1:
            # DISCOVER's PDE grammar is reused for ODEs: its x1 token represents
            # time and diff(u1, x1) therefore represents a temporal derivative.
            u = values.reshape(-1, 1)
            discover_x = [np.asarray(t, dtype=float).reshape(-1, 1)]
            discover_t = np.zeros((1, 1), dtype=float)
        else:
            u = values.T
            discover_x = [np.asarray(x, dtype=float).reshape(-1, 1)]
            discover_t = np.asarray(t, dtype=float).reshape(-1, 1)

        return {
            "u": [u],
            "x": discover_x,
            "t": discover_t,
            "ut": target_values.T.reshape(-1, 1) if values.ndim == 2 else target_values.reshape(-1, 1),
            "features": None,
            "feature_names": None,
            "derivatives": {},
            "sym_true": params.get("sym_true"),
            "target_name": target_name,
            "n_input_var": 1,
            "n_state_var": 1,
        }

    bundle = compute_derivative_bundle(
        values,
        x=x,
        y=y,
        z=z,
        t=t,
        variable_names=["u"],
        max_orders=configured_max_deriv_order(values.shape, sindy_config),
    )
    target = sindy_config.get("targets", default_targets(["u"]))[0]
    crop_slices = build_crop_slices(values.shape, sindy_config.get("crop", 0))
    target_name, features, feature_names, target_values = build_target_problem(
        target,
        sindy_config,
        bundle,
        crop_slices,
        values.shape,
        x,
        t,
    )

    if values.ndim == 1:
        u = values.reshape(-1, 1)
        discover_x = [np.zeros((values.size, 1), dtype=float)]
    else:
        u = values.T
        discover_x = [np.asarray(x, dtype=float).reshape(-1, 1)]

    derivatives = {}
    if values.ndim == 2:
        for order in range(1, 5):
            try:
                derivatives[order] = get_derivative(bundle, "u", "x", order).T
            except KeyError:
                pass

    return {
        "u": [u],
        "x": discover_x,
        "t": np.asarray(t, dtype=float).reshape(-1, 1),
        "ut": target_values.reshape(-1, 1),
        "features": features,
        "feature_names": feature_names,
        "derivatives": derivatives,
        "sym_true": params.get("sym_true"),
        "target_name": target_name,
        "n_input_var": 1,
        "n_state_var": 1,
    }


def discover_overrides(filename, params):
    return {
        "experiment": {
            "logdir": str(RESULTS_DIR),
            "seed": params.get("seed", 0),
        },
        "task": {
            "task_type": "pde",
            "dataset": filename,
            "function_set": params["function_set"],
            "metric": "pde_reward",
            "metric_params": [params.get("metric_param", 0.01)],
            "threshold": params.get("threshold", 5e-4),
            "protected": params.get("protected", False),
            "decision_tree_threshold_set": [],
            "eq_num": 1,
            "spatial_error": params.get("spatial_error", False),
        },
        "training": {
            "n_samples": params["n_samples"],
            "batch_size": params["batch_size"],
            "epsilon": params["epsilon"],
            "n_cores_batch": params["n_cores_batch"],
            "early_stopping": params["early_stopping"],
            "verbose": params.get("verbose", True),
        },
        "controller": {
            "attention": False,
        },
        "prior": {
            "length": {"min_": 1, "max_": params.get("max_length", 15), "on": True},
            "repeat": {"tokens": "add", "min_": None, "max_": params.get("max_add_count", 8), "on": True},
            "inverse": {"on": False},
            "trig": {"on": False},
            "diff_left": {"on": False},
            "diff_right": {"on": False},
            "diff_descedent": {"on": False},
            "soft_length": {"loc": 5, "scale": 3, "on": True},
        },
        "gp_meld": {
            "run_gp_meld": False,
        },
        "gp_agg": {
            "run_gp_agg": False,
        },
        "pinn": {
            "use_pinn": False,
        },
        "parameterized": {
            "on": False,
        },
    }


@contextmanager
def native_library_mode(protocol):
    """Let DISCOVER build its standard grammar for raw external datasets."""

    if protocol != NATIVE_PROTOCOL:
        yield
        return

    from dso.task.pde import pde

    original_builder = pde._build_external_library

    def build_library(dataset):
        library_data = pde.get_external_library(dataset)
        if library_data and library_data.get("features") is None:
            return None
        return original_builder(dataset)

    pde._build_external_library = build_library
    try:
        yield
    finally:
        pde._build_external_library = original_builder


def term_to_feature(term, feature_names=None, ode=False):
    text = repr(term)
    if feature_names and text.startswith("theta_"):
        try:
            index = int(text.split("_", 1)[1])
            return feature_names[index]
        except (ValueError, IndexError):
            return text

    replacements = {
        "u1": "u",
        "diff(u1,x1)": "u_x",
        "diff2(u1,x1)": "u_xx",
        "diff3(u1,x1)": "u_xxx",
        "diff4(u1,x1)": "u_xxxx",
        "n2(u1)": "u^2",
        "n3(u1)": "u^3",
        "mul(u1,diff(u1,x1))": "u u_x",
        "mul(diff(u1,x1),u1)": "u u_x",
        "mul(n2(u1),diff(u1,x1))": "u^2 u_x",
        "mul(diff(u1,x1),n2(u1))": "u^2 u_x",
        "mul(u,diff(u,x1))": "u u_x",
        "mul(u1,diff2(u1,x1))": "u u_xx",
        "mul(diff2(u1,x1),u1)": "u u_xx",
        "mul(diff(u1,x1),diff(u1,x1))": "u_x^2",
        "div(diff(u1,x1),x1)": "(1/x) u_x",
        "sin(x1)": "sin(x)",
        "cos(x1)": "cos(x)",
        "mul(sin(x1),cos(x2))": "sin(x) cos(t)",
    }
    feature = replacements.get(text, text)
    if ode:
        ode_replacements = {
            "x1": "t",
            "u_x": "u_t",
            "u_xx": "u_tt",
            "u_xxx": "u_ttt",
            "u_xxxx": "u_tttt",
            "u^2 u_x": "u^2 u_t",
            "sin(x)": "sin(t)",
            "cos(x)": "cos(t)",
            "sin(x1)": "sin(t)",
            "cos(x1)": "cos(t)",
        }
        feature = ode_replacements.get(feature, feature)
        feature = feature.replace("diff(u1,x1)", "u_t")
    return feature


def extract_result(train_result, filename, target_name, elapsed_time, params, protocol=FIXED_PROTOCOL):
    from dso.task.pde import data_load

    library_data = data_load.get_external_library(filename) or {}
    feature_names = library_data.get("feature_names")

    program = train_result["program"]
    terms = getattr(program.STRidge, "terms", [])
    coefficients = [float(value) for value in np.asarray(program.w, dtype=float).reshape(-1)]
    is_ode = np.asarray(data_load.EXTERNAL_PDE_PROBLEMS[filename]["u"][0]).shape[1] == 1
    selected_features = [term_to_feature(term, feature_names, ode=is_ode) for term in terms]

    if feature_names:
        features = list(feature_names)
        coefficient_by_feature = dict(zip(selected_features, coefficients))
        coefficients = [coefficient_by_feature.get(feature_name, 0.0) for feature_name in features]
    else:
        features = selected_features
        if len(coefficients) > len(features):
            features.append("__constant__")

    coefficient_tol = (
        params
        .get("sindy_config", {})
        .get("optimizer", {})
        .get("coefficient_tol", 1e-12)
    )
    coefficients = [
        0.0 if abs(coefficient) < coefficient_tol else coefficient
        for coefficient in coefficients
    ]
    library_size = len(features) if protocol == FIXED_PROTOCOL else ""

    return {
        "dataset": filename.split(".")[0],
        "targets": [target_name],
        "features": [features],
        "coefficients": [coefficients[: len(features)]],
        "active_terms": [[feature for feature, coefficient in zip(features, coefficients) if abs(coefficient) > 1e-12]],
        "library_sizes": {target_name: library_size},
        "library_size": library_size,
        "model": train_result.get("expression", ""),
        "equation_texts": [train_result.get("expression", "")],
        "time": elapsed_time,
        "discover_reward": train_result.get("r", ""),
        "discover_nmse": train_result.get("nmse_test", ""),
        "protocol": protocol,
    }


def run_discover(
    data,
    x,
    y,
    z,
    t,
    filename,
    only_print=True,
    protocol=FIXED_PROTOCOL,
    native_options=None,
):
    validate_protocol(protocol)
    params = build_run_params(filename, protocol=protocol, native_options=native_options)

    try:
        from dso import DeepSymbolicOptimizer_PDE
        from dso.task.pde import data_load
    except Exception as error:
        raise RuntimeError(
            "DISCOVER is expected in discover/discover/dso, but its TensorFlow 1.x "
            "stack is not importable in the current environment."
        ) from error

    problem = build_external_problem(data, x, y, z, t, filename, params, protocol=protocol)
    if protocol == NATIVE_PROTOCOL and np.asarray(problem["u"][0]).shape[1] == 1:
        target_order = int(params["sindy_config"].get("targets", [{}])[0].get("order", 1))
        params["function_set"] = native_ode_function_set(
            params["function_set"], target_order
        )
    data_load.set_external_pde_problem(filename, **problem)
    start = time.perf_counter()
    try:
        with native_library_mode(protocol):
            model = DeepSymbolicOptimizer_PDE(
                str(discover_config_path(params)),
                pde_config=discover_overrides(filename, params),
            )
            raw_result = model.train()
        train_result = raw_result[0] if isinstance(raw_result, list) else raw_result
        elapsed_time = time.perf_counter() - start
        result = extract_result(
            train_result,
            filename,
            params["target"],
            elapsed_time,
            params,
            protocol=protocol,
        )
    finally:
        data_load.clear_external_pde_problem(filename)

    if only_print:
        print(result["model"])
    return result


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            all_results.append(run_discover(data, x, y, z, t, dataset))
        except Exception as error:
            print(f"Error processing {dataset}: {error}")

    if all_results:
        save_combined_results(all_results)
    print("\nAll experiments completed!")
