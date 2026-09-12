import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent / "EPDE"))

from epde.interface.interface import EpdeSearch
from epde.interface.prepared_tokens import CustomTokens, CustomEvaluator
from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
from epde.operators.common.sparsity import VWSRSparsity

from utils.dataloader import load_data
from data.config import COMMON_PARAMS, NATIVE_EPDE_DEFAULTS, TRUE_COEFFICIENTS, epde_params
from utils.derivatives import build_epde_derivatives, numpy_gradient_derivative
from utils.protocols import FIXED_PROTOCOL, NATIVE_PROTOCOL, validate_protocol

RESULTS_DIR = Path("results/epde")
COEFFICIENT_TOLERANCE = 1e-12
DATASETS = list(epde_params)


def save_combined_results(results):
    """Save all dataset results into one JSON file."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump([results], f, indent=2)


def get_coordinate_tensors(coordinate_tensors, t, x, y, z):
    """Build coordinate tensors in the format expected by EPDE."""

    if coordinate_tensors is None:
        # 2d
        return np.meshgrid(t, x, indexing = 'ij')

    elif coordinate_tensors == '1d':
        return [t, ]

    elif coordinate_tensors == '3d':
        return np.meshgrid(t, y, x, indexing = 'ij')


def get_additional_tokens(additional_tokens, grid, data, trig_tokens_freq):
    """Create optional EPDE token families configured for a dataset."""

    if isinstance(additional_tokens, (list, tuple)):
        tokens = []
        for token_name in additional_tokens:
            tokens.extend(get_additional_tokens(token_name, grid, data, trig_tokens_freq))
        return tokens

    if additional_tokens is None:
        return []

    dimensionality = data[0].ndim - 1

    if additional_tokens == 'GridTokens':
        return [
            GridTokens(
                ["x"],
                dimensionality=dimensionality,
                max_power=2,
            )
        ]

    elif additional_tokens == 'TrigonometricTokens':
        trig_tokens = TrigonometricTokens(freq=trig_tokens_freq,
                                          dimensionality=dimensionality)
        return [trig_tokens]

    elif additional_tokens in {'TrigonometricTokens, GridTokens', 'ODE_simple_discovery'}:
        dimensionality = 0
        trig_tokens = TrigonometricTokens(
            freq=trig_tokens_freq, dimensionality=dimensionality
        )
        grid_tokens = GridTokens(
            [
                "x_0",
            ],
            dimensionality=dimensionality,
            max_power=2,
        )
        return [trig_tokens, grid_tokens]

    elif additional_tokens == 'custom_trig_tokens':
        custom_trigonometric_eval_fun = {
            "cos(t)sin(x)": lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1]))
            ** kwargs["power"]
        }
        custom_trig_evaluator = CustomEvaluator(
            custom_trigonometric_eval_fun, eval_fun_params_labels=["power"]
        )
        trig_params_ranges = {"power": (1, 1)}
        trig_params_equal_ranges = {}
        custom_trig_tokens = CustomTokens(
            token_type="trigonometric",
            token_labels=["cos(t)sin(x)"],
            evaluator=custom_trig_evaluator,
            params_ranges=trig_params_ranges,
            params_equality_ranges=trig_params_equal_ranges,
            meaningful=True,
            unique_token_type=False,
        )
        return [custom_trig_tokens]

    elif additional_tokens == 'sindy_pde_custom_tokens':
        values = np.asarray(data[0], dtype=float)
        t_grid, x_grid = grid[0], grid[1]
        x_step = float(x_grid[0, 1] - x_grid[0, 0]) if x_grid.shape[1] > 1 else 1.0
        u_x = numpy_gradient_derivative(values, x_step, axis=1, order=1)

        token_tensors = {
            "t": t_grid,
            "sin(x)": np.sin(x_grid),
            "cos(t)": np.cos(t_grid),
            "sin(x)cos(t)": np.sin(x_grid) * np.cos(t_grid),
            "cos(t)sin(x)": np.cos(t_grid) * np.sin(x_grid),
            "cos(x)sin(t)": np.cos(x_grid) * np.sin(t_grid),
            "d_x_u_u_x": numpy_gradient_derivative(values * u_x, x_step, axis=1, order=1),
        }
        return [
            CacheStoredTokens(
                token_type="sindy_pde_custom",
                token_labels=list(token_tensors),
                token_tensors=token_tensors,
                params_ranges={"power": (1, 1)},
                params_equality_ranges=None,
                unique_token_type=False,
                meaningful=True,
            )
        ]

    elif additional_tokens == 'sindy_time_token':
        t_grid = np.asarray(grid[0], dtype=float)
        return [
            CacheStoredTokens(
                token_type="sindy_time",
                token_labels=["t"],
                token_tensors={"t": t_grid},
                params_ranges={"power": (1, 1)},
                params_equality_ranges=None,
                unique_token_type=False,
                meaningful=True,
            )
        ]

    elif additional_tokens == 'sindy_ode_custom_tokens':
        values = np.asarray(data[0], dtype=float)
        t_grid = np.asarray(grid[0], dtype=float)
        t_step = float(t_grid[1] - t_grid[0]) if t_grid.shape[0] > 1 else 1.0
        u_t = numpy_gradient_derivative(values, t_step, axis=0, order=1)
        token_tensors = {
            "t": t_grid,
            "sin(t)": np.sin(t_grid),
            "cos(t)": np.cos(t_grid),
            "u_t sin(2 t)": u_t * np.sin(2 * t_grid),
        }
        return [
            CacheStoredTokens(
                token_type="sindy_ode_custom",
                token_labels=list(token_tensors),
                token_tensors=token_tensors,
                params_ranges={"power": (1, 1)},
                params_equality_ranges=None,
                unique_token_type=False,
                meaningful=True,
            )
        ]


def resolve_boundary(bounds, grid):
    """Resolve automatic 10% boundary from coordinate tensors."""

    if bounds != "auto_10pct":
        return bounds

    sample = np.asarray(grid[0])
    if sample.ndim <= 1:
        return max(1, len(sample) // 10)
    return tuple(max(1, axis_size // 10) for axis_size in sample.shape)


def normalize_epde_max_deriv_order(max_deriv_order, data):
    """Interpret (time, spatial) derivative limits for any dataset dimensionality."""

    ndim = np.asarray(data).ndim
    if isinstance(max_deriv_order, int):
        return tuple([max_deriv_order] * ndim)

    orders = tuple(max_deriv_order)
    if len(orders) == ndim:
        return orders
    if len(orders) > ndim:
        return orders[:ndim]
    if len(orders) == 2 and ndim > 2:
        return (orders[0],) + tuple([orders[1]] * (ndim - 1))

    raise ValueError(f"Expected derivative order for {ndim} axes, got {max_deriv_order}")


def term_signature(term):
    """Return a stable comparable representation of an EPDE term."""

    if hasattr(term, "term_label"):
        return repr(term.term_label)
    if hasattr(term, "name"):
        return str(term.name)
    return str(term)


def active_terms(equation, tolerance=COEFFICIENT_TOLERANCE):
    """Return active non-target terms for one EPDE equation."""

    coefficients = np.asarray(equation.weights_final, dtype=float)
    terms = []
    for term_index, term in enumerate(equation.structure):
        if term_index == equation.target_idx:
            continue
        coefficient_index = (
            term_index
            if term_index < equation.target_idx
            else term_index - 1
        )
        if abs(coefficients[coefficient_index]) > tolerance:
            terms.append(term_signature(term))

    if abs(coefficients[-1]) > tolerance:
        terms.append("__constant__")

    return sorted(terms)


def target_name(equation):
    """Return the selected right-hand-side term name for one EPDE equation."""

    return equation.structure[equation.target_idx].name


def benchmark_target_names(filename, equation_count):
    """Return benchmark target names for the selected EPDE solution."""

    configured_targets = list(TRUE_COEFFICIENTS.get(filename, {}))
    if len(configured_targets) >= equation_count:
        return configured_targets[:equation_count]
    return []


def pareto_solutions(search_obj):
    """Return discovered EPDE solutions from the first reported Pareto level."""

    solutions = search_obj.equations(only_print=False, num=1)
    if search_obj.multiobjective_mode:
        if not solutions or not solutions[0]:
            raise ValueError("EPDE returned an empty first non-dominated level")
        return solutions[0]

    if not solutions:
        raise ValueError("EPDE returned an empty population")
    return solutions


def select_solution(search_obj, solution_index=0):
    """Select one discovered EPDE solution from the first reported level."""

    return pareto_solutions(search_obj)[solution_index]


def extract_result(solution, filename, elapsed_time):
    """Convert an EPDE solution into a PySINDy-like result dictionary."""

    equations = list(solution)
    targets = []
    features = []
    coefficients = []
    active = []
    equation_texts = []

    benchmark_targets = benchmark_target_names(filename, len(equations))

    for equation_index, equation in enumerate(equations):
        equation_coefficients = np.asarray(equation.weights_final, dtype=float)
        equation_features = []
        full_coefficients = []
        for term_index, term in enumerate(equation.structure):
            if term_index == equation.target_idx:
                continue
            coefficient_index = (
                term_index
                if term_index < equation.target_idx
                else term_index - 1
            )
            equation_features.append(term_signature(term))
            full_coefficients.append(float(equation_coefficients[coefficient_index]))

        equation_features.append("__constant__")
        full_coefficients.append(float(equation_coefficients[-1]))

        targets.append(
            benchmark_targets[equation_index]
            if equation_index < len(benchmark_targets)
            else target_name(equation)
        )
        features.append(equation_features)
        coefficients.append(full_coefficients)
        active.append(active_terms(equation))
        equation_texts.append(equation.text_form)

    return {
        "dataset": filename.split(".")[0],
        "targets": targets,
        "features": features,
        "coefficients": coefficients,
        "active_terms": active,
        "equation_texts": equation_texts,
        "model": solution.text_form,
        "time": elapsed_time,
    }


def run_epde(
    data,
    x,
    y,
    z,
    t,
    filename,
    device="cuda",
    solution_index=0,
    only_print=True,
    visualize=False,
    return_all=False,
    protocol=FIXED_PROTOCOL,
    native_options=None,
):
    """Run EPDE with fixed derivatives or its native preprocessing pipeline."""

    start = time.perf_counter()
    validate_protocol(protocol)

    configured_params = epde_params[filename]
    if protocol == NATIVE_PROTOCOL:
        params = {
            **NATIVE_EPDE_DEFAULTS,
            **(native_options or {}),
            "variable_names": configured_params["variable_names"],
            "max_deriv_order": configured_params.get(
                "max_deriv_order", COMMON_PARAMS["max_deriv_order"]
            ),
            "coordinate_tensors": configured_params.get("coordinate_tensors"),
            "additional_tokens": (
                None
                if len(configured_params["variable_names"]) > 1
                else ["GridTokens", "TrigonometricTokens"]
            ),
            "trig_tokens_freq": (0.999, 1.001),
            "use_solver": False,
            "fourier_layers": False,
        }
        if native_options and native_options.get("max_iterations") is not None:
            params["training_epochs"] = int(native_options["max_iterations"])
    else:
        params = configured_params
    epde_data = data if isinstance(data, list) else [data]
    
    use_solver = params["use_solver"]
    bounds = params["boundary"]
    multiobjective_mode = params.get("multiobjective_mode", True)
    use_pic = params.get("use_pic", True)

    population_size = params["population_size"]
    training_epochs = params["training_epochs"]

    max_deriv_order = normalize_epde_max_deriv_order(params.get("max_deriv_order", COMMON_PARAMS["max_deriv_order"]), epde_data[0])
    derivs = params.get("derivs", None)
    equation_terms_max_number = params["equation_terms_max_number"]
    data_fun_pow = params.get("data_fun_pow", COMMON_PARAMS["data_fun_pow"])
    deriv_fun_pow = params.get("deriv_fun_pow", 1)
    additional_tokens = params.get("additional_tokens", None)
    equation_factors_max_number = params["equation_factors_max_number"]
    eq_sparsity_interval = params["eq_sparsity_interval"]
    default_preprocessor_type = params["default_preprocessor_type"]
    variable_names = params["variable_names"]
    fourier_layers = params.get("fourier_layers", False)
    coordinate_tensors = params.get("coordinate_tensors", None)
    trig_tokens_freq = params.get("trig_tokens_freq", None)


    grid = get_coordinate_tensors(coordinate_tensors, t, x, y, z)
    bounds = resolve_boundary(bounds, grid)
    if protocol == FIXED_PROTOCOL and derivs is None:
        derivs = build_epde_derivatives(
            data=epde_data,
            x=x,
            y=y,
            z=z,
            t=t,
            variable_names=variable_names,
            max_deriv_order=max_deriv_order,
        )

    epde_search_obj = EpdeSearch(
        use_solver=use_solver,
        multiobjective_mode=multiobjective_mode,
        use_pic=use_pic,
        boundary=bounds,
        coordinate_tensors=grid,
        device=device,
        discrepancy_metric=params.get("discrepancy_metric", "wape"),
        sparsity_cls=VWSRSparsity if protocol == FIXED_PROTOCOL else None,
    )

    epde_search_obj.set_preprocessor(
        default_preprocessor_type=default_preprocessor_type, preprocessor_kwargs={}
    )

    epde_search_obj.set_moeadd_params(population_size=population_size,
                                        training_epochs=training_epochs)

    epde_search_obj.fit(
        data=epde_data,
        variable_names=variable_names,
        max_deriv_order=max_deriv_order,
        derivs=derivs,
        equation_terms_max_number=equation_terms_max_number,
        data_fun_pow=data_fun_pow,
        deriv_fun_pow=deriv_fun_pow,
        additional_tokens=get_additional_tokens(additional_tokens, grid, epde_data, trig_tokens_freq),
        equation_factors_max_number=equation_factors_max_number,
        eq_sparsity_interval=eq_sparsity_interval,
        fourier_layers=fourier_layers
    )


    if only_print:
        epde_search_obj.equations(only_print=True, num=1)
    finish = time.perf_counter()
    elapsed_time = finish - start
    if visualize:
        epde_search_obj.visualize_solutions()

    if return_all:
        candidates = [
            extract_result(solution, filename, elapsed_time)
            for solution in pareto_solutions(epde_search_obj)
        ]
        for candidate in candidates:
            candidate["protocol"] = protocol
            if protocol == NATIVE_PROTOCOL:
                candidate["library_sizes"] = {
                    target: "" for target in candidate["targets"]
                }
                candidate["library_size"] = ""
        return {
            "dataset": filename.split(".")[0],
            "candidates": candidates,
            "time": elapsed_time,
            "model": "\n\n".join(candidate.get("model", "") for candidate in candidates),
            "protocol": protocol,
        }

    solution = select_solution(epde_search_obj, solution_index)
    result = extract_result(solution, filename, elapsed_time)
    result["protocol"] = protocol
    if protocol == NATIVE_PROTOCOL:
        result["library_sizes"] = {target: "" for target in result["targets"]}
        result["library_size"] = ""
    return result


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected_datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    print("CUDA available: ", torch.cuda.is_available())
    print("EPDE device: ", device)
    all_results = []
    for dataset in selected_datasets:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)
            result = run_epde(data, x, y, z, t, dataset, device=device)
            all_results.append(result)
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    if all_results:
        save_combined_results(all_results)
    print("\nAll experiments completed!")
