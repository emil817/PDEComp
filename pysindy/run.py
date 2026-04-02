import numpy as np
import pysindy as ps
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path().absolute()))

from data.dataloader import load_data
from config import sindy_params

DATA_DIR = Path("data")
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
    "ODE_simple_discovery"
]


def save_combined_results(results):
    """Save results to a common JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"results_{timestamp}.json"
    output_file.parent.mkdir(exist_ok=True)

    result = [results]

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def build_pde_library(lib_config, x):
    return ps.PDELibrary(derivative_order=lib_config.get("derivative_order", 3),
                         spatial_grid=x,
                         include_bias=lib_config.get("pde_include_bias", True),
                         function_library=ps.PolynomialLibrary(
                            degree=lib_config.get("poly_degree", 2),
                            include_bias=lib_config.get("poly_include_bias", False),
                         ),
                         differentiation_method=ps.FiniteDifference,
    )


def build_library(lib_config, x):
    """Создание объектов Library на основе конфига"""

    lib_type = lib_config.get('type', 'polynomial')

    if lib_type == 'polynomial':
        return ps.PolynomialLibrary(
            degree=lib_config.get('degree', 2),
            include_bias=lib_config.get('include_bias', True)
        )

    elif lib_type == 'pde':
        return build_pde_library(lib_config, x)

    elif lib_type == 'pde_custom_concat':
        base_lib = build_pde_library(lib_config, x)
        functions = [lambda val : 1 / val]
        functions_names = [lambda val : "1/" + val]
        custom_lib = ps.CustomLibrary(library_functions=functions, function_names=functions_names) * base_lib
        return ps.ConcatLibrary([custom_lib, base_lib])

    elif lib_type == 'poly_and_fourier':
        poly_lib = ps.PolynomialLibrary(
            degree=lib_config.get('poly_degree', 2),
            include_bias=lib_config.get('poly_include_bias', True)
        )
        trig_lib = ps.FourierLibrary(n_frequencies=lib_config.get('n_frequencies', 1))
        return poly_lib + trig_lib

    raise ValueError(f"Unknown library type: {lib_type}")


def build_optimizer(opt_config):
    """Создание объектов Optimizer на основе конфига"""

    opt_type = opt_config.get('type', 'STLSQ')

    if opt_type == 'STLSQ':
        return ps.STLSQ(
            threshold=opt_config.get('threshold', 0.1),
            alpha=opt_config.get('alpha', 0.05),
            normalize_columns=opt_config.get('normalize_columns', False)
        )

    elif opt_type == 'SR3':
        return ps.SR3(max_iter=opt_config.get('max_iter', 30),
                      tol=opt_config.get('tol', 1e-5),
                      normalize_columns=opt_config.get('normalize_columns', False),
                      reg_weight_lam = opt_config.get('reg_weight_lam', opt_config.get('threshold', 0.1)),
                      regularizer=opt_config.get('regularizer', opt_config.get('thresholder', 'L0')))

    raise ValueError(f"Unknown optimizer type: {opt_type}")


def prepare_standard_data(data, x, t, filename):
    if isinstance(data, list):
        data = np.array(data)
    if len(data.shape) == 1:
        data = data.T.reshape(len(t), 1)
    elif filename in ["lotka_data.npy", "lorenz_data.npy", "ODE_simple_discovery"]:
        data = data.T
    elif len(data.shape) == 2:
        data = data.T.reshape(len(x), len(t), 1)
    return data


def finite_difference(data, step, axis, order=1):
    derivative = np.asarray(data, dtype=float)
    for _ in range(order):
        derivative = np.gradient(derivative, step, axis=axis)
    return derivative


def build_crop_slices(shape, crop):
    if crop <= 0:
        return tuple(slice(None) for _ in shape)
    return tuple(slice(crop, dim - crop) for dim in shape)


def print_sparse_equation(target_name, feature_names, coefficients, precision=4):
    active_terms = []
    for coef, feature in zip(np.ravel(coefficients), feature_names):
        if abs(coef) > 1e-12:
            active_terms.append(f"{coef:.{precision}f} {feature}")

    rhs = " + ".join(active_terms).replace("+ -", "- ")
    if not rhs:
        rhs = "0"
    print(f"{target_name} = {rhs}")


def fit_manual_system(feature_matrix, target_vector, feature_names, target_name, filename, opt_config):
    optimizer = build_optimizer(opt_config)
    optimizer.fit(feature_matrix, target_vector)

    coefficients = np.asarray(optimizer.coef_)
    if coefficients.ndim == 1:
        coefficients = coefficients[np.newaxis, :]

    print_sparse_equation(target_name, feature_names, coefficients[0])

    result = {
        "dataset": filename.split(".")[0],
        "target": target_name,
        "coefficients": coefficients.tolist(),
        "features": feature_names,
    }

    return result


def run_manual_dataset(data, x, y, z, t, filename, params):
    crop = params.get("crop", 0)

    if isinstance(data, list):
        data = np.array(data)

    if filename == "ode_data.npy":
        u = np.asarray(data, dtype=float).reshape(-1)
        dt = t[1] - t[0]
        u_t = finite_difference(u, dt, axis=0, order=1)
        u_tt = finite_difference(u, dt, axis=0, order=2)
        s = build_crop_slices(u.shape, crop)[0]

        features = np.column_stack([u[s], t[s], (u_t * np.sin(2 * t))[s]])
        feature_names = ["u", "t", "u_t sin(2 t)"]

        return fit_manual_system(features, u_tt[s], feature_names, "u_tt", filename, params["optimizer"])

    if filename == "vdp_data.npy":
        u = np.asarray(data, dtype=float).reshape(-1)
        dt = t[1] - t[0]
        u_t = finite_difference(u, dt, axis=0, order=1)
        u_tt = finite_difference(u, dt, axis=0, order=2)
        s = build_crop_slices(u.shape, crop)[0]
    
        features = np.column_stack([u[s], u_t[s], ((u ** 2) * u_t)[s]])
        feature_names = ["u", "u_t", "u^2 u_t"]

        return fit_manual_system(features, u_tt[s], feature_names, "u_tt", filename, params["optimizer"])

    if filename == "ODE_simple_discovery":
        u = np.asarray(data[0], dtype=float).reshape(-1)
        dt = t[1] - t[0]
        u_t = finite_difference(u, dt, axis=0, order=1)
        features = np.column_stack([np.sin(t), np.cos(t)])
        feature_names = ["sin(t)", "cos(t)"]

        return fit_manual_system(features, u_t, feature_names, "u_t", filename, params["optimizer"])

    if filename == "ns_data.mat":
        u = np.asarray(data[0], dtype=float)
        v = np.asarray(data[1], dtype=float)
        p = np.asarray(data[2], dtype=float)
        dt = t[1] - t[0]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        crop_slices = build_crop_slices(u.shape, crop)

        u_t = finite_difference(u, dt, axis=0, order=1)
        v_t = finite_difference(v, dt, axis=0, order=1)
        u_x = finite_difference(u, dx, axis=2, order=1)
        u_y = finite_difference(u, dy, axis=1, order=1)
        u_xx = finite_difference(u, dx, axis=2, order=2)
        u_yy = finite_difference(u, dy, axis=1, order=2)
        v_x = finite_difference(v, dx, axis=2, order=1)
        v_y = finite_difference(v, dy, axis=1, order=1)
        v_xx = finite_difference(v, dx, axis=2, order=2)
        v_yy = finite_difference(v, dy, axis=1, order=2)
        p_x = finite_difference(p, dx, axis=2, order=1)
        p_y = finite_difference(p, dy, axis=1, order=1)

        optimizer = build_optimizer(params["optimizer"])

        u_features = np.column_stack([
            (u[crop_slices] * u_x[crop_slices]).ravel(),
            (v[crop_slices] * u_y[crop_slices]).ravel(),
            p_x[crop_slices].ravel(),
            (u_xx[crop_slices] + u_yy[crop_slices]).ravel(),
            u[crop_slices].ravel(),
            np.ones(u[crop_slices].size),
        ])
        u_feature_names = ["u u_x", "v u_y", "p_x", "(u_xx + u_yy)", "u", "1"]
        optimizer.fit(u_features, u_t[crop_slices].ravel())
        u_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("u_t", u_feature_names, u_coefficients)

        optimizer = build_optimizer(params["optimizer"])
        v_features = np.column_stack([
            (u[crop_slices] * v_x[crop_slices]).ravel(),
            (v[crop_slices] * v_y[crop_slices]).ravel(),
            p_y[crop_slices].ravel(),
            (v_xx[crop_slices] + v_yy[crop_slices]).ravel(),
            v[crop_slices].ravel(),
            np.ones(v[crop_slices].size),
        ])
        v_feature_names = ["u v_x", "v v_y", "p_y", "(v_xx + v_yy)", "v", "1"]
        optimizer.fit(v_features, v_t[crop_slices].ravel())
        v_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("v_t", v_feature_names, v_coefficients)

        optimizer = build_optimizer(params["optimizer"])
        continuity_features = np.column_stack([
            v_y[crop_slices].ravel(),
            np.ones(u[crop_slices].size),
        ])
        continuity_feature_names = ["v_y", "1"]
        optimizer.fit(continuity_features, u_x[crop_slices].ravel())
        continuity_coefficients = np.asarray(optimizer.coef_).reshape(-1)
        print_sparse_equation("u_x", continuity_feature_names, continuity_coefficients)

        return {
            "dataset": filename.split(".")[0],
            "targets": ["u_t", "v_t", "u_x"],
            "coefficients": [
                u_coefficients.tolist(),
                v_coefficients.tolist(),
                continuity_coefficients.tolist(),
            ],
            "features": [
                u_feature_names,
                v_feature_names,
                continuity_feature_names,
            ],
        }

    u = np.asarray(data, dtype=float)
    dt = t[1] - t[0]
    dx = x[1] - x[0]
    crop_slices = build_crop_slices(u.shape, crop)

    if filename == "wave_data.csv":
        u_tt = finite_difference(u, dt, axis=0, order=2)
        u_xx = finite_difference(u, dx, axis=1, order=2)
        features = np.column_stack([
            u_xx[crop_slices].ravel(),
            u[crop_slices].ravel(),
            np.ones(u[crop_slices].size),
        ])
        feature_names = ["u_xx", "u", "1"]
        target = u_tt[crop_slices].ravel()

        return fit_manual_system(features, target, feature_names, "u_tt", filename, params["optimizer"])

    if filename == "pde_divide_data.npy":
        u_t = finite_difference(u, dt, axis=0, order=1)
        u_x = finite_difference(u, dx, axis=1, order=1)
        u_xx = finite_difference(u, dx, axis=1, order=2)
        x_grid = np.broadcast_to(x, u.shape)
        features = np.column_stack([
            u_xx[crop_slices].ravel(),
            (u_x[crop_slices] / x_grid[crop_slices]).ravel(),
            u_x[crop_slices].ravel(),
            u[crop_slices].ravel(),
            np.ones(u[crop_slices].size),
        ])
        feature_names = ["u_xx", "(1/x) u_x", "u_x", "u", "1"]
        target = u_t[crop_slices].ravel()

        return fit_manual_system(features, target, feature_names, "u_t", filename, params["optimizer"])

    if filename == "pde_compound_data.npy":
        u_t = finite_difference(u, dt, axis=0, order=1)
        u_x = finite_difference(u, dx, axis=1, order=1)
        nonlinear_derivative = finite_difference(u * u_x, dx, axis=1, order=1)
        features = np.column_stack([
            nonlinear_derivative[crop_slices].ravel(),
            np.ones(u[crop_slices].size),
        ])
        feature_names = ["d_x(u u_x)", "1"]
        target = u_t[crop_slices].ravel()

        return fit_manual_system(features, target, feature_names, "u_t", filename, params["optimizer"])

    if filename == "kdv_periodic_data.npy":
        u_t = finite_difference(u, dt, axis=0, order=1)
        u_x = finite_difference(u, dx, axis=1, order=1)
        u_xxx = finite_difference(u, dx, axis=1, order=3)
        x_grid = np.broadcast_to(x, u.shape)
        t_grid = np.broadcast_to(t[:, np.newaxis], u.shape)
        features = np.column_stack([
            (u[crop_slices] * u_x[crop_slices]).ravel(),
            u_xxx[crop_slices].ravel(),
            (np.sin(x_grid[crop_slices]) * np.cos(t_grid[crop_slices])).ravel(),
            np.ones(u[crop_slices].size),
            u[crop_slices].ravel(),
        ])
        feature_names = ["u u_x", "u_xxx", "sin(x) cos(t)", "1", "u"]
        target = u_t[crop_slices].ravel()

        return fit_manual_system(features, target, feature_names, "u_t", filename, params["optimizer"])

    raise ValueError(f"Unknown manual mode for dataset: {filename}")


def run_sindy(data, x, y, z, t, filename):
    """Основная логика идентификации"""
    params = sindy_params[filename]

    if params.get("manual_mode"):
        return run_manual_dataset(data, x, y, z, t, filename, params)

    data = prepare_standard_data(data, x, t, filename)

    if params.get('preprocess', {}).get('moveaxis', False):
        data = np.moveaxis(data, 0, -1)

    library = build_library(params['library'], x)
    optimizer = build_optimizer(params['optimizer'])

    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(data, t=t[1] - t[0])
    model.print(precision=4)

    result = {
        "dataset": filename.split(".")[0],
        "coefficients": model.coefficients().tolist(),
        "features": model.get_feature_names(),
    }

    return result


if __name__ == "__main__":
    all_results = []
    for dataset in DATASETS:
        print(f"\n=== Processing {dataset} ===")
        try:
            data, x, y, z, t = load_data(dataset)

            result = run_sindy(data, x, y, z, t, dataset)
            all_results.append(result)

        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
