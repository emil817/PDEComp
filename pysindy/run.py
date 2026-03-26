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


def build_library(lib_config, x):
    """Создание объектов Library на основе конфига"""

    lib_type = lib_config.get('type', 'polynomial')
    
    if lib_type == 'polynomial':
        return ps.PolynomialLibrary(
            degree=lib_config.get('degree', 2),
            include_bias=lib_config.get('include_bias', True)
        )
        
    elif lib_type == 'pde':
        func_lib = ps.PolynomialLibrary(
            degree=lib_config.get('poly_degree', 2),
            include_bias=lib_config.get('poly_include_bias', False)
        )
        return ps.PDELibrary(
            function_library=func_lib,
            derivative_order=lib_config.get('derivative_order', 3),
            spatial_grid=x,
            include_bias=lib_config.get('pde_include_bias', True)
        )

    elif lib_type == 'pde_custom_concat':
        func_lib = ps.PolynomialLibrary(
            degree=lib_config.get('poly_degree', 2),
            include_bias=lib_config.get('poly_include_bias', False)
        )
        base_lib = ps.PDELibrary(
            function_library=func_lib,
            derivative_order=lib_config.get('derivative_order', 2),
            spatial_grid=x,
            include_bias=lib_config.get('pde_include_bias', True)
        )
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
        return ps.SR3(
            reg_weight_lam=opt_config.get('reg_weight_lam', 0.1),
            max_iter=opt_config.get('max_iter', 30),
            tol=opt_config.get('tol', 1e-5),
            regularizer=opt_config.get('regularizer', 'L0'),
            normalize_columns=opt_config.get('normalize_columns', False)
        )
        
    raise ValueError(f"Unknown optimizer type: {opt_type}")


def run_sindy(data, x, t, filename):
    """Основная логика идентификации"""
    params = sindy_params[filename]
    
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
            
            if type(data) == list:
                data = np.array(data)
            if len(data.shape) == 1:
                data = data.T.reshape(len(t), 1)
            elif dataset in ["lotka_data.npy", "lorenz_data.npy", "ODE_simple_discovery"]:
                data = data.T
            elif len(data.shape) == 2:
                data = data.T.reshape(len(x), len(t), 1)
                
            result = run_sindy(data, x, t, dataset)
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

    save_combined_results(all_results)
    print("\nAll experiments completed!")
