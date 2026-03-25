import numpy as np
import pysindy as ps
from pathlib import Path
import scipy.io as scio
import json
from datetime import datetime
import sys
sys.path.append(str(Path().absolute()))

from data.dataloader import load_data


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

    result = []

    result.append(results)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)


def run_sindy(data, x, t, filename):
    """Основная логика идентификации"""
    if filename == "ac_data.npy":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=3, include_bias=False),
            # function_names=library_function_names,
            derivative_order=3,
            spatial_grid=x,
            # temporal_grid=t,
            include_bias=True,
        ).fit(data)
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)

    elif filename == "kdv_data.mat":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=5, alpha=1e-5, normalize_columns=True)

    elif filename == "kdv_periodic_data.npy":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=0.1, alpha=1e-5, normalize_columns=True)

    elif filename == "burgers_data.mat":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=True)
    
    elif filename == "burgers_sln_100_data.csv":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=3,
            spatial_grid=x,
            include_bias=True,
        ).fit(data)

        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=10, alpha=1e-5, normalize_columns=True)

    elif filename == "pde_divide_data.npy" or filename == "pde_compound_data.npy":
        functions = [lambda x : 1/x]
        functions_names = [lambda x : "1/" + x]
        
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=2,
            spatial_grid=x,
            include_bias=True,
        )

        lib_custom = ps.CustomLibrary(library_functions=functions, function_names=functions_names) * library

        library = ps.ConcatLibrary([lib_custom, library]).fit(data)
        print(library.get_feature_names(), "\n")

        optimizer = ps.STLSQ(threshold=1, alpha=1e-5, normalize_columns=False)
        # optimizer = ps.SR3(tol=1e-15, normalize_columns=True, max_iter=10000)
        # optimizer = ps.SR3(threshold=5, max_iter=10000, tol=1e-15, thresholder='l1', normalize_columns=True)
        # optimizer = ps.FROLS(normalize_columns=True, kappa=1e-5)
        # optimizer = ps.SSR(normalize_columns=True, kappa=5e-3)

    elif filename == "ks_data.mat":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=2, include_bias=False),
            derivative_order=4,
            spatial_grid=x,
            include_bias=True,
        )
        optimizer = ps.SR3(reg_weight_lam=0.8, max_iter=5000, tol=1e-6, regularizer='L0', normalize_columns=True)

    elif filename == "wave_data.csv":
        library = ps.PDELibrary(
            function_library=ps.PolynomialLibrary(degree=1, include_bias=True),
            derivative_order=2,
            spatial_grid=x,
            include_bias=True,
        )
        optimizer = ps.STLSQ(threshold=0.1)

    elif filename == "lorenz_data.npy":
        library = ps.PolynomialLibrary(degree=2)
        optimizer = ps.STLSQ(threshold=0.1)

    elif filename == "lotka_data.npy":
        library = ps.PolynomialLibrary(degree=2, include_bias=False)
        optimizer = ps.STLSQ(threshold=0.1)

    elif filename == "vdp_data.npy":
        library = ps.PolynomialLibrary(degree=3, include_bias=True)
        
        optimizer = ps.STLSQ(
            threshold=0.05, 
            alpha=1e-05, 
            normalize_columns=True
        )

    elif filename == "ode_data.npy":
        library = ps.PolynomialLibrary(degree=3)
        optimizer = ps.STLSQ(threshold=0.01)

    elif filename == "ns_data.mat":
        data = np.moveaxis(data, 0, -1)

        library = ps.PolynomialLibrary(degree=2)
        optimizer = ps.STLSQ(threshold=0.1)

    elif filename == "ODE_simple_discovery":
        poly_lib = ps.PolynomialLibrary(degree=2, include_bias=True)
        
        trig_lib = ps.FourierLibrary(n_frequencies=1)
        
        library = poly_lib + trig_lib
        
        optimizer = ps.STLSQ(
            threshold=0.01, 
            alpha=1e-5, 
            normalize_columns=True
        )


    model = ps.SINDy(optimizer=optimizer, feature_library=library)
    model.fit(data, t=t[1] - t[0])
    model.print(precision=4)

    result = {
        "dataset": filename.split(".")[0],
        "coefficients": model.coefficients().tolist(),
        "features": model.get_feature_names(),
        # "model_str": str(model.print(precision=4))
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
