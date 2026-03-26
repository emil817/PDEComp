sindy_params = {
    'ac_data.npy': {
        'library': {'type': 'pde', 'poly_degree': 3, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    },
    
    'kdv_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 5, 'alpha': 1e-5, 'normalize_columns': True}
    },

    'kdv_periodic_data.npy': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 1e-5, 'normalize_columns': True}
    },

    'burgers_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    },
    
    'burgers_sln_100_data.csv': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 10, 'alpha': 1e-5, 'normalize_columns': True}
    },

    'pde_divide_data.npy': {
        'library': {'type': 'pde_custom_concat', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': False}
    },

    'pde_compound_data.npy': {
        'library': {'type': 'pde_custom_concat', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': False}
    },

    'ks_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 4, 'pde_include_bias': True},
        'optimizer': {'type': 'SR3', 'reg_weight_lam': 0.8, 'max_iter': 5000, 'tol': 1e-6, 'regularizer': 'L0', 'normalize_columns': True}
    },

    'wave_data.csv': {
        'library': {'type': 'pde', 'poly_degree': 1, 'poly_include_bias': True, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 0.05, 'normalize_columns': False}
    },

    'lorenz_data.npy': {
        'library': {'type': 'polynomial', 'degree': 2, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 0.05, 'normalize_columns': False}
    },

    'lotka_data.npy': {
        'library': {'type': 'polynomial', 'degree': 2, 'include_bias': False},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 0.05, 'normalize_columns': False}
    },

    'vdp_data.npy': {
        'library': {'type': 'polynomial', 'degree': 3, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.05, 'alpha': 1e-05, 'normalize_columns': True}
    },

    'ode_data.npy': {
        'library': {'type': 'polynomial', 'degree': 3, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 0.05, 'normalize_columns': False}
    },

    'ns_data.mat': {
        'library': {'type': 'polynomial', 'degree': 2, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.1, 'alpha': 0.05, 'normalize_columns': False},
        'preprocess': {'moveaxis': True}
    },

    'ODE_simple_discovery': {
        'library': {'type': 'poly_and_fourier', 'poly_degree': 2, 'poly_include_bias': True, 'n_frequencies': 1},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-5, 'normalize_columns': True}
    }
}