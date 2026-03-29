sindy_params = {
    'ac_data.npy': {
        'library': {'type': 'pde', 'poly_degree': 3, 'poly_include_bias': False, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    },
    
    'kdv_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 5, 'alpha': 1e-5, 'normalize_columns': True}
    },

    'kdv_periodic_data.npy': {
        'manual_mode': True,
        'crop': 10,
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.05, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'burgers_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1, 'alpha': 1e-5, 'normalize_columns': True}
    },
    
    'burgers_sln_100_data.csv': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 3, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.5, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'pde_divide_data.npy': {
        'manual_mode': True,
        'crop': 10,
        'library': {'type': 'pde_custom_concat', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'pde_compound_data.npy': {
        'manual_mode': True,
        'crop': 10,
        'library': {'type': 'pde_custom_concat', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'ks_data.mat': {
        'library': {'type': 'pde', 'poly_degree': 2, 'poly_include_bias': False, 'derivative_order': 4, 'pde_include_bias': True, 'diff_kwargs': {'periodic': True}},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
    },

    'wave_data.csv': {
        'manual_mode': True,
        'crop': 10,
        'library': {'type': 'pde', 'poly_degree': 1, 'poly_include_bias': True, 'derivative_order': 2, 'pde_include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False}
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
        'manual_mode': True,
        'library': {'type': 'polynomial', 'degree': 3, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True}
    },

    'ode_data.npy': {
        'manual_mode': True,
        'library': {'type': 'polynomial', 'degree': 3, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True}
    },

    'ns_data.mat': {
        'manual_mode': True,
        'library': {'type': 'polynomial', 'degree': 2, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.005, 'alpha': 1e-10, 'normalize_columns': False},
        'preprocess': {'moveaxis': True}
    },

    'ODE_simple_discovery': {
        'manual_mode': True,
        'library': {'type': 'poly_and_fourier', 'poly_degree': 2, 'poly_include_bias': True, 'n_frequencies': 1},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-12, 'normalize_columns': True}
    }
}
