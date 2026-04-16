epde_params = {
    'burgers_sln_100_data.csv': {
        'population_size': 16,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': (10, 10),
        'default_preprocessor_type': 'poly',
        'variable_names': ['u', ],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 3,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
        'eq_sparsity_interval': (1e-6, 1e-0),
        'fourier_layers': True
    },
    
    'ac_data.npy': {
        'population_size': 8,
        'training_epochs': 30,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 3,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-5, 1e-2),
        'fourier_layers': False
    },

    'kdv_data.mat': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'max_deriv_order': (1, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-5, 1e-2),
        'fourier_layers': False
    },

    'burgers_data.mat': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', ],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 3,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False
    },

    'ks_data.mat': {
        'population_size': 16,
        'training_epochs': 10,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 5,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (1, 4),
        'equation_terms_max_number': 10,
        'data_fun_pow': 1,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-0),
        'fourier_layers': False
    },

    'pde_divide_data.npy': {
        'population_size': 8,
        'training_epochs': 50,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'CacheStoredTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-9, 1e-2),
        'fourier_layers': False
    },

    'pde_compound_data.npy': {
        'population_size': 8,
        'training_epochs': 50,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'CacheStoredTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-9, 1e-2),
        'fourier_layers': False
    },

    'lorenz_data.npy': {
        'population_size': 8,
        'training_epochs': 2,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'w'],
        'max_deriv_order': (1,),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'TrigonometricTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas' : [0.8, 0.2]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'lotka_data.npy': {
        'population_size': 8,
        'training_epochs': 1,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v'],
        'max_deriv_order': (1,),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'TrigonometricTokens',
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas' : [0.8, 0.2]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'ode_data.npy': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 2),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'TrigonometricTokens, GridTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-4),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'kdv_periodic_data.npy': {
        'population_size': 12,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (1, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 1,
        'additional_tokens': 'custom_trig_tokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-10, 1e-2),
        'fourier_layers': False
    },

    'vdp_data.npy': {
        'population_size': 8,
        'training_epochs': 15,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 2),
        'equation_terms_max_number': 5,
        'data_fun_pow': 2,
        'additional_tokens': 'TrigonometricTokens, GridTokens',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-8, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (2 - 1e-8, 2 + 1e-8)
    },

    'wave_data.csv': {
        'population_size': 8,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 20,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 3,
        'additional_tokens': None,
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-12, 1e-2),
        'fourier_layers': False
    },

    'ns_data.mat': {
        'population_size': 16,
        'training_epochs': 50,
        'use_solver': False,
        'multiobjective_mode': True,
        'use_pic': True,
        'boundary': 5,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u', 'v', 'p'],
        'max_deriv_order': (1, 2, 2),
        'equation_terms_max_number': 10,
        'data_fun_pow': 1,
        'additional_tokens': None,
        'equation_factors_max_number': {'factors_num': [1, 2], 'probas': [0.8, 0.2]},
        'eq_sparsity_interval': (1e-12, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '3d'
    },

    'ODE_simple_discovery': {
        'population_size': 8,
        'training_epochs': 5,
        'use_solver': False,
        'use_pic': True,
        'boundary': 10,
        'default_preprocessor_type': 'FD',
        'variable_names': ['u'],
        'max_deriv_order': (2, 3),
        'equation_terms_max_number': 5,
        'data_fun_pow': 3,
        'additional_tokens': 'ODE_simple_discovery',
        'equation_factors_max_number': {"factors_num": [1, 2], "probas": [0.65, 0.35]},
        'eq_sparsity_interval': (1e-4, 1e-0),
        'fourier_layers': False,
        'coordinate_tensors': '1d',
        'trig_tokens_freq': (0.999, 1.001)
    },
}

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
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'ode_data.npy': {
        'manual_mode': True,
        'crop': 10,
        'library': {'type': 'polynomial', 'degree': 3, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-10, 'normalize_columns': True, 'coefficient_tol': 0.01}
    },

    'ns_data.mat': {
        'manual_mode': True,
        'library': {'type': 'polynomial', 'degree': 2, 'include_bias': True},
        'optimizer': {'type': 'STLSQ', 'threshold': 0.01, 'alpha': 1e-10, 'normalize_columns': False},
        'preprocess': {'moveaxis': True}
    },

    'ODE_simple_discovery': {
        'manual_mode': True,
        'library': {'type': 'poly_and_fourier', 'poly_degree': 2, 'poly_include_bias': True, 'n_frequencies': 1},
        'optimizer': {'type': 'STLSQ', 'threshold': 1e-6, 'alpha': 1e-12, 'normalize_columns': True}
    }
}
