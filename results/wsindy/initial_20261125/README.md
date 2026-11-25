# WSINDy clean results

### WSINDy (clean baseline, 2026-11-25)

Pinned upstream MSTLS, with its stock threshold-loss scan, on the common fixed
library and NumPy derivatives. No dataset-specific tuning or ground-truth
selection. Correct structures: **11/15**. `RE sum` is shown only for exact
structure recovery; `HD` counts missing and extra terms.

| Dataset | Target | Status | Time, s | Library | HD | RE sum |
|---|---|---|---:|---:|---:|---:|
| ode_data.npy | u_tt | ok | 0.209019 | 12 | 0 | 0.01661991 |
| vdp_data.npy | u_tt | ok | 0.2771276 | 12 | 2 | - |
| lotka_data.npy | system | ok | 0.4962877 | 20 | 3 | - |
| lorenz_data.npy | system | ok | 1.509584 | 60 | 0 | 0.0007661314 |
| burgers_data.mat | u_t | ok | 3.118546 | 33 | 0 | 0.0185416 |
| burgers_sln_100_data.csv | u_t | ok | 0.8576271 | 33 | 0 | 0.0002592347 |
| ac_data.npy | u_t | ok | 0.9573232 | 33 | 0 | 0.01331548 |
| kdv_data.mat | u_t | ok | 21.88089 | 33 | 0 | 0.03038906 |
| kdv_periodic_data.npy | u_t | ok | 1.061481 | 33 | 0 | 0.006400459 |
| wave_data.csv | u_tt | ok | 1.384771 | 33 | 2 | - |
| pde_compound_data.npy | u_t | ok | 1.791736 | 33 | 0 | 0.000299912 |
| pde_divide_data.npy | u_t | ok | 0.8345714 | 33 | 0 | 0.0003660788 |
| ks_data.mat | u_t | ok | 8.992818 | 33 | 0 | 0.03601402 |
| ns_data.mat | system | ok | 14.52337 | 23 | 1 | - |
| ODE_simple_discovery | u_t | ok | 0.2906429 | 11 | 0 | 0.001328941 |

Sources: `results/wsindy/clean_run_metrics.csv`, `clean_models.json`, and
`clean_manifest.json`. The manifest records input/code hashes, pinned upstream
revision and package versions. These are single clean CPU runs on Windows
(Python 3.9.25, one BLAS/PyTorch thread); runtimes were measured separately from
the historical tables and should not be interpreted as a shared-hardware race.

WENDy has no fixed sparse-optimizer mode. Its fixed CSV contains 15 explicit
`unsupported` records. Actual WENDy measurements are in the native section.

## Native WSINDy and WENDy Clean Runs

Clean CPU baseline collected on 2026-11-25, seed 0, with one numerical thread.
The same shared loader and metric evaluator score both adapters. Each runtime
includes data loading, weak library construction and fitting; dependency imports
and scoring are outside the timed interval. No hyperparameter tuning was done.

### WSINDy

The upstream convolutional weak pipeline uses a conservative polynomial library
and MSTLS. Its library differs from the fixed candidate space and omits coordinate
forcing and general nonconservative products. Exact conservative identities are
mapped to benchmark names with the proper coefficient factors. Coupled PDE/NS
mapping is unsupported. Correct structures: **7/14 measured**, one unsupported.

| Dataset | Target | Status | Time, s | Library | HD | RE sum |
|---|---|---|---:|---:|---:|---:|
| ode_data.npy | u_tt | ok | 0.1151364 | 7 | 3 | - |
| vdp_data.npy | u_tt | ok | 0.0771068 | 7 | 2 | - |
| lotka_data.npy | system | ok | 0.1637474 | 18 | 2 | - |
| lorenz_data.npy | system | ok | 0.5029552 | 57 | 1 | - |
| burgers_data.mat | u_t | ok | 0.4241641 | 19 | 0 | 1.629102e-05 |
| burgers_sln_100_data.csv | u_t | ok | 0.37997 | 19 | 0 | 3.28495e-08 |
| ac_data.npy | u_t | ok | 0.3627789 | 19 | 0 | 0.003651902 |
| kdv_data.mat | u_t | ok | 0.6455426 | 19 | 0 | 0.0001495457 |
| kdv_periodic_data.npy | u_t | ok | 0.2001222 | 19 | 8 | - |
| wave_data.csv | u_tt | ok | 0.1763436 | 19 | 0 | 0.04510591 |
| pde_compound_data.npy | u_t | ok | 0.6467937 | 19 | 0 | 0.005216803 |
| pde_divide_data.npy | u_t | ok | 0.5136178 | 19 | 10 | - |
| ks_data.mat | u_t | ok | 1.742883 | 19 | 0 | 7.779437e-06 |
| ns_data.mat | - | unsupported | - | - | - | - |
| ODE_simple_discovery | u_t | ok | 0.0590861 | 7 | 4 | - |

Artifacts: `results/wsindy/native/clean_run_metrics.csv`, `clean_models.json`, and
`clean_manifest.json`.
