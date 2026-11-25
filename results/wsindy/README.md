# WSINDy clean results

### WSINDy (clean baseline, 2026-11-25)

Pinned upstream MSTLS, with its stock threshold-loss scan, on the common fixed
library and NumPy derivatives. No dataset-specific tuning or ground-truth
selection. All 15 datasets measured; correct structures: **11/15**.
`RE sum` is reported only for exact structure recovery; `Diagnostic RE` also
includes incorrect structures. `HD` counts missing and extra terms.

| Dataset | Target | Status | Time, s | Library | HD | RE sum | Diagnostic RE |
|---|---|---|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | ok | 0.0816402 | 12 | 0 | 0.01661991 | 0.01661991 |
| vdp_data.npy | u_tt | ok | 0.3514566 | 12 | 2 | - | 2.006687 |
| lotka_data.npy | system | ok | 0.1772265 | 20 | 3 | - | 0.08656562 |
| lorenz_data.npy | system | ok | 0.5860376 | 60 | 0 | 0.0007661314 | 0.0007661314 |
| burgers_data.mat | u_t | ok | 1.677508 | 33 | 0 | 0.0185416 | 0.0185416 |
| burgers_sln_100_data.csv | u_t | ok | 0.2407245 | 33 | 0 | 0.0002592347 | 0.0002592347 |
| ac_data.npy | u_t | ok | 0.4918582 | 33 | 0 | 0.01331548 | 0.01331548 |
| kdv_data.mat | u_t | ok | 9.214448 | 33 | 0 | 0.03038906 | 0.03038906 |
| kdv_periodic_data.npy | u_t | ok | 0.3581998 | 33 | 0 | 0.006400459 | 0.006400459 |
| wave_data.csv | u_tt | ok | 0.4952316 | 33 | 2 | - | 0.006484156 |
| pde_compound_data.npy | u_t | ok | 1.114266 | 33 | 0 | 0.000299912 | 0.000299912 |
| pde_divide_data.npy | u_t | ok | 0.9682045 | 33 | 0 | 0.0003660788 | 0.0003660788 |
| ks_data.mat | u_t | ok | 11.04037 | 33 | 0 | 0.03601402 | 0.03601402 |
| ns_data.mat | system | ok | 9.951943 | 23 | 1 | - | 1.340515 |
| ODE_simple_discovery | u_t | ok | 0.0916481 | 11 | 0 | 0.001328941 | 0.001328941 |

Artifacts: `results/wsindy/clean_run_metrics.csv`, `clean_models.json`,
`clean_manifest.json`, and `coverage.json`. The fixed fitting procedure is
unchanged from the initial integration; this table records a new complete run.
Single clean CPU runs on Windows/Python 3.9.25 used one numerical thread.
Historical framework runtimes were collected separately and are not a
shared-hardware speed comparison.

WENDy has no fixed sparse-optimizer mode; all 15 capability records explain
this limitation. Actual WENDy measurements are in the native section.

## Native coverage

The expanded native adapter measures **all 15 datasets**, including NS.
Correct structures: **9/15**. There are no unsupported native datasets and no
execution errors in this run. Coordinate forcing and variable-coefficient
terms use upstream custom weak columns. Their integration-by-parts identities
include coefficient derivatives and preserve the physical coefficient scale.
Reciprocal terms are offered only on spatial domains that do not cross zero.

For NS, the raw velocity/pressure fields feed upstream weak convolutions.
Nonconservative advection uses SciPy cubic-spline spatial derivatives before
weak integration; this is an explicit adapter extension. No shared fixed
feature matrix or derivative bundle is passed to the native optimizer.
The divergence constraint is recovered (`HD=0`, RE about `6.42e-6`), while both
momentum equations omit viscosity. The aggregate NS structure is therefore
incorrect (`HD=2`), rather than unsupported.

| Dataset | Target | Status | Time, s | Library | HD | RE sum | Diagnostic RE |
|---|---|---|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | ok | 0.1410788 | 13 | 0 | 0.005383019 | 0.005383019 |
| vdp_data.npy | u_tt | ok | 0.0802315 | 13 | 2 | - | 2.000755 |
| lotka_data.npy | system | ok | 0.1465358 | 18 | 2 | - | 1.004037 |
| lorenz_data.npy | system | ok | 0.3244573 | 57 | 1 | - | 1.142248 |
| burgers_data.mat | u_t | ok | 0.20819 | 28 | 0 | 1.629102e-05 | 1.629102e-05 |
| burgers_sln_100_data.csv | u_t | ok | 0.1671068 | 28 | 0 | 3.28495e-08 | 3.28495e-08 |
| ac_data.npy | u_t | ok | 0.1848812 | 28 | 0 | 0.003651902 | 0.003651902 |
| kdv_data.mat | u_t | ok | 0.416497 | 28 | 0 | 0.0001495457 | 0.0001495457 |
| kdv_periodic_data.npy | u_t | ok | 0.2238046 | 28 | 14 | - | 4.951019 |
| wave_data.csv | u_tt | ok | 0.2161913 | 29 | 2 | - | 0.004879964 |
| pde_compound_data.npy | u_t | ok | 0.2940162 | 30 | 0 | 0.005216803 | 0.005216803 |
| pde_divide_data.npy | u_t | ok | 0.211612 | 30 | 0 | 0.007057415 | 0.007057415 |
| ks_data.mat | u_t | ok | 1.022585 | 30 | 0 | 7.779437e-06 | 7.779437e-06 |
| ns_data.mat | system | ok | 1.646377 | 23 | 2 | - | 2.17721 |
| ODE_simple_discovery | u_t | ok | 0.0713885 | 12 | 0 | 7.745088e-06 | 7.745088e-06 |

Compared with the initial adapter, `ode_data.npy`, `pde_divide_data.npy` and
`ODE_simple_discovery` now recover the structure. The larger candidate library
makes the wave result incorrect; that regression is retained, with no
truth-based choice between the old and new fits. Periodic KdV, Van der Pol,
Lotka-Volterra, Lorenz and NS also remain structurally incorrect.

Current artifacts are `results/<framework>/[native/]clean_run_metrics.csv`,
`clean_models.json`, `clean_manifest.json`, and `coverage.json`. The initial
limited adapters and measurements are preserved under
`results/<framework>/initial_20261125/`; archived metrics are not current results.

Reproduce with `python measure_weak_clean.py` after the setup and thread settings
in `README.md`. Source revisions, packages, input/code hashes and auxiliary fits
are recorded. All 38 tests pass, including the second-order ODE lift, weak-form
coefficient scaling and spatial divergence constraint. Docker Compose validates;
Docker Engine was unavailable, so image builds were not tested.
