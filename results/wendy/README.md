# WENDy clean results

The expanded adapter measures **all five ODE datasets**. Correct structures:
**1/5**, on `ODE_simple_discovery`. The ten PDE datasets are the only native
`unsupported` records: the pinned pyWENDy implementation forms first-order ODE
weak equations and has no spatial PDE library or spatial noise covariance.

Second-order scalar ODEs are lifted to `(u, u_t)` using a not-a-knot cubic spline
to estimate velocity from the observed position. Only the requested acceleration
equation is scored. A clock state supplies explicit time, sine and cosine
features; auxiliary kinematic and clock fits are saved in the model JSON.
The upstream diagonal observation-noise model does not account for correlations
between position and derived velocity. These clean results therefore describe
this adapter pipeline, not a validation of noisy lifted-state uncertainty.

Upstream IRLS estimates every candidate coefficient without sparse selection.
No true support, coefficient threshold or truth-based model selection is used.
`Diagnostic RE` measures coefficient error even for an incorrect structure and
must not be interpreted as successful equation discovery.

| Dataset | Target | Status | Time, s | Library | HD | RE sum | Diagnostic RE |
|---|---|---|---:|---:|---:|---:|---:|
| ode_data.npy | u_tt | ok | 0.5239925 | 15 | 12 | - | 0.05357378 |
| vdp_data.npy | u_tt | ok | 0.2279695 | 15 | 12 | - | 0.006598474 |
| lotka_data.npy | system | ok | 0.4215355 | 28 | 24 | - | 0.03137663 |
| lorenz_data.npy | system | ok | 5.056447 | 72 | 65 | - | 14.93968 |
| burgers_data.mat | - | unsupported | - | - | - | - | - |
| burgers_sln_100_data.csv | - | unsupported | - | - | - | - | - |
| ac_data.npy | - | unsupported | - | - | - | - | - |
| kdv_data.mat | - | unsupported | - | - | - | - | - |
| kdv_periodic_data.npy | - | unsupported | - | - | - | - | - |
| wave_data.csv | - | unsupported | - | - | - | - | - |
| pde_compound_data.npy | - | unsupported | - | - | - | - | - |
| pde_divide_data.npy | - | unsupported | - | - | - | - | - |
| ks_data.mat | - | unsupported | - | - | - | - | - |
| ns_data.mat | - | unsupported | - | - | - | - | - |
| ODE_simple_discovery | u_t | ok | 0.1115037 | 9 | 0 | 7.745088e-06 | 7.745088e-06 |

The expanded Lotka-Volterra fit triggers the upstream residual-normality fallback
and returns an earlier iterate; this is marked in its CSV selection policy,
model diagnostics and `lotka_data.log`. Lorenz now returns an iterate without
that fallback, but still has poor coefficient and structural accuracy. Neither
case is hidden as an unsupported dataset.

The fixed protocol is unsupported for all 15 datasets; use native.

Current artifacts are `results/<framework>/[native/]clean_run_metrics.csv`,
`clean_models.json`, `clean_manifest.json`, and `coverage.json`. The initial
limited adapters and measurements are preserved under
`results/<framework>/initial_20261125/`; archived metrics are not current results.

Reproduce with `python measure_weak_clean.py` after the setup and thread settings
in `README.md`. Source revisions, packages, input/code hashes and auxiliary fits
are recorded. All 38 tests pass, including the second-order ODE lift, weak-form
coefficient scaling and spatial divergence constraint. Docker Compose validates;
Docker Engine was unavailable, so image builds were not tested.
