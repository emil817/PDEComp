# WENDy clean results

The upstream IRLS parameter estimator fits all polynomial candidates, without
sparsification or access to true support. Three first-order ODE datasets are
measured; 12 datasets are unsupported. All three fits fail exact structure
recovery because of extra and/or missing terms. `Diagnostic RE` is the common
`relative_error_sum`; it does not certify a correct equation. RE for a correct
structure remains blank in the CSV.

| Dataset | Target | Status | Time, s | Library | HD | RE sum | Diagnostic RE |
|---|---|---|---:|---:|---:|---:|---:|
| lotka_data.npy | system | ok | 0.9176893 | 18 | 14 | - | 0.02576632 |
| lorenz_data.npy | system | ok | 4.769947 | 57 | 50 | - | 14.20474 |
| ODE_simple_discovery | u_t | ok | 0.0195975 | 4 | 4 | - | 2.449251 |

For Lorenz, upstream IRLS reports `WENDy iterates diverged` and returns the
iterate with the largest residual-normality p-value. This fallback is retained,
marked in `selection_policy` and model diagnostics, and preserved in
`results/wendy/native/lorenz_data.log`; it is not labelled as convergence.

Unsupported: all PDE datasets, plus the second-order scalar `ode_data.npy` and
`vdp_data.npy`, for which observed velocities are unavailable. WENDy is an ODE
parameter estimator, so these records are capability limits, not timed fits.

Artifacts: `results/wendy/native/clean_run_metrics.csv`, `clean_models.json`, and
`clean_manifest.json`; fixed capability records are under `results/wendy/`.

Reproduce all four framework/protocol combinations with
`python measure_weak_clean.py` after the setup and thread settings in `README.md`.
The raw source code is fetched at pinned revisions; manifests include SHA-256
hashes. Docker Compose configuration was validated, but Docker Engine was not
running during this collection, so the supplied images were not built or timed.
