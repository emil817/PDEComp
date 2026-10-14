"""Standalone structural-recovery sweep. All writes stay next to this script."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path
import sys
import time
import warnings

# Keep repository data read-only; loaders and truth live inside this project.
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import numpy as np
import pysindy as ps
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from datasets import (DATA_DIR, load_data, get_data_axes, TRUE_COEFFICIENTS,
                      TRUE_COEFFICIENT_ALTERNATIVES)
from library import Observations, build_problem, expand_truth, make_factors, term_name


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def file_digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def seed_for(seed, *parts):
    return int(digest([seed, *parts])[:8], 16)


def settings_from_grid(grid):
    required = {"optimizers", "thresholds", "frols_iterations", "alphas", "normalize_columns",
                "coefficient_tolerances", "stlsq_max_iter", "unbias"}
    if set(grid) != required:
        raise ValueError(f"Grid keys must be {sorted(required)}")
    for key in ("optimizers", "thresholds", "frols_iterations", "alphas",
                "normalize_columns", "coefficient_tolerances"):
        if not isinstance(grid[key], list) or not grid[key] or len(set(grid[key])) != len(grid[key]):
            raise ValueError(f"Grid {key} must be a nonempty list with unique values")
    if set(grid["optimizers"]) - {"STLSQ", "FROLS"}:
        raise ValueError("Supported optimizers: STLSQ, FROLS")
    for key in ("thresholds", "alphas", "coefficient_tolerances"):
        if any(not np.isfinite(v) or v < 0 for v in grid[key]):
            raise ValueError(f"{key} must be finite and nonnegative")
    if any(type(v) is not int or v < 1 for v in grid["frols_iterations"]):
        raise ValueError("FROLS iterations must be positive integers")
    if type(grid["stlsq_max_iter"]) is not int or grid["stlsq_max_iter"] < 1:
        raise ValueError("stlsq_max_iter must be a positive integer")
    if type(grid["unbias"]) is not bool or any(type(v) is not bool for v in grid["normalize_columns"]):
        raise ValueError("unbias and normalize_columns must be booleans")
    settings = []
    for optimizer in grid["optimizers"]:
        controls = grid["thresholds"] if optimizer == "STLSQ" else grid["frols_iterations"]
        for control, alpha, normalize in product(controls, grid["alphas"], grid["normalize_columns"]):
            settings.append(dict(optimizer=optimizer,
                                 threshold=control if optimizer == "STLSQ" else None,
                                 max_iter=grid["stlsq_max_iter"] if optimizer == "STLSQ" else control,
                                 alpha=alpha, normalize_columns=normalize, unbias=grid["unbias"]))
    return settings


def optimizer_for(setting):
    kwargs = {k: v for k, v in setting.items() if k not in ("optimizer", "threshold")}
    cls = getattr(ps, setting["optimizer"])
    if "unbias" not in inspect.signature(cls).parameters:
        raise RuntimeError("This experiment requires PySINDy 2.1.0; install requirements.txt")
    if setting["optimizer"] == "STLSQ":
        kwargs["threshold"] = setting["threshold"]
    return cls(**kwargs)


def observations_for(dataset, noise_percent, run, seed):
    data, x, y, z, t = load_data(dataset)
    arrays = data if isinstance(data, list) else [data]
    names = (["u", "v", "p"] if dataset == "ns_data.mat" else
             ["u"] if len(arrays) == 1 else [f"x{i}" for i in range(len(arrays))])
    rng = np.random.default_rng(seed_for(seed, "noise", dataset, noise_percent, run))
    fields = {}
    for name, values in zip(names, arrays):
        values = np.asarray(values, dtype=float)
        fields[name] = values + noise_percent * 0.01 * np.std(values) * rng.standard_normal(values.shape)
    return Observations(fields, get_data_axes(arrays[0], x, y, z, t))


def target_names(dataset, lhs):
    expected = list(TRUE_COEFFICIENTS[dataset])
    if lhs == "both" and len(expected) == 1:
        return ["u_t", "u_tt"]
    return expected


def truth_forms(dataset, target):
    primary = TRUE_COEFFICIENTS[dataset].get(target)
    if primary is None:
        return []
    return [expand_truth(form) for form in
            [primary, *TRUE_COEFFICIENT_ALTERNATIVES.get(dataset, {}).get(target, [])]]


def score_fit(record, tolerance):
    base = {k: v for k, v in record.items() if k not in ("coefficients", "feature_keys")}
    base["coefficient_tolerance"] = tolerance
    if record["status"] != "ok":
        return dict(base, structure_success=False, hamming=None, coefficient_error=None,
                    missing_terms=[], extra_terms=[], active_terms=[], active_coefficients={})
    coefficients = {tuple((atom, power) for atom, power in key): value
                    for key, value in zip(record["feature_keys"], record["coefficients"])}
    active = {key for key, value in coefficients.items() if abs(value) > tolerance}
    forms = truth_forms(record["dataset"], record["target"])
    base.update(active_terms=sorted(term_name(key) for key in active),
                active_coefficients={term_name(key): coefficients[key] for key in active})
    if not forms:
        return dict(base, structure_success=False, hamming=None, coefficient_error=None,
                    missing_terms=[], extra_terms=[term_name(key) for key in active])
    matches = []
    for form in forms:
        missing, extra = set(form) - active, active - set(form)
        error = sum(abs(coefficients.get(key, 0.0) - value) / abs(value) for key, value in form.items())
        matches.append((len(missing) + len(extra), error, missing, extra))
    hamming, error, missing, extra = min(matches, key=lambda m: m[:2])
    return dict(base, structure_success=hamming == 0, hamming=hamming,
                coefficient_error=error if hamming == 0 else None,
                missing_terms=sorted(term_name(key) for key in missing),
                extra_terms=sorted(term_name(key) for key in extra))


def write_csv(path, rows):
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    temp.replace(path)


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def export_results(output, records, args, settings, grid):
    rows = [score_fit(record, tolerance) for record in records
            for tolerance in grid["coefficient_tolerances"]]
    write_csv(output / "trials.csv", rows)
    grouped = {}
    for row in rows:
        key = row["dataset"], row["noise_percent"], row["setting_id"], row["coefficient_tolerance"]
        grouped.setdefault(key, []).append(row)
    dataset_summary, target_summary = [], []
    for dataset, noise, setting in product(args.datasets, args.noise_levels, settings):
        sid = digest(setting)
        expected = list(TRUE_COEFFICIENTS[dataset])
        for tolerance in grid["coefficient_tolerances"]:
            group = grouped.get((dataset, noise, sid, tolerance), [])
            common = dict(dataset=dataset, noise_percent=noise, setting_id=sid,
                          **setting, coefficient_tolerance=tolerance)
            completed, correct = 0, 0
            for run in range(args.runs):
                run_rows = {r["target"]: r for r in group if r["run"] == run}
                ok = all(t in run_rows and run_rows[t]["status"] == "ok" for t in expected)
                completed += int(ok)
                correct += int(ok and all(run_rows[t]["structure_success"] for t in expected))
            dataset_summary.append(dict(common, planned_runs=args.runs, completed_runs=completed,
                                        failed_or_missing_runs=args.runs - completed,
                                        correct_count=correct, success_rate=correct / args.runs))
            for target in expected:
                target_rows = [r for r in group if r["target"] == target]
                ok_rows = [r for r in target_rows if r["status"] == "ok"]
                count = sum(r["structure_success"] for r in ok_rows)
                target_summary.append(dict(common, target=target, planned_runs=args.runs,
                                           completed_runs=len(ok_rows), correct_count=count,
                                           success_rate=count / args.runs,
                                           hamming_mean=float(np.mean([r["hamming"] for r in ok_rows]))
                                           if ok_rows else None))
    write_csv(output / "by_dataset.csv", dataset_summary)
    write_csv(output / "by_target.csv", target_summary)
    datasets_by_setting, targets_by_setting, datasets_by_noise = {}, {}, {}
    for row in dataset_summary:
        key = row["noise_percent"], row["setting_id"], row["coefficient_tolerance"]
        datasets_by_setting.setdefault(key, []).append(row)
        datasets_by_noise.setdefault((row["dataset"], row["noise_percent"]), []).append(row)
    for row in target_summary:
        key = row["noise_percent"], row["setting_id"], row["coefficient_tolerance"]
        targets_by_setting.setdefault(key, []).append(row)
    totals = []
    for noise, setting, tolerance in product(args.noise_levels, settings, grid["coefficient_tolerances"]):
        sid = digest(setting)
        selected = datasets_by_setting[noise, sid, tolerance]
        targets = targets_by_setting[noise, sid, tolerance]
        totals.append(dict(noise_percent=noise, setting_id=sid, **setting,
                           coefficient_tolerance=tolerance,
                           planned_dataset_runs=len(args.datasets) * args.runs,
                           completed_dataset_runs=sum(r["completed_runs"] for r in selected),
                           correct_dataset_runs=sum(r["correct_count"] for r in selected),
                           datasets_with_any_success=sum(r["correct_count"] > 0 for r in selected),
                           planned_equation_runs=len(targets) * args.runs,
                           correct_equation_runs=sum(r["correct_count"] for r in targets)))
    write_csv(output / "by_setting.csv", totals)
    # Explicitly diagnostic: choosing a separate setting using ground truth.
    best = []
    for dataset, noise in product(args.datasets, args.noise_levels):
        selected = datasets_by_noise[dataset, noise]
        winner = max(selected, key=lambda r: (r["correct_count"], r["completed_runs"]))
        best.append(dict(winner, selection_policy="ground_truth_oracle"))
    write_csv(output / "oracle_best_by_dataset.csv", best)


def fit_problem(problem, setting):
    optimizer = optimizer_for(setting)
    # Remove exact zero columns, restore their coefficients as zero afterwards.
    keep = np.any(problem.matrix != 0, axis=0)
    optimizer.fit(problem.matrix[:, keep], problem.response)
    coefficients = np.zeros(len(problem.names))
    coefficients[keep] = np.asarray(optimizer.coef_).ravel()
    if not np.isfinite(coefficients).all():
        raise ValueError("Optimizer produced non-finite coefficients")
    return coefficients


def manifest_for(args, grid, settings):
    inputs = [DATA_DIR / dataset for dataset in args.datasets if (DATA_DIR / dataset).is_file()]
    for dataset in args.datasets:
        if dataset in ("lorenz_data.npy", "lotka_data.npy"):
            inputs.append(DATA_DIR / dataset.replace("_data", "_time"))
    sources = [HERE / "library.py", HERE / "run_sweep.py", HERE / "datasets.py"]
    parameters = {k: v for k, v in vars(args).items()
                  if k not in ("output_dir", "resume", "dry_run", "grid")}
    return dict(schema_version=1, parameters=parameters, grid=grid, settings=settings,
                versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                              sklearn=sklearn.__version__, pysindy=ps.__version__),
                input_hashes={str(p.relative_to(ROOT)): file_digest(p) for p in inputs + sources})


def read_checkpoint(path):
    if not path.exists():
        return []
    records = []
    # Recover a partially written final line after an interrupted process.
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.read():
                    raise ValueError("Corrupt checkpoint before the last line")
                handle.seek(offset)
                handle.truncate()
                break
            records.append(record)
            if not line.endswith(b"\n"):
                # A valid JSON record can still lack its delimiter after a crash.
                handle.seek(0, 2)
                handle.write(b"\n")
    if len({r["fit_id"] for r in records}) != len(records):
        raise ValueError("Duplicate fit IDs in checkpoint")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(TRUE_COEFFICIENTS), choices=list(TRUE_COEFFICIENTS))
    parser.add_argument("--grid", type=Path, default=HERE / "paper_grid.json")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0], help="Percent of field standard deviation")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", type=int, default=5, help="Same number of boundary points on every axis")
    parser.add_argument("--max-samples", type=int, default=10000, help="Sample after differentiation/crop; 0 = all rows")
    parser.add_argument("--differentiation", choices=["fd", "gradient", "smoothed"], default="fd")
    parser.add_argument("--fd-order", type=int, default=2, help="Finite difference accuracy order")
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--lhs", choices=["known", "both"], default="known")
    parser.add_argument("--threads", type=int, default=1, help="BLAS threads per fit")
    parser.add_argument("--limit-settings", type=int, help="Use the first N optimizer settings for a smoke test")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Check libraries/coverage and write a plan; do not fit")
    args = parser.parse_args()
    if (args.runs < 1 or args.crop < 0 or args.max_samples < 0 or args.threads < 1 or
            args.fd_order < 1 or args.smoothing_window < 5 or args.smoothing_window % 2 != 1 or
            any(not np.isfinite(v) or v < 0 for v in args.noise_levels) or
            (args.limit_settings is not None and args.limit_settings < 1)):
        parser.error("Invalid run count, crop, sample size, differentiation, threads, or noise")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.noise_levels)) != len(args.noise_levels):
        parser.error("Datasets and noise levels must not contain duplicates")
    if args.resume and (args.output_dir is None or args.dry_run):
        parser.error("--resume requires --output-dir and cannot be used with --dry-run")
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    settings = settings_from_grid(grid)
    if args.limit_settings:
        settings = settings[:args.limit_settings]
    # Fail before creating output when the optimizer environment is incompatible.
    for setting in settings:
        optimizer_for(setting)
    output = (args.output_dir or HERE / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        parser.error("--output-dir must be a subdirectory of projects/pysindy_sweep")
    manifest = manifest_for(args, grid, settings)
    if args.resume:
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if old != manifest:
            parser.error("Resume parameters, versions, source or dataset hashes do not match manifest.json")
    elif output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a fresh directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    checkpoint = output / "fits.jsonl"
    records = read_checkpoint(checkpoint) if args.resume else []
    done = {r["fit_id"] for r in records}
    library_manifest = (json.loads((output / "libraries.json").read_text(encoding="utf-8"))
                        if args.resume and (output / "libraries.json").exists() else {})
    planned = sum(len(target_names(ds, args.lhs)) for ds in args.datasets) * len(args.noise_levels) * args.runs * len(settings)
    print(f"Output: {output}\nOptimizer settings: {len(settings)}; scored settings: "
          f"{len(settings) * len(grid['coefficient_tolerances'])}; fits: {planned}", flush=True)
    interrupted = False
    with checkpoint.open("a", encoding="utf-8") as handle, threadpool_limits(limits=args.threads):
        try:
            for dataset, noise, run in product(args.datasets, args.noise_levels, range(args.runs)):
                print(f"{dataset}, noise={noise:g}%, run={run + 1}/{args.runs}", flush=True)
                targets = target_names(dataset, args.lhs)
                if not args.dry_run and all(digest([dataset, noise, run, t, s]) in done for t in targets for s in settings):
                    continue
                prep_started = time.perf_counter()
                try:
                    obs = observations_for(dataset, noise, run, args.seed)
                    factors, derivatives = make_factors(obs, args.differentiation, args.fd_order, args.smoothing_window)
                    prep_error = None
                except Exception as exc:
                    prep_error = f"{type(exc).__name__}: {exc}"
                prep_seconds = time.perf_counter() - prep_started
                for target in targets:
                    problem = None
                    error = prep_error
                    started = time.perf_counter()
                    if error is None:
                        try:
                            problem = build_problem(factors, derivatives, target, args.crop, args.max_samples,
                                                    seed_for(args.seed, "rows", dataset, run))
                            forms = truth_forms(dataset, target)
                            covered = [all(key in problem.keys for key in form) for form in forms]
                            if covered and not any(covered):
                                raise ValueError("General library does not cover any accepted true structure")
                            library_id = digest([dataset, noise, run, target])
                            library_manifest[library_id] = dict(dataset=dataset, noise_percent=noise, run=run,
                                target=target, features=problem.names, library_size=len(problem.names),
                                full_rows=problem.full_rows, valid_rows=problem.valid_rows,
                                fitted_rows=len(problem.response), zero_columns=problem.dropped_zero_columns,
                                truth_coverage=covered,
                                row_indices_sha256=hashlib.sha256(problem.row_indices.tobytes()).hexdigest())
                            write_json(output / "libraries.json", library_manifest)
                            print(f"  {target}: {len(problem.names)} features, {len(problem.response)} rows", flush=True)
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                    library_seconds = time.perf_counter() - started
                    if args.dry_run:
                        if error:
                            raise RuntimeError(f"{dataset}/{target}: {error}")
                        continue
                    for setting in settings:
                        fit_id = digest([dataset, noise, run, target, setting])
                        if fit_id in done:
                            continue
                        record = dict(fit_id=fit_id, dataset=dataset, target=target, noise_percent=noise,
                                      run=run, setting_id=digest(setting), **setting,
                                      library_size=len(problem.names) if problem else None,
                                      fitted_rows=len(problem.response) if problem else None,
                                      preparation_seconds=prep_seconds, library_seconds=library_seconds,
                                      status="error", error=error or "", coefficients=[], feature_keys=[], warnings=[])
                        fit_started = time.perf_counter()
                        if error is None:
                            try:
                                with warnings.catch_warnings(record=True) as caught:
                                    warnings.simplefilter("always")
                                    coefficients = fit_problem(problem, setting)
                                record.update(status="ok", coefficients=coefficients.tolist(),
                                              feature_keys=problem.keys,
                                              warnings=list(dict.fromkeys(str(w.message) for w in caught)))
                            except Exception as exc:
                                record["error"] = f"{type(exc).__name__}: {exc}"
                        record["fit_seconds"] = time.perf_counter() - fit_started
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                        handle.flush()
                        records.append(record)
                        done.add(fit_id)
                        if record["status"] == "error":
                            print(f"  ERROR: {record['error']}", flush=True)
                        elif len(done) % 10 == 0:
                            print(f"  Fits completed: {len(done)}/{planned}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saved fits can be resumed.", flush=True)
    if args.dry_run:
        print("Library and ground-truth coverage checks passed; no regression fits performed.")
        return 0
    export_results(output, records, args, settings, grid)
    errors = sum(r["status"] == "error" for r in records)
    print(f"Saved {len(records)}/{planned} fits ({errors} errors) and CSV summaries to {output}")
    return 130 if interrupted else 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
