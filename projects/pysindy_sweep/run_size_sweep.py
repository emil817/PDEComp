"""Library-size sweep with complete grammars, shared rows and resumable jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from threadpoolctl import threadpool_limits

from library import expand_truth, term_name
from run_sweep import (TRUE_COEFFICIENTS, digest, file_digest, fit_problem,
                       manifest_for, observations_for, optimizer_for, read_checkpoint,
                       score_fit, seed_for, settings_from_grid, truth_forms, write_csv, write_json)
from variable_library import (FactorBank, LibrarySpec, contains_term, enumerate_keys, factor_keys)


def read_catalog(path, requested=None):
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1 or not catalog.get("libraries"):
        raise ValueError("Expected a nonempty schema_version=1 library catalog")
    entries = catalog["libraries"]
    identifiers = [name for entry in entries for name in [entry["id"], *entry.get("aliases", [])]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Catalog IDs/aliases must be unique")
    for entry in entries:
        LibrarySpec(**entry["spec"])
    if requested:
        unknown = set(requested) - set(identifiers)
        if unknown:
            raise ValueError(f"Unknown library IDs: {sorted(unknown)}")
        entries = [entry for entry in entries
                   if set(requested).intersection([entry["id"], *entry.get("aliases", [])])]
    return dict(catalog, libraries=entries)


def benchmark_and_full_ac_forms(dataset, target):
    forms = truth_forms(dataset, target)
    full_ac = [expand_truth({"u": 5., "u^3": -5., "u_xx": 1e-4})] if dataset == "ac_data.npy" else []
    return forms, full_ac


def metadata_rows(obs, crop, max_samples, seed):
    shape = next(iter(obs.fields.values())).shape
    if any(dim <= 2 * crop for dim in shape):
        raise ValueError(f"Crop {crop} leaves no interior for {shape}")
    slices = tuple(slice(crop, -crop) if crop else slice(None) for _ in shape)
    indices = np.arange(np.prod(shape)).reshape(shape)[slices].ravel()
    valid = np.ones(shape, dtype=bool)
    for _, axis, grid in obs.axes:
        grid_shape = [1] * len(shape)
        grid_shape[axis] = len(grid)
        valid &= np.abs(grid.reshape(grid_shape)) > 1e-12
    rows = indices[valid.ravel()[indices]]
    if max_samples and len(rows) > max_samples:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_samples, replace=False))
    return rows


def make_plan(args, catalog, settings, grid):
    entries, plan = catalog["libraries"], []
    for dataset in args.datasets:
        obs = observations_for(dataset, 0., 0, args.seed)
        fields = list(obs.fields)
        axes = [name for name, _, _ in obs.axes]
        rows = metadata_rows(obs, args.crop, args.max_samples,
                             seed_for(args.seed, "rows", dataset, 0))
        targets = list(TRUE_COEFFICIENTS[dataset])
        for target, entry in product(targets, entries):
            spec = LibrarySpec(**entry["spec"])
            forms, full_ac = benchmark_and_full_ac_forms(dataset, target)
            covered = [all(contains_term(key, fields, axes, target, spec) for key in form) for form in forms]
            full_covered = [all(contains_term(key, fields, axes, target, spec) for key in form) for form in full_ac]
            if not any(covered) or (full_covered and not any(full_covered)):
                raise ValueError(f"Incomplete library {entry['id']} for {dataset}/{target}; "
                                 "no fits are allowed unless all selected laws are covered")
            keys, count = enumerate_keys(fields, axes, target, spec, args.max_features)
            matrix_mb = count * len(rows) * 8 / 2**20
            # Conservative fit workspace proxy: matrix/copies and two dense p*p arrays.
            working_mb = 3 * matrix_mb + 16 * count**2 / 2**20
            reasons = []
            if keys is None:
                reasons.append("max_features")
            if matrix_mb > args.max_matrix_mb:
                reasons.append("max_matrix_mb")
            if working_mb > args.max_working_mb:
                reasons.append("max_working_mb")
            if reasons:
                raise ValueError(f"Library {entry['id']} exceeds the resource budget for {dataset}/{target}: "
                                 f"{','.join(reasons)}; features={'>' if keys is None else ''}{count}, "
                                 f"matrix={matrix_mb:.1f} MiB, working={working_mb:.1f} MiB. "
                                 "Reduce the catalog parameter bounds; this experiment does not skip pairs.")
            effective = digest(keys) if keys is not None else None
            plan.append(dict(dataset=dataset, target=target, library_id=entry["id"],
                             aliases=entry.get("aliases", []), groups=entry.get("groups", []),
                             **spec.as_dict(), library_size=count, size_is_lower_bound=keys is None,
                             effective_library_id=effective, truth_coverage=covered,
                             full_ac_truth_coverage=full_covered, estimated_rows=len(rows),
                             estimated_matrix_mb=round(matrix_mb, 3),
                             estimated_fit_working_mb=round(working_mb, 3),
                             state="ready", skip_reason=""))
        del obs
    ready = {(p["dataset"], p["target"], p["effective_library_id"]) for p in plan if p["state"] == "ready"}
    repetitions = len(args.noise_levels) * args.runs
    summary = dict(library_specifications=len(entries), datasets=len(args.datasets),
                   equations=sum(len(TRUE_COEFFICIENTS[ds]) for ds in args.datasets),
                   configuration_equation_pairs=len(plan),
                   resource_skipped_pairs=sum(p["state"] != "ready" for p in plan),
                   runnable_pairs=sum(p["state"] == "ready" for p in plan),
                   unique_effective_libraries=len(ready), condition_repeats=repetitions,
                   unique_jobs=len(ready) * repetitions, fit_settings=len(settings),
                   tolerances=len(grid["coefficient_tolerances"]),
                   planned_fits=len(ready) * repetitions * len(settings),
                   planned_scores=len(ready) * repetitions * len(settings) * len(grid["coefficient_tolerances"]),
                   comparison_policy="Identical common finite rows per dataset/noise/run across libraries and targets",
                   coverage_policy="Every library must contain an accepted benchmark structure and full Allen-Cahn",
                   budget_policy="Every configured pair must fit; an oversized catalog is rejected before any output or fit",
                   working_memory_note="Fit proxy excludes Python/process overhead; factor preprocessing uses full fields")
    return plan, summary


def job_id(dataset, target, effective, noise, run):
    return digest([dataset, target, effective, noise, run])


def maximum_coefficient_error(coefficients, keys, forms, tolerance):
    values = dict(zip(keys, coefficients))
    active = {key for key, value in values.items() if abs(value) > tolerance}
    errors = [max(abs(values[key] - value) / abs(value) for key, value in form.items())
              for form in forms if active == set(form)]
    return min(errors) if errors else None


def score_sparse(record, keys, tolerance):
    coefficients = np.zeros(len(keys))
    coefficients[record["nonzero_indices"]] = record["nonzero_coefficients"]
    small = {key: value for key, value in record.items()
             if key not in ("nonzero_indices", "nonzero_coefficients", "warnings")}
    result = score_fit(dict(small, feature_keys=keys, coefficients=coefficients.tolist()), tolerance)
    forms, full_ac = benchmark_and_full_ac_forms(record["dataset"], record["target"])
    error = maximum_coefficient_error(coefficients, keys, forms, tolerance) if record["status"] == "ok" else None
    full_error = (maximum_coefficient_error(coefficients, keys, full_ac, tolerance)
                  if full_ac and record["status"] == "ok" else None)
    return dict(small, coefficient_tolerance=tolerance, structure_success=result["structure_success"],
                hamming=result["hamming"], active_count=len(result["active_terms"]),
                missing_count=len(result["missing_terms"]), extra_count=len(result["extra_terms"]),
                coefficient_error=result["coefficient_error"], max_coefficient_error=error,
                law_success=error is not None and error <= .1,
                full_ac_structure_success=(full_error is not None) if full_ac else None,
                full_ac_law_success=(full_error is not None and full_error <= .1) if full_ac else None,
                warning_count=len(record["warnings"]))


def run_job(output, dataset, target, effective, noise, run, problem, entry, settings, grid,
            rows_hash, resume=False, retry_errors=False):
    jid = job_id(dataset, target, effective, noise, run)
    directory = output / "jobs" / jid
    directory.mkdir(parents=True, exist_ok=True)
    header = dict(job_id=jid, dataset=dataset, target=target, effective_library_id=effective,
                  noise_percent=noise, run=run, representative_library_id=entry["library_id"],
                  library_size=len(problem.keys), fitted_rows=len(problem.response),
                  full_rows=problem.full_rows, valid_rows=problem.valid_rows,
                  row_indices_sha256=rows_hash, zero_columns=problem.dropped_zero_columns,
                  nonzero_columns=len(problem.keys) - len(problem.dropped_zero_columns),
                  feature_keys=problem.keys, feature_names=problem.names)
    write_json(directory / "library.json", header)
    path = directory / "fits.jsonl"
    records = read_checkpoint(path) if resume else []
    if retry_errors:
        records = [r for r in records if r["status"] == "ok"]
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in records), encoding="utf-8")
        temporary.replace(path)
    done = {r["fit_id"] for r in records}
    with path.open("a", encoding="utf-8") as handle:
        for setting in settings:
            sid = digest(setting)
            fit_id = digest([jid, sid])
            if fit_id in done:
                continue
            record = dict(fit_id=fit_id, job_id=jid, dataset=dataset, target=target,
                          effective_library_id=effective, noise_percent=noise, run=run,
                          setting_id=sid, **setting, status="error", error="",
                          nonzero_indices=[], nonzero_coefficients=[], warnings=[])
            started = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    coefficients = fit_problem(problem, setting)
                indices = np.flatnonzero(coefficients)
                record.update(status="ok", nonzero_indices=indices.tolist(),
                              nonzero_coefficients=coefficients[indices].tolist(),
                              warnings=list(dict.fromkeys(str(w.message) for w in caught)))
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["fit_seconds"] = time.perf_counter() - started
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            records.append(record)
            if len(records) % 10 == 0 or record["status"] != "ok":
                print(f"    fits {len(records)}/{len(settings)}; {record['status']}", flush=True)
    write_csv(directory / "scores.csv", [score_sparse(record, problem.keys, tolerance)
                                         for record in records for tolerance in grid["coefficient_tolerances"]])
    errors = sum(r["status"] != "ok" for r in records)
    write_json(directory / "status.json", dict(state="completed", planned_fits=len(settings),
                                               completed_fits=len(records), fit_errors=errors,
                                               fit_seconds=sum(r["fit_seconds"] for r in records)))
    return errors


def validate_data(args, catalog):
    result = []
    specs = [LibrarySpec(**entry["spec"]) for entry in catalog["libraries"]]
    for dataset, noise, run in product(args.datasets, args.noise_levels, range(args.runs)):
        obs = observations_for(dataset, noise, run, args.seed)
        bank = FactorBank(obs, args.differentiation, args.fd_order, args.smoothing_window)
        rows, full, valid = bank.common_rows(specs, list(TRUE_COEFFICIENTS[dataset]), args.crop,
                                            args.max_samples, seed_for(args.seed, "rows", dataset, run))
        result.append(dict(dataset=dataset, noise_percent=noise, run=run, full_rows=full,
                           valid_rows=valid, fitted_rows=len(rows), factor_count=len(bank.cache),
                           row_indices_sha256=hashlib.sha256(rows.tobytes()).hexdigest()))
        print(f"Data validated: {dataset}, noise={noise:g}%, run={run}; {len(rows)} shared rows", flush=True)
        del bank, obs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=HERE / "size_catalog.json")
    parser.add_argument("--library-ids", nargs="+")
    parser.add_argument("--datasets", nargs="+", default=list(TRUE_COEFFICIENTS), choices=list(TRUE_COEFFICIENTS))
    parser.add_argument("--grid", type=Path, default=HERE / "size_full_grid.json")
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--crop", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--differentiation", choices=["fd", "gradient", "smoothed"], default="fd")
    parser.add_argument("--fd-order", type=int, default=2)
    parser.add_argument("--smoothing-window", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=6000)
    parser.add_argument("--max-matrix-mb", type=float, default=256.)
    parser.add_argument("--max-working-mb", type=float, default=2048.)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true", help="Check all grammars and write plan; no fits")
    parser.add_argument("--validate-data", action="store_true", help="Also compute every primitive factor and common row masks")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--limit-jobs", type=int, help="Stop after N additional jobs, then resume without this flag")
    args = parser.parse_args()
    if (args.runs < 1 or args.crop < 0 or args.max_samples < 0 or args.threads < 1 or
            args.max_features < 1 or args.fd_order < 1 or args.smoothing_window < 5 or
            args.smoothing_window % 2 != 1 or
            any(not np.isfinite(v) or v < 0 for v in args.noise_levels) or
            any(not np.isfinite(v) or v <= 0 for v in (args.max_matrix_mb, args.max_working_mb)) or
            (args.limit_jobs is not None and args.limit_jobs < 1)):
        parser.error("Invalid run, sampling, derivative, noise or resource parameters")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.noise_levels)) != len(args.noise_levels):
        parser.error("Datasets and noise levels must be unique")
    if args.retry_errors and not args.resume:
        parser.error("--retry-errors requires --resume")
    catalog = read_catalog(args.catalog, args.library_ids)
    grid = json.loads(args.grid.read_text(encoding="utf-8"))
    settings = settings_from_grid(grid)
    for setting in settings:
        optimizer_for(setting)
    output = args.output_dir.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        parser.error("All output must stay within projects/pysindy_sweep")
    manifest_args = argparse.Namespace(**{k: v for k, v in vars(args).items()
        if k not in ("plan", "validate_data", "limit_jobs", "retry_errors", "catalog", "library_ids")})
    manifest = manifest_for(manifest_args, grid, settings)
    manifest.update(schema_version=2, experiment="library_size", catalog=catalog)
    for path in (HERE / "variable_library.py", HERE / "run_size_sweep.py"):
        manifest["input_hashes"][str(path.relative_to(HERE.parents[1]))] = file_digest(path)
    if args.resume:
        if json.loads((output / "manifest.json").read_text(encoding="utf-8")) != manifest:
            parser.error("Resume configuration, data/source hashes or versions differ")
    elif output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose another or use --resume")
    # Validate every selected grammar BEFORE creating output or fitting anything.
    plan, summary = make_plan(args, catalog, settings, grid)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", manifest)
    write_json(output / "plan.json", dict(summary=summary, libraries=plan))
    write_csv(output / "plan.csv", plan)
    print(json.dumps(summary, indent=2), flush=True)
    if args.validate_data:
        with threadpool_limits(limits=args.threads):
            write_json(output / "data_validation.json", validate_data(args, catalog))
    if args.plan:
        print("Plan saved; no regression fits performed.", flush=True)
        return 0
    specs = [LibrarySpec(**entry["spec"]) for entry in catalog["libraries"]]
    ready = {}
    for entry in plan:
        if entry["state"] == "ready":
            ready.setdefault((entry["dataset"], entry["target"], entry["effective_library_id"]), entry)
    completed, errors, processed = 0, 0, 0

    def save_state(state):
        write_json(output / "run_status.json", dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                   planned_jobs=summary["unique_jobs"], completed_jobs=completed,
                   errors=errors, additional_jobs_this_invocation=processed))

    save_state("running")
    try:
        with threadpool_limits(limits=args.threads):
            for dataset, noise, run in product(args.datasets, args.noise_levels, range(args.runs)):
                bank, row_data = None, None
                for (ds, target, effective), entry in ready.items():
                    if ds != dataset:
                        continue
                    jid = job_id(ds, target, effective, noise, run)
                    directory = output / "jobs" / jid
                    status_path = directory / "status.json"
                    if args.resume and status_path.exists():
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        if status["state"] == "completed" and not (args.retry_errors and status.get("fit_errors", 0)):
                            completed += 1
                            errors += status.get("fit_errors", 0)
                            continue
                    if args.limit_jobs is not None and processed >= args.limit_jobs:
                        save_state("partial")
                        print("Job limit reached; continue with the same command and --resume.", flush=True)
                        return 0
                    print(f"Job {completed + 1}/{summary['unique_jobs']}: {dataset}/{target}; "
                          f"library={entry['library_id']} p={entry['library_size']}; noise={noise:g}% run={run}", flush=True)
                    try:
                        if bank is None:
                            obs = observations_for(dataset, noise, run, args.seed)
                            bank = FactorBank(obs, args.differentiation, args.fd_order, args.smoothing_window)
                            row_data = bank.common_rows(specs, list(TRUE_COEFFICIENTS[dataset]), args.crop,
                                args.max_samples, seed_for(args.seed, "rows", dataset, run))
                            bank.freeze_rows(row_data[0])
                        spec = LibrarySpec(**{name: entry[name] for name in LibrarySpec.__dataclass_fields__})
                        keys, _ = enumerate_keys(list(bank.observations.fields), list(bank.axes), target, spec)
                        if digest(keys) != effective:
                            raise ValueError("Symbolic grammar differs from preflight plan")
                        rows, full, valid = row_data
                        problem = bank.build(keys, target, rows, full, valid)
                        errors += run_job(output, dataset, target, effective, noise, run, problem, entry,
                                          settings, grid, hashlib.sha256(rows.tobytes()).hexdigest(),
                                          args.resume, args.retry_errors)
                        del problem
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        errors += 1
                        directory.mkdir(parents=True, exist_ok=True)
                        write_json(directory / "status.json", dict(state="preparation_error",
                                                                  error=f"{type(exc).__name__}: {exc}"))
                        print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
                    completed += 1
                    processed += 1
                    save_state("running")
                del bank
                row_data = None
    except KeyboardInterrupt:
        save_state("interrupted")
        print("Interrupted. Finished fits were checkpointed; use --resume.", flush=True)
        return 130
    save_state("completed_with_errors" if errors else "completed")
    print(f"Finished {completed} jobs; errors={errors}. Run summarize_size_sweep.py on this output.", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
