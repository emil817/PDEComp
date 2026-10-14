"""Aggregate complete and partial size sweeps; plot recovery versus library size."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
from run_sweep import TRUE_COEFFICIENTS, digest, write_csv, write_json
from run_size_sweep import job_id, score_sparse


def read_job(output, dataset, target, effective, noise, run, combinations):
    count = len(combinations)
    result = dict(success=np.zeros(count, dtype=np.uint8), law=np.zeros(count, dtype=np.uint8),
                  full_ac=np.zeros(count, dtype=np.uint8), ok=np.zeros(count, dtype=np.uint8),
                  state="not_run", fits=0, errors=0, seconds=0., min_hamming=None,
                  header={}, nonzero_columns=None)
    if effective is None:
        result["state"] = "resource_skip"
        return result
    directory = output / "jobs" / job_id(dataset, target, effective, noise, run)
    status_path = directory / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        result.update(state=status["state"], fits=status.get("completed_fits", 0),
                      errors=status.get("fit_errors", 0), seconds=status.get("fit_seconds", 0.))
    header_path = directory / "library.json"
    if header_path.exists():
        result["header"] = json.loads(header_path.read_text(encoding="utf-8"))
        result["nonzero_columns"] = result["header"]["nonzero_columns"]
    scores_path = directory / "scores.csv"
    if not scores_path.exists():
        # Inspect partial checkpoints without truncating a line that a live runner
        # might currently be writing. A missing score file is not a measured failure.
        checkpoint = directory / "fits.jsonl"
        if checkpoint.exists() and result["header"]:
            records = []
            with checkpoint.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.endswith("\n"):
                        break
                    records.append(json.loads(line))
            keys = [tuple((atom, power) for atom, power in key)
                    for key in result["header"]["feature_keys"]]
            result.update(state="partial", fits=len(records),
                          errors=sum(r["status"] != "ok" for r in records),
                          seconds=sum(r["fit_seconds"] for r in records))
            for record in records:
                for sid, tolerance in combinations:
                    if sid != record["setting_id"]:
                        continue
                    row = score_sparse(record, keys, tolerance)
                    index = combinations[sid, tolerance]
                    result["ok"][index] = row["status"] == "ok"
                    result["success"][index] = row["structure_success"]
                    result["law"][index] = row["law_success"]
                    result["full_ac"][index] = row["full_ac_structure_success"] is True
                    if row["hamming"] is not None:
                        result["min_hamming"] = (row["hamming"] if result["min_hamming"] is None
                                                 else min(row["hamming"], result["min_hamming"]))
        return result
    seen = set()
    with scores_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index = combinations[(row["setting_id"], float(row["coefficient_tolerance"]))]
            if index in seen:
                raise ValueError(f"Duplicate score in {scores_path}")
            seen.add(index)
            result["ok"][index] = row["status"] == "ok"
            result["success"][index] = row["structure_success"] == "True"
            result["law"][index] = row["law_success"] == "True"
            result["full_ac"][index] = row["full_ac_structure_success"] == "True"
            if row["hamming"]:
                value = int(row["hamming"])
                result["min_hamming"] = (value if result["min_hamming"] is None
                                           else min(value, result["min_hamming"]))
    return result


def aggregate(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    args, settings, grid = manifest["parameters"], manifest["settings"], manifest["grid"]
    libraries = manifest["catalog"]["libraries"]
    combos_list = [(digest(setting), tolerance) for setting, tolerance in
                   product(settings, grid["coefficient_tolerances"])]
    combinations = {value: index for index, value in enumerate(combos_list)}
    count, runs = len(combinations), args["runs"]
    lookup = {(p["dataset"], p["target"], p["library_id"]): p for p in plan["libraries"]}
    target_rows, dataset_rows, totals = [], [], {}
    checked_hashes = {}
    unique_finished = set()
    for dataset, noise in product(args["datasets"], args["noise_levels"]):
        # Cache only one dataset/noise at a time, rather than millions of CSV dictionaries.
        cache = {}
        for entry in libraries:
            lib_id = entry["id"]
            common = dict(dataset=dataset, noise_percent=noise, library_id=lib_id,
                          aliases=entry.get("aliases", []), groups=entry.get("groups", []), **entry["spec"])
            component_vectors = []
            component_ok = []
            component_finished = []
            resource_skip = False
            for target in TRUE_COEFFICIENTS[dataset]:
                p = lookup[dataset, target, lib_id]
                effective = p["effective_library_id"] if p["state"] == "ready" else None
                resource_skip |= effective is None
                jobs = []
                for run in range(runs):
                    key = (target, effective, run)
                    if key not in cache:
                        cache[key] = read_job(output, dataset, target, effective, noise, run, combinations)
                        header = cache[key]["header"]
                        if header:
                            row_key = (dataset, noise, run)
                            row_hash = header["row_indices_sha256"]
                            if row_key in checked_hashes and checked_hashes[row_key] != row_hash:
                                raise ValueError(f"Libraries used different rows for {row_key}")
                            checked_hashes[row_key] = row_hash
                        if cache[key]["state"] == "completed":
                            unique_finished.add((dataset, target, effective, noise, run))
                    jobs.append(cache[key])
                successes = np.stack([j["success"] for j in jobs])
                ok = np.stack([j["ok"] for j in jobs])
                finished = sum(j["state"] == "completed" for j in jobs)
                success_counts = successes.sum(axis=0)
                law_counts = np.stack([j["law"] for j in jobs]).sum(axis=0)
                full_counts = np.stack([j["full_ac"] for j in jobs]).sum(axis=0)
                hd = [j["min_hamming"] for j in jobs if j["min_hamming"] is not None]
                state = ("resource_skip" if effective is None else "complete" if finished == runs
                         else "partial" if any(j["header"] for j in jobs) else "not_run")
                target_rows.append(dict(common, target=target, effective_library_id=effective,
                    library_size=p["library_size"], size_is_lower_bound=p["size_is_lower_bound"],
                    state=state, skip_reason=p["skip_reason"], truth_covered=any(p["truth_coverage"]),
                    planned_runs=runs, completed_runs=finished,
                    completed_fits=sum(j["fits"] for j in jobs), fit_errors=sum(j["errors"] for j in jobs),
                    completed_scores=int(ok.sum()), planned_scores=runs * count,
                    successful_scores=int(successes.sum()),
                    successful_settings_fraction=float(successes.sum() / ok.sum()) if ok.sum() else None,
                    oracle_fixed_setting_success_rate=float(success_counts.max() / runs) if ok.sum() else None,
                    oracle_fixed_setting_law_rate=float(law_counts.max() / runs) if ok.sum() else None,
                    full_ac_oracle_rate=float(full_counts.max() / runs) if dataset == "ac_data.npy" and ok.sum() else None,
                    best_setting_id=combos_list[int(success_counts.argmax())][0] if ok.sum() else None,
                    best_tolerance=combos_list[int(success_counts.argmax())][1] if ok.sum() else None,
                    minimum_hamming=min(hd) if hd else None, fit_seconds=sum(j["seconds"] for j in jobs),
                    fitted_rows=next((j["header"]["fitted_rows"] for j in jobs if j["header"]), None),
                    nonzero_columns_mean=float(np.mean([j["nonzero_columns"] for j in jobs
                                                        if j["nonzero_columns"] is not None]))
                    if any(j["nonzero_columns"] is not None for j in jobs) else None))
                component_vectors.append(successes)
                component_ok.append(ok)
                component_finished.append(finished)
            whole = np.logical_and.reduce(component_vectors).sum(axis=0)
            complete_scores = np.logical_and.reduce(component_ok)
            dataset_rows.append(dict(common, state="resource_skip" if resource_skip else
                "complete" if all(n == runs for n in component_finished) else "partial",
                planned_runs=runs, completed_scores=int(complete_scores.sum()),
                oracle_fixed_setting_success_rate=float(whole.max() / runs) if complete_scores.any() else None,
                successful_scores=int(whole.sum()),
                successful_settings_fraction=float(whole.sum() / complete_scores.sum()) if complete_scores.any() else None))
            total = totals.setdefault((lib_id, noise), dict(equations=np.zeros(count, dtype=int),
                datasets=np.zeros(count, dtype=int), completed_equation_runs=0, resource_skipped_equations=0))
            total["equations"] += np.sum(component_vectors, axis=(0, 1)).astype(int)
            total["datasets"] += whole.astype(int)
            total["completed_equation_runs"] += sum(component_finished)
            total["resource_skipped_equations"] += sum(not lookup[dataset, t, lib_id]["state"] == "ready"
                                                       for t in TRUE_COEFFICIENTS[dataset])
        del cache
    common_rows = []
    equations = sum(len(TRUE_COEFFICIENTS[d]) for d in args["datasets"])
    for entry, noise in product(libraries, args["noise_levels"]):
        total = totals[entry["id"], noise]
        index = int(total["equations"].argmax())
        common_rows.append(dict(library_id=entry["id"], noise_percent=noise, groups=entry.get("groups", []),
            **entry["spec"], planned_equation_runs=equations * runs,
            completed_equation_runs=total["completed_equation_runs"],
            resource_skipped_equations=total["resource_skipped_equations"],
            complete=total["completed_equation_runs"] == equations * runs,
            best_common_correct_equations_per_run=float(total["equations"].max() / runs),
            best_common_correct_datasets_per_run=float(total["datasets"].max() / runs),
            best_setting_id=combos_list[index][0], best_tolerance=combos_list[index][1]))
    write_csv(output / "recovery_by_target.csv", target_rows)
    write_csv(output / "recovery_by_dataset.csv", dataset_rows)
    write_csv(output / "recovery_by_library.csv", common_rows)
    summary = dict(plan["summary"], completed_unique_jobs=len(unique_finished),
                   shared_row_conditions_checked=len(checked_hashes),
                   all_jobs_finished=len(unique_finished) == plan["summary"]["unique_jobs"],
                   interpretation="Missing/resource-skipped fits are not measured failures; oracle choices use ground truth")
    write_json(output / "size_summary.json", summary)
    lines = ["# Эксперимент зависимости восстановления от библиотеки", "",
             f"Библиотек в каталоге: **{summary['library_specifications']}**; "
             f"завершённых уникальных заданий: **{len(unique_finished)}/{summary['unique_jobs']}**.", "",
             "Каждая библиотека покрывает эталон. Ресурсные пропуски и незавершённые задания отмечены отдельно.",
             "Результаты выбора лучшей настройки используют ground truth; это диагностический максимум.", "",
             "- `recovery_by_target.csv`: размер, состав, доля успешных настроек, максимум по повторам, ε∞≤0.1 и полный Allen–Cahn.",
             "- `recovery_by_dataset.csv`: восстановление всех компонент при одной настройке.",
             "- `recovery_by_library.csv`: максимум при одной общей настройке по всем задачам.",
             "- `plan.csv`: все конфигурации, эквивалентные библиотеки, покрытие и ограничения ресурсов.", "",
             "Размер сам по себе не определяет сложность: библиотеки одинакового размера могут иметь разный состав.",
             "Число успешных настроек характеризует заданную сетку, а не вероятность успеха на новых траекториях."]
    (output / "SIZE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target_rows, common_rows, summary


def plot(output, target_rows, common_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter
    for noise in sorted({row["noise_percent"] for row in target_rows}):
        equations = list(dict.fromkeys((r["dataset"], r["target"]) for r in target_rows))
        figure, axes = plt.subplots(int(np.ceil(len(equations) / 4)), 4, figsize=(18, 3.5 * int(np.ceil(len(equations) / 4))), squeeze=False)
        for axis, (dataset, target) in zip(axes.flat, equations):
            selected = [r for r in target_rows if r["dataset"] == dataset and r["target"] == target
                        and r["noise_percent"] == noise and r["successful_settings_fraction"] is not None]
            # Effective aliases must not overweight the scatter.
            selected = list({r["effective_library_id"]: r for r in selected}.values())
            axis.scatter([r["library_size"] for r in selected],
                         [100 * r["successful_settings_fraction"] for r in selected], alpha=.55, s=20)
            axis.set_xscale("log")
            axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_ylim(-2, 102)
            axis.set_title(f"{dataset}: {target}", fontsize=9)
            axis.set_xlabel("Library columns")
            axis.set_ylabel("Successful scored settings (%)")
            axis.grid(alpha=.2)
        for axis in list(axes.flat)[len(equations):]:
            axis.set_visible(False)
        figure.suptitle(f"Recovery vs library size; noise={noise:g}% (completed fits; grid dependent)")
        figure.tight_layout()
        figure.savefig(output / f"size_recovery_noise_{noise:g}.png", dpi=140)
        plt.close(figure)
        families = ("field_degree", "space_order", "time_order", "max_factors", "trig_frequency", "coordinate_power")
        figure, axes = plt.subplots(2, 3, figsize=(15, 8))
        for axis, family in zip(axes.flat, families):
            path = [(g["value"], r) for r in common_rows if r["noise_percent"] == noise
                    and r["completed_equation_runs"] > 0
                    for g in r["groups"] if g["family"] == family]
            path.sort(key=lambda item: item[0])
            axis.plot([v for v, _ in path], [r["best_common_correct_equations_per_run"] for _, r in path], "o-")
            axis.set_title(family)
            axis.set_xticks([v for v, _ in path])
            axis.set_ylim(0, len(equations))
            axis.set_ylabel("Best common correct equations / run")
            axis.set_xlabel("Expansion control")
            axis.grid(alpha=.2)
        figure.suptitle(f"Controlled paths; noise={noise:g}%; maxima use truth, unfinished fits are lower bounds")
        figure.tight_layout()
        figure.savefig(output / f"controlled_paths_noise_{noise:g}.png", dpi=140)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(HERE) or output == HERE:
        parser.error("Output must stay within projects/pysindy_sweep")
    targets, common, summary = aggregate(output)
    if not args.no_plots:
        plot(output, targets, common)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
