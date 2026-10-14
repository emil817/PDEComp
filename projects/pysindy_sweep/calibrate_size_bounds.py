"""Choose a complete Cartesian library grid fitting the existing resource limits.

Maximize the number of distinct parameter combinations, retaining at least two
levels of each expansion control. Budget decisions use metadata, never fit scores.
"""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_sweep import TRUE_COEFFICIENTS, observations_for, write_json
from run_size_sweep import metadata_rows
from variable_library import LibrarySpec, enumerate_keys


def calibrate():
    budgets = dict(max_features=6000, max_matrix_mb=256., max_working_mb=2048.)
    # A small buffer below the existing plan limits, without changing sample size.
    margin = .05
    metadata = []
    for dataset in TRUE_COEFFICIENTS:
        obs = observations_for(dataset, 0., 0, 0)
        rows = len(metadata_rows(obs, 5, 10000, 0))
        metadata.append(dict(dataset=dataset, fields=list(obs.fields),
                             axes=[name for name, _, _ in obs.axes], rows=rows,
                             targets=list(TRUE_COEFFICIENTS[dataset])))
    heavy = max(metadata, key=lambda item: (len(item["fields"]), len(item["axes"]), item["rows"]))
    candidates = []
    # All five numeric expansion controls must retain actual variation.
    for degree, space, time, trig, coordinate in product(range(4, 9), range(5, 9),
                                                       range(2, 5), range(3, 5), range(2, 4)):
        spec = LibrarySpec(field_power=degree, total_field_degree=degree,
                           space_order=space, time_order=time, trig_frequency=trig,
                           coordinate_power=coordinate, max_factors=2,
                           derivative_interactions="all", time_scope="all_fields")
        _, count = enumerate_keys(heavy["fields"], heavy["axes"], heavy["targets"][0], spec,
                                  budgets["max_features"])
        matrix = count * heavy["rows"] * 8 / 2**20
        working = 3 * matrix + 16 * count**2 / 2**20
        if (count <= budgets["max_features"] * (1 - margin)
                and matrix <= budgets["max_matrix_mb"] * (1 - margin)
                and working <= budgets["max_working_mb"] * (1 - margin)):
            volume = (degree - 2) * (space - 3) * time * (trig - 1) * coordinate
            candidates.append(dict(degree=degree, space=space, time=time, trig=trig,
                                   coordinate=coordinate, numeric_grid_volume=volume,
                                   features=count, matrix_mb=matrix, working_mb=working))
    if not candidates:
        raise ValueError("No complete multi-level grid fits the current budgets")
    # Prefer the largest complete grid; ties favor derivative-order variation,
    # then field degrees, then lower memory usage. This rule is fixed before fits.
    candidates.sort(key=lambda c: (-c["numeric_grid_volume"], -c["space"], -c["time"],
                                   -c["degree"], c["features"], c["trig"], c["coordinate"]))
    winner = candidates[0]
    spec = LibrarySpec(field_power=winner["degree"], total_field_degree=winner["degree"],
                       space_order=winner["space"], time_order=winner["time"],
                       trig_frequency=winner["trig"], coordinate_power=winner["coordinate"],
                       derivative_interactions="all", time_scope="all_fields")
    checked = []
    for item in metadata:
        for target in item["targets"]:
            _, count = enumerate_keys(item["fields"], item["axes"], target, spec)
            matrix = count * item["rows"] * 8 / 2**20
            working = 3 * matrix + 16 * count**2 / 2**20
            if count > budgets["max_features"] or matrix > budgets["max_matrix_mb"] or working > budgets["max_working_mb"]:
                raise ValueError(f"Envelope fails for {item['dataset']}/{target}")
            checked.append(dict(dataset=item["dataset"], target=target, features=count,
                                matrix_mb=matrix, working_mb=working))
    bounds = dict(field_degree=list(range(3, winner["degree"] + 1)),
                  space_order=list(range(4, winner["space"] + 1)),
                  time_order=list(range(1, winner["time"] + 1)),
                  trig_frequency=list(range(2, winner["trig"] + 1)),
                  coordinate_power=list(range(1, winner["coordinate"] + 1)),
                  max_factors=[2], derivative_interactions=["squares", "all"],
                  time_scope=["target_field", "all_fields"])
    return dict(schema_version=1, bounds=bounds, budgets=budgets, budget_margin=margin,
                worst_case_dataset=heavy["dataset"], winner=winner, feasible_envelopes=len(candidates),
                selection_rule="Maximum complete grid volume; ties: space order, time order, field degree, lower memory",
                checked_envelope=checked, top_envelopes=candidates[:10],
                reason_for_two_factors="Three factors already exceed the budget for the complete baseline on Navier-Stokes")


def main():
    result = calibrate()
    path = HERE / "size_bounds.json"
    write_json(path, result)
    print(json.dumps({k: result[k] for k in ("bounds", "winner", "feasible_envelopes")}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
