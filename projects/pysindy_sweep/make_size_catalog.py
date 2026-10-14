"""Generate a fixed catalog: complete baseline, controlled paths, factorial grid."""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from variable_library import LibrarySpec

HERE = Path(__file__).resolve().parent


def build_catalog(bounds=None):
    calibration = json.loads((HERE / "size_bounds.json").read_text(encoding="utf-8"))
    bounds = bounds or calibration["bounds"]
    entries, lookup = [], {}
    base = LibrarySpec()

    def add(name, spec, family, value):
        signature = json.dumps(spec.as_dict(), sort_keys=True)
        if signature in lookup:
            entry = lookup[signature]
            if name != entry["id"] and name not in entry["aliases"]:
                entry["aliases"].append(name)
        else:
            entry = dict(id=name, aliases=[], groups=[], spec=spec.as_dict())
            entries.append(entry)
            lookup[signature] = entry
        group = dict(family=family, value=value)
        if group not in entry["groups"]:
            entry["groups"].append(group)

    add("baseline", base, "reference", "complete_baseline")
    add("paper_reference", LibrarySpec(total_field_degree=None, time_order=2,
        derivative_interactions="all", time_scope="all_fields"), "reference", "paper")
    paths = bounds
    for family, values in paths.items():
        for value in values:
            changes = (dict(field_power=value, total_field_degree=value) if family == "field_degree"
                       else {family: value})
            add(f"{family}_{value}", replace(base, **changes), family, value)
    # A complete Cartesian grid, including both time scopes and coordinate powers.
    for degree, space, time, factors, trig, coordinate, interactions, scope in product(
            bounds["field_degree"], bounds["space_order"], bounds["time_order"],
            bounds["max_factors"], bounds["trig_frequency"], bounds["coordinate_power"],
            bounds["derivative_interactions"], bounds["time_scope"]):
        spec = replace(base, field_power=degree, total_field_degree=degree,
                       space_order=space, time_order=time, max_factors=factors,
                       trig_frequency=trig, coordinate_power=coordinate,
                       derivative_interactions=interactions, time_scope=scope)
        name = f"grid_d{degree}_x{space}_t{time}_f{factors}_w{trig}_q{coordinate}_{interactions}_{scope}"
        add(name, spec, "factorial", name)
    growing = base
    add("growth_0", growing, "nested_growth", 0)
    for step, changes in enumerate((
            dict(field_power=max(bounds["field_degree"]), total_field_degree=max(bounds["field_degree"])),
            dict(space_order=max(bounds["space_order"])),
            dict(time_order=max(bounds["time_order"]), time_scope="all_fields"),
            dict(trig_frequency=max(bounds["trig_frequency"]), coordinate_power=max(bounds["coordinate_power"])),
            dict(derivative_interactions="all")), 1):
        growing = replace(growing, **changes)
        add(f"growth_{step}", growing, "nested_growth", step)
    add("maximal_library", growing, "reference", "maximal")
    return dict(schema_version=1, description="Complete resource-calibrated grid; every grammar must cover all selected laws; no skipped pairs.",
                bounds=bounds, calibration={k: calibration[k] for k in ("budgets", "budget_margin", "winner", "selection_rule")},
                libraries=entries)


def main():
    path = HERE / "size_catalog.json"
    catalog = build_catalog()
    path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"{path}: {len(catalog['libraries'])} library specifications")


if __name__ == "__main__":
    main()
