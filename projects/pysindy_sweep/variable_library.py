"""Configurable, truth-independent grammars for the library-size experiment.

The original paper library and its runner are left unchanged. Truth is used by
the new runner only to reject catalogs lacking a required structure and to score.
"""
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
import re

import numpy as np

from library import Problem, differentiate, multiply_keys, term_name


@dataclass(frozen=True)
class LibrarySpec:
    field_power: int = 3
    total_field_degree: int | None = 3
    space_order: int = 4
    time_order: int = 1
    max_factors: int = 2
    trig_frequency: int = 2
    coordinate_power: int = 1
    derivative_interactions: str = "squares"
    time_scope: str = "target_field"

    def __post_init__(self):
        for name in ("field_power", "max_factors", "trig_frequency", "coordinate_power"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("space_order", "time_order"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_field_degree is not None and (
                type(self.total_field_degree) is not int or self.total_field_degree < 1):
            raise ValueError("total_field_degree must be positive or null")
        if self.max_factors > 3:
            raise ValueError("This experiment supports at most three factors")
        if self.derivative_interactions not in ("squares", "all"):
            raise ValueError("derivative_interactions must be squares or all")
        if self.time_scope not in ("target_field", "all_fields"):
            raise ValueError("time_scope must be target_field or all_fields")

    def as_dict(self):
        return asdict(self)


def factor_keys(field_names, axis_names, target, spec):
    """Symbolic factors, requiring only observation metadata."""
    keys = []
    own_field = target.rsplit("_", 1)[0]
    for field in field_names:
        keys.extend(((field, p),) for p in range(1, spec.field_power + 1))
        for coordinate in axis_names:
            if coordinate == "t" and spec.time_scope == "target_field" and field != own_field:
                continue
            order = spec.time_order if coordinate == "t" else spec.space_order
            keys.extend(((f"{field}_{coordinate * d}", 1),) for d in range(1, order + 1))
    for coordinate in axis_names:
        keys.extend(((coordinate, p),) for p in range(1, spec.coordinate_power + 1))
        keys.append(((coordinate, -1),))
        for frequency in range(1, spec.trig_frequency + 1):
            argument = coordinate if frequency == 1 else f"{frequency} {coordinate}"
            keys.extend(((f"{function}({argument})", 1),) for function in ("sin", "cos"))
    return [key for key in keys if target not in dict(key)]


def allowed_key(key, field_names, spec):
    if spec.total_field_degree is not None:
        if sum(p for atom, p in key if atom in field_names) > spec.total_field_degree:
            return False
    if spec.derivative_interactions == "squares":
        # Retains u_x^2 required by the compound law, excludes u_x*u_xx.
        if sum("_" in atom for atom, _ in key) > 1:
            return False
    return True


def enumerate_keys(field_names, axis_names, target, spec, max_features=None):
    factors = factor_keys(field_names, axis_names, target, spec)
    keys = {()}
    for count in range(1, spec.max_factors + 1):
        for combination in combinations_with_replacement(factors, count):
            key = multiply_keys(*combination)
            if allowed_key(key, field_names, spec):
                keys.add(key)
                if max_features is not None and len(keys) > max_features:
                    # Report a violated budget; the runner rejects the whole catalog.
                    return None, len(keys)
    return sorted(keys), len(keys)


def contains_term(key, field_names, axis_names, target, spec):
    """Exact representability for benchmark monomials, even in oversized grammars.

    Benchmark terms do not need cancellation of opposing coordinate powers.
    Filter factors to the desired atoms, then enumerate only that small subset.
    """
    if target in dict(key) or not allowed_key(key, field_names, spec):
        return False
    if not key:
        return True
    wanted = dict(key)
    factors = [factor for factor in factor_keys(field_names, axis_names, target, spec)
               if all(atom in wanted and p * wanted[atom] > 0 and abs(p) <= abs(wanted[atom])
                      for atom, p in factor)]
    return any(multiply_keys(*combination) == key
               for count in range(1, spec.max_factors + 1)
               for combination in combinations_with_replacement(factors, count))


class FactorBank:
    """Compute each factor once on the full field, then use identical sampled rows."""
    def __init__(self, observations, method="fd", fd_order=2, smoothing_window=11):
        self.observations = observations
        self.method, self.fd_order, self.smoothing_window = method, fd_order, smoothing_window
        self.shape = next(iter(observations.fields.values())).shape
        self.axes = {name: (axis, grid) for name, axis, grid in observations.axes}
        self.cache = {}
        self.rows = None

    def value(self, key):
        if key in self.cache:
            return self.cache[key]
        ((atom, power),) = key
        fields = self.observations.fields
        if atom in fields:
            with np.errstate(over="ignore", invalid="ignore"):
                result = fields[atom] ** power
        elif atom in self.axes:
            axis, grid = self.axes[atom]
            shape = [1] * len(self.shape)
            shape[axis] = len(grid)
            q = np.broadcast_to(grid.reshape(shape), self.shape)
            if power == -1:
                result = np.divide(1., q, out=np.full(self.shape, np.nan), where=np.abs(q) > 1e-12)
            else:
                result = q ** power
        elif "_" in atom:
            field, suffix = atom.rsplit("_", 1)
            if not suffix or len(set(suffix)) != 1 or power != 1:
                raise ValueError(f"Unsupported derivative factor {key}")
            axis, grid = self.axes[suffix[0]]
            result = differentiate(fields[field], grid, axis, len(suffix), self.method,
                                   self.fd_order, self.smoothing_window)
        else:
            match = re.fullmatch(r"(sin|cos)\((?:(\d+) )?([txyz])\)", atom)
            if not match or power != 1:
                raise ValueError(f"Unsupported coordinate factor {key}")
            function, frequency, coordinate = match.groups()
            axis, grid = self.axes[coordinate]
            shape = [1] * len(self.shape)
            shape[axis] = len(grid)
            result = getattr(np, function)(int(frequency or 1) * grid.reshape(shape))
            result = np.broadcast_to(result, self.shape)
        self.cache[key] = (np.asarray(result, dtype=float).ravel()[self.rows] if self.rows is not None
                           else np.asarray(result, dtype=float))
        return self.cache[key]

    def common_rows(self, specs, targets, crop, max_samples, seed):
        if crop < 0 or any(dim <= 2 * crop for dim in self.shape):
            raise ValueError(f"Crop {crop} leaves no interior in {self.shape}")
        slices = tuple(slice(crop, -crop) if crop else slice(None) for _ in self.shape)
        full = np.arange(np.prod(self.shape)).reshape(self.shape)[slices].ravel()
        field_names, axis_names = list(self.observations.fields), list(self.axes)
        required = {((target, 1),) for target in targets}
        for spec in specs:
            for target in targets:
                required.update(factor_keys(field_names, axis_names, target, spec))
        valid = np.ones(len(full), dtype=bool)
        for key in sorted(required):
            valid &= np.isfinite(self.value(key).ravel()[full])
        rows = full[valid]
        valid_rows = len(rows)
        if not valid_rows:
            raise ValueError("No common finite rows remain")
        if max_samples and len(rows) > max_samples:
            rows = np.sort(np.random.default_rng(seed).choice(rows, max_samples, replace=False))
        return rows, len(full), valid_rows

    def freeze_rows(self, rows):
        """Keep only sampled factors after the common mask has been established."""
        if self.rows is not None:
            raise ValueError("Factor bank has already been sampled")
        self.cache = {key: value.ravel()[rows] for key, value in self.cache.items()}
        self.rows = rows

    def sample(self, key, rows):
        return self.value(key) if self.rows is not None else self.value(key).ravel()[rows]

    def build(self, keys, target, rows, full_rows, valid_rows):
        matrix = np.empty((len(rows), len(keys)), dtype=float)
        for index, key in enumerate(keys):
            column = np.ones(len(rows))
            for atom, power in key:
                # Canonical monomials may exceed the primitive power cap.
                # Assemble using primitives so reciprocal powers keep their mask.
                primitive = ((atom, -1 if power < 0 else 1),)
                with np.errstate(over="ignore", invalid="ignore"):
                    column *= self.sample(primitive, rows) ** abs(power)
            matrix[:, index] = column
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite product columns; no library-specific rows were removed")
        response = self.sample(((target, 1),), rows)
        zero = np.all(matrix == 0, axis=0)
        names = [term_name(key) for key in keys]
        return Problem(target, matrix, response, keys, names, full_rows, valid_rows,
                       [name for name, z in zip(names, zero) if z], rows)
