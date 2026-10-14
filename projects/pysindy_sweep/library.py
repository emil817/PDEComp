"""One equation-independent factor grammar, following Appendix C.3/Table 6.

Only benchmark observation metadata is used. No truth or optimizer configuration
is imported here. Products are deduplicated algebraically, never by observed
column correlation. The target factor is excluded BEFORE taking products.
"""
from dataclasses import dataclass
from itertools import combinations_with_replacement
import re

import numpy as np
import pysindy as ps


Key = tuple[tuple[str, int], ...]


def multiply_keys(*keys: Key) -> Key:
    powers = {}
    for key in keys:
        for name, power in key:
            powers[name] = powers.get(name, 0) + power
    return tuple(sorted((name, power) for name, power in powers.items() if power))


def term_key(name: str) -> Key:
    """Parse benchmark monomials, including trig atoms and reciprocal factors."""
    pattern = r"(?:sin|cos)\([^)]*\)(?:\^\d+)?|\(1/[txyz]\)(?:\^\d+)?|[A-Za-z]\w*(?:\^\d+)?|1"
    matches = list(re.finditer(pattern, name))
    remainder = re.sub(pattern, "", name).strip()
    if remainder:
        raise ValueError(f"Not a monomial: {name!r}")
    keys = []
    for match in matches:
        atom, _, exponent = match.group().partition("^")
        power = int(exponent) if exponent else 1
        if atom == "1":
            continue
        if atom.startswith("(1/"):
            atom, power = atom[3:-1], -power
        keys.append(((atom, power),))
    return multiply_keys(*keys)


def term_name(key: Key) -> str:
    # Put fields before derivatives, and coordinates/trigonometry last.
    def order(item):
        atom = item[0]
        kind = 3 if "(" in atom else 2 if atom in "txyz" else 1 if "_" in atom else 0
        return kind, atom
    parts = []
    for atom, power in sorted(key, key=order):
        if power < 0:
            atom, power = f"(1/{atom})", -power
        parts.append(atom if power == 1 else f"{atom}^{power}")
    return " ".join(parts) if parts else "1"


@dataclass
class Observations:
    fields: dict[str, np.ndarray]
    axes: list[tuple[str, int, np.ndarray]]


@dataclass
class Problem:
    target: str
    matrix: np.ndarray
    response: np.ndarray
    keys: list[Key]
    names: list[str]
    full_rows: int
    valid_rows: int
    dropped_zero_columns: list[str]
    row_indices: np.ndarray


def differentiate(values, grid, axis, derivative_order, method="fd", fd_order=2,
                  smoothing_window=11):
    if method == "gradient":
        result = values
        for _ in range(derivative_order):
            result = np.gradient(result, grid, axis=axis, edge_order=2)
        return result
    kwargs = dict(d=derivative_order, order=fd_order, axis=axis)
    if method == "smoothed":
        differentiator = ps.SmoothedFiniteDifference(
            smoother_kws={"window_length": smoothing_window, "polyorder": 3, "axis": axis},
            **kwargs,
        )
    else:
        differentiator = ps.FiniteDifference(**kwargs)
    return np.asarray(differentiator._differentiate(values, grid), dtype=float)


def make_factors(observations, method="fd", fd_order=2, smoothing_window=11):
    """Factors are powers 1..3, pure spatial derivatives 1..4, time 1..2,
    and q, 1/q, sin(q), cos(q), sin(2q), cos(2q) for every available coordinate.
    """
    fields, axes = observations.fields, observations.axes
    shape = next(iter(fields.values())).shape
    factors = {}
    derivatives = {}
    for variable, values in fields.items():
        for power in range(1, 4):
            factors[((variable, power),)] = values ** power
        for coordinate, axis, grid in axes:
            for d in range(1, (2 if coordinate == "t" else 4) + 1):
                name = f"{variable}_{coordinate * d}"
                values_d = differentiate(values, grid, axis, d, method, fd_order, smoothing_window)
                factors[((name, 1),)] = values_d
                derivatives[name] = values_d
    for coordinate, axis, grid in axes:
        grid_shape = [1] * len(shape)
        grid_shape[axis] = len(grid)
        q = np.broadcast_to(grid.reshape(grid_shape), shape)
        factors[((coordinate, 1),)] = q
        factors[((coordinate, -1),)] = np.divide(
            1.0, q, out=np.full(shape, np.nan), where=np.abs(q) > 1e-12,
        )
        for frequency in (1, 2):
            argument = coordinate if frequency == 1 else f"2 {coordinate}"
            for function, operation in (("sin", np.sin), ("cos", np.cos)):
                factors[((f"{function}({argument})", 1),)] = operation(frequency * q)
    return factors, derivatives


def build_problem(factors, derivatives, target, crop=5, max_samples=10000, seed=0):
    shape = derivatives[target].shape
    if crop < 0 or any(dim <= 2 * crop for dim in shape):
        raise ValueError(f"Crop {crop} leaves no interior for shape {shape}")
    slices = tuple(slice(crop, -crop) if crop else slice(None) for _ in shape)
    full_indices = np.arange(np.prod(shape)).reshape(shape)[slices].ravel()
    # Exclusion applies to all powers/products of the target, including squares.
    available = {key: values.ravel()[full_indices] for key, values in factors.items()
                 if target not in dict(key)}
    response = derivatives[target].ravel()[full_indices]
    valid = np.isfinite(response)
    for values in available.values():
        valid &= np.isfinite(values)
    indices = np.flatnonzero(valid)
    valid_rows = len(indices)
    if not valid_rows:
        raise ValueError("No finite rows remain after crop and reciprocal masking")
    if max_samples and len(indices) > max_samples:
        indices = np.sort(np.random.default_rng(seed).choice(indices, max_samples, replace=False))
    available = {key: values[indices] for key, values in available.items()}
    specs = {(): np.ones(len(indices))}
    specs.update(available)
    for left, right in combinations_with_replacement(available, 2):
        key = multiply_keys(left, right)
        if key not in specs:
            specs[key] = available[left] * available[right]
    keys = list(specs)
    matrix = np.column_stack(list(specs.values()))
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite product columns (overflow)")
    # Keep zero columns in the exported grammar but omit them during fitting.
    # Numerically collinear nonzero columns remain in the library.
    zero = np.all(matrix == 0, axis=0)
    return Problem(target, matrix, response[indices], keys, [term_name(k) for k in keys],
                   len(full_indices), valid_rows, [term_name(k) for k, z in zip(keys, zero) if z],
                   full_indices[indices])


def expand_truth(coefficients):
    """Express accepted benchmark laws in the general monomial grammar.

    Expands the product rule and Laplacian; it does not simplify using data.
    The benchmark's accepted alternatives are also scored independently.
    """
    result = {}
    for name, coefficient in coefficients.items():
        compound = re.fullmatch(r"d_([xy])\((\w+) \2_\1\)", name)
        if compound:
            axis, variable = compound.groups()
            terms = [f"{variable}_{axis}^2", f"{variable} {variable}_{axis * 2}"]
        elif name.startswith("(") and " + " in name:
            terms = name[1:-1].split(" + ")
        else:
            terms = [name]
        for term in terms:
            key = term_key(term)
            result[key] = result.get(key, 0.0) + coefficient
    return {key: value for key, value in result.items() if value != 0}
