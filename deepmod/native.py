"""Near-native DeePyMoD pipeline used by the framework-level protocol."""

from pathlib import Path

import numpy as np

from data.config import NATIVE_DEEPMOD_DEFAULTS, sindy_params
from utils.protocols import ProtocolUnavailableError


def library_1d_names(poly_order, diff_order):
    names = []
    for power in range(poly_order + 1):
        for derivative_order in range(diff_order + 1):
            factors = []
            if power == 1:
                factors.append("u")
            elif power > 1:
                factors.append(f"u^{power}")
            if derivative_order == 1:
                factors.append("u_x")
            elif derivative_order > 1:
                factors.append(f"u_{'x' * derivative_order}")
            names.append(" ".join(factors) or "1")
    return names


def scalar_pde_coordinates(values, x, t):
    t_grid, x_grid = np.meshgrid(
        np.asarray(t, dtype=float),
        np.asarray(x, dtype=float),
        indexing="ij",
    )
    coordinates = np.column_stack((t_grid.ravel(), x_grid.ravel()))
    targets = np.asarray(values, dtype=float).reshape(-1, 1)
    return coordinates, targets


def subsample(coordinates, targets, max_samples, seed):
    if max_samples is None or len(coordinates) <= max_samples:
        return coordinates, targets
    rng = np.random.default_rng(seed)
    indexes = np.sort(rng.choice(len(coordinates), size=int(max_samples), replace=False))
    return coordinates[indexes], targets[indexes]


def run_native_deepmod(data, x, y, z, t, filename, options=None):
    """Train DeePyMoD's NN/autograd/constraint pipeline on a scalar 1D PDE."""

    import torch
    from deepymod.data import Dataset, get_train_test_loader
    from deepymod.model.constraint import LeastSquares
    from deepymod.model.deepmod import DeepMoD
    from deepymod.model.func_approx import NN
    from deepymod.model.library import Library1D
    from deepymod.model.sparse_estimators import Threshold
    from deepymod.training import train
    from deepymod.training.sparsity_scheduler import TrainTest

    if isinstance(data, list):
        if len(data) != 1:
            raise ProtocolUnavailableError(
                "DeePyMoD's stock Library1D does not provide the benchmark's coupled-system library"
            )
        data = data[0]
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or x is None:
        raise ProtocolUnavailableError(
            "DeePyMoD native mode currently uses its stock Library1D and therefore requires scalar 1D PDE data"
        )

    target = sindy_params[filename].get("targets", [{"name": "u_t", "axis": "t", "order": 1}])[0]
    if target.get("axis", "t") != "t" or target.get("order", 1) != 1:
        raise ProtocolUnavailableError(
            "DeePyMoD's stock Library1D fixes the left-hand side to the first time derivative"
        )

    config = {**NATIVE_DEEPMOD_DEFAULTS, **(options or {})}
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    coordinates, targets = scalar_pde_coordinates(values, x, t)
    coordinates, targets = subsample(
        coordinates,
        targets,
        config.get("max_samples"),
        seed,
    )
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)

    dataset = Dataset(
        lambda: (coordinate_tensor, target_tensor),
        shuffle=True,
        device=device,
    )
    train_loader, test_loader = get_train_test_loader(
        dataset,
        train_test_split=float(config["train_fraction"]),
    )

    poly_order = int(config["poly_order"])
    diff_order = int(config["diff_order"])
    model = DeepMoD(
        NN(2, list(config["hidden_layers"]), 1),
        Library1D(poly_order=poly_order, diff_order=diff_order),
        Threshold(float(config["threshold"])),
        LeastSquares(),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.99, 0.999),
        amsgrad=True,
    )
    scheduler = TrainTest(
        patience=int(config.get("scheduler_patience", 200)),
        delta=float(config.get("scheduler_delta", 1e-5)),
    )
    log_dir = Path("results/deepmod/native/logs") / filename.replace(".", "_")
    log_dir.mkdir(parents=True, exist_ok=True)
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        log_dir=str(log_dir),
        max_iterations=int(config["max_iterations"]),
        write_iterations=int(config["write_iterations"]),
        delta=float(config.get("convergence_delta", 1e-5)),
        patience=int(config.get("convergence_patience", 200)),
    )

    train_coordinates, _ = train_loader[0]
    _, time_derivatives, thetas = model(train_coordinates)
    model.constraint.sparsity_masks = model.sparse_estimator(thetas, time_derivatives)
    model(train_coordinates)
    coefficients = (
        model.constraint_coeffs(scaled=False, sparse=True)[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    coefficients[np.abs(coefficients) < float(config["coefficient_tol"])] = 0.0
    feature_names = library_1d_names(poly_order, diff_order)
    return {
        "dataset": filename.split(".")[0],
        "targets": [target.get("name", "u_t")],
        "coefficients": [coefficients.tolist()],
        "features": [feature_names],
        "library_sizes": {target.get("name", "u_t"): len(feature_names)},
        "library_size": len(feature_names),
        "protocol": "native",
    }
