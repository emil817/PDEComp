"""Observation loader and truth snapshot for this standalone project.

Coordinates, trajectory truncation and accepted structures match the original
large sweep. Data arrays are read from the repository's already tracked data/.
No modules from other branches or the benchmark implementation are imported.
"""
from pathlib import Path
import numpy as np
import scipy.io as scio
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def load_data(filename):
    """Загрузка данных без привязки к структуре базового класса"""

    if filename == "ode_data.npy" or filename == "vdp_data.npy":
        data = np.load(DATA_DIR / filename)
        step = 0.05
        steps_num = 320
        t = np.arange(start=0.0, stop=step * steps_num, step=step)
        x = None
        y = None
        z = None

    elif filename == "kdv_periodic_data.npy":
        data = np.load(DATA_DIR / filename)
        shape = len(data)
        t = np.linspace(0, 1, shape)
        x = np.linspace(0, 1, shape)
        y = None
        z = None

    elif filename == "ac_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0.0, 1.0, 51)
        x = np.linspace(-1.0, 0.984375, 128)
        y = None
        z = None

    elif filename == "kdv_data.mat" or filename == "burgers_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        data = np.real(system["usol"]).T
        t = np.ravel(system["t"])
        x = np.ravel(system["x"])
        y = None
        z = None

    elif filename == "pde_divide_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 1, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "pde_compound_data.npy":
        data = np.load(DATA_DIR / filename)
        t = np.linspace(0, 0.5, 251)
        x = np.linspace(1, 2, 100)
        y = None
        z = None

    elif filename == "wave_data.csv":
        data = np.loadtxt(DATA_DIR / filename, delimiter=',').T
        t = np.linspace(0, 1, 81)
        x = np.linspace(0, 1, 81)
        y = None
        z = None

    elif filename == "lorenz_data.npy":
        system = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lorenz_time.npy")
        end = 1000
        data = [system[:end, 0], system[:end, 1], system[:end, 2]]
        t = t[:end]
        x = None
        y = None
        z = None

    elif filename == "lotka_data.npy":
        system = np.load(DATA_DIR / filename)
        t = np.load(DATA_DIR / "lotka_time.npy")
        end = 150
        t = t[:end]
        data = [system[:end, 0], system[:end, 1]]
        x = None
        y = None
        z = None

    elif filename == "ns_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        U_star = system['U_star']  # N x 2 x T
        P_star = system['p_star']  # N x T
        t_star = system['t']  # T x 1
        X_star = system['X_star']  # N x 2

        t_train = 50

        t = t_star.flatten()  # N x T
        x = np.unique(X_star[:, 0:1].flatten())  # N x T
        y = np.unique(X_star[:, 1:2].flatten()) # N x T
        z = None

        u = U_star[:, 0, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
        v = U_star[:, 1, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
        p = P_star.T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]   # N x T

        t = t[:t_train]

        data = [u, v, p]

    elif filename == "ks_data.mat":
        system = scio.loadmat(DATA_DIR / filename)
        data = system['uu'].T
        t = np.ravel(system['tt'])
        x = np.ravel(system['x'])
        y = None
        z = None

    elif filename == "burgers_sln_100_data.csv":
        df = pd.read_csv(DATA_DIR / filename, header=None)

        u = df.values
        data = np.transpose(u)
        t = np.linspace(0, 1, 101)
        x = np.linspace(-1000, 0, 101)
        y = None
        z = None

    elif filename == "ODE_simple_discovery":
        C = 1.3
        t = np.linspace(0, 4 * np.pi, 200)
        data = np.sin(t) + C * np.cos(t)
        data = [data, ]
        x = None
        y = None
        z = None

    return data, x, y, z, t


def get_data_axes(data, x, y, z, t):
    values = np.asarray(data)
    if values.ndim == 1:
        return [("t", 0, np.asarray(t, dtype=float))]
    if values.ndim == 2:
        return [("t", 0, np.asarray(t, dtype=float)), ("x", 1, np.asarray(x, dtype=float))]
    if values.ndim == 3:
        return [
            ("t", 0, np.asarray(t, dtype=float)),
            ("y", 1, np.asarray(y, dtype=float)),
            ("x", 2, np.asarray(x, dtype=float)),
        ]
    raise ValueError(f"Unsupported data dimensionality for derivatives: {values.ndim}")


TRUE_COEFFICIENTS = {
    "ode_data.npy": {
        "u_tt": {
            "u_t sin(2 t)": -1.0,
            "t": 1.5,
            "u": -4.0,
        },
    },
    "vdp_data.npy": {
        "u_tt": {
            "u_t": 0.2,
            "u^2 u_t": -0.2,
            "u": -1.0,
        },
    },
    "lotka_data.npy": {
        "x0_t": {
            "x0": 20.0,
            "x0 x1": -20.0,
        },
        "x1_t": {
            "x0 x1": 20.0,
            "x1": -20.0,
        },
    },
    "lorenz_data.npy": {
        "x0_t": {
            "x0": -10.0,
            "x1": 10.0,
        },
        "x1_t": {
            "x0": 28.0,
            "x0 x2": -1.0,
            "x1": -1.0,
        },
        "x2_t": {
            "x0 x1": 1.0,
            "x2": -8.0 / 3.0,
        },
    },
    "burgers_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": 0.1,
        },
    },
    "burgers_sln_100_data.csv": {
        "u_t": {
            "u u_x": -1.0,
        },
    },
    "ac_data.npy": {
        "u_t": {
            "u^3": -5.0,
            "u": 5.0,
        },
    },
    "kdv_data.mat": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
        },
    },
    "kdv_periodic_data.npy": {
        "u_t": {
            "u u_x": -6.0,
            "u_xxx": -1.0,
            "sin(x) cos(t)": 1.0,
        },
    },
    "wave_data.csv": {
        "u_tt": {
            "u_xx": 0.04,
        },
    },
    "pde_compound_data.npy": {
        "u_t": {
            "d_x(u u_x)": 1.0,
        },
    },
    "pde_divide_data.npy": {
        "u_t": {
            "u_xx": 0.25,
            "(1/x) u_x": -1.0,
        },
    },
    "ks_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "u_xx": -1.0,
            "u_xxxx": -1.0,
        },
    },
    "ns_data.mat": {
        "u_t": {
            "u u_x": -1.0,
            "v u_y": -1.0,
            "p_x": -1.0,
            "(u_xx + u_yy)": 0.01,
        },
        "v_t": {
            "u v_x": -1.0,
            "v v_y": -1.0,
            "p_y": -1.0,
            "(v_xx + v_yy)": 0.01,
        },
        "u_x": {
            "v_y": -1.0,
        },
    },
    "ODE_simple_discovery": {
        "u_t": {
            "cos(t)": 1.0,
            "sin(t)": -1.3,
        },
    },
}


TRUE_COEFFICIENT_ALTERNATIVES = {
    "pde_compound_data.npy": {
        "u_t": [
            {
                "u_x^2": 1.0,
                "u u_xx": 1.0,
            },
        ],
    },
    "ODE_simple_discovery": {
        "u_t": [
            {
                "u": 1.0 / 1.3,
                "sin(t)": -(1.0 / 1.3 + 1.3),
            },
        ],
    },
}
