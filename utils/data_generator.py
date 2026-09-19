import numpy as np


def generate_random(n: int, min_val: float, max_val: float) -> np.ndarray:
    """Génère n valeurs aléatoires uniformes dans [min_val, max_val]."""
    return np.random.uniform(min_val, max_val, n)


def generate_dataset_regression(n=200,
                                 x1_min=-10, x1_max=10,
                                 x2_min=-5,  x2_max=15,
                                 x3_min=0,   x3_max=20,
                                 y_min=-30,  y_max=30,
                                 noise=True):
    """
    Crée un dataset (X1, X2, X3 → Y) avec bruit aléatoire.
    Y = a0 + a1*X1 + a2*X2 + a3*X3 + bruit
    """
    X1 = generate_random(n, x1_min, x1_max)
    X2 = generate_random(n, x2_min, x2_max)
    X3 = generate_random(n, x3_min, x3_max)

    # Coefficients cachés (simulation réelle)
    a0, a1, a2, a3 = 2.5, 0.8, -1.2, 0.5
    Y = a0 + a1 * X1 + a2 * X2 + a3 * X3

    if noise:
        span = (y_max - y_min) * 0.05
        Y += generate_random(n, -span, span)

    # Clamp Y dans [y_min, y_max]
    Y = np.clip(Y, y_min, y_max)
    return X1, X2, X3, Y


def generate_dataset_clustering(n=200,
                                  x1_min=-10, x1_max=10,
                                  x2_min=-5,  x2_max=15,
                                  x3_min=0,   x3_max=20):
    """Dataset 3D pour clustering."""
    X1 = generate_random(n, x1_min, x1_max)
    X2 = generate_random(n, x2_min, x2_max)
    X3 = generate_random(n, x3_min, x3_max)
    return X1, X2, X3


def generate_timeseries(n=200, y_min=0, y_max=15, trend=True, seasonal=True):
    """Série temporelle avec tendance et saisonnalité."""
    t = np.arange(n)
    base = generate_random(n, y_min, y_max)
    signal = base.copy()
    if trend:
        signal += t * (y_max - y_min) / (n * 3)
    if seasonal:
        signal += np.sin(t * 2 * np.pi / 30) * (y_max - y_min) * 0.15
    return t, np.clip(signal, y_min, y_max)


def compute_regression_metrics(y_true, y_pred):
    """Calcule R², MSE, RMSE."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    mse  = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return {"R²": round(r2, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4)}