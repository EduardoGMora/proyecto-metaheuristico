import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        beta0: float = 1.0,
        gamma: float = 1.0,
        alpha: float = 0.2,
        alpha_decay: float = 0.98,
        seed: int = 0,
        **kwargs) -> float:
    """
    Firefly Algorithm (FA) for n-dimensional minimization.

    Fireflies are attracted to brighter (lower-value) neighbors.
    Brightness is inversely proportional to the objective value.

    Parameters
    ----------
    func        : callable, receives np.ndarray shape (dim,) and returns float
    dim         : number of dimensions
    bounds      : (lo, hi) — search domain
    n_iter      : number of iterations
    init        : np.ndarray shape (num_fireflies, dim) — initial population
    beta0       : maximum attractiveness (at distance 0)
    gamma       : light absorption coefficient
    alpha       : randomness scaling factor
    alpha_decay : multiplicative decay applied to alpha each iteration
    seed        : seed for internal operators (does not affect init)

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    scale_noise = hi - lo
    num_fireflies = len(init)

    positions = init.copy().astype(float)
    values = np.array([func(p) for p in positions])

    best_val = float(np.min(values))
    alpha_curr = alpha

    for _ in range(n_iter):
        # Sort ascending: index 0 = best (lowest value)
        order = np.argsort(values)

        for idx_i in range(num_fireflies):
            i = order[idx_i]
            xi = positions[i].copy()

            # Move toward every firefly that is brighter (lower value)
            for idx_j in range(0, idx_i):
                j = order[idx_j]
                r = np.linalg.norm(xi - positions[j])
                beta = beta0 * np.exp(-gamma * r ** 2)
                noise = rng.normal(0.0, 1.0, size=dim)
                xi = xi + beta * (positions[j] - xi) + alpha_curr * noise * (scale_noise / 2.0)
                xi = np.clip(xi, lo, hi)

            positions[i] = xi
            values[i] = func(xi)

        alpha_curr *= alpha_decay

        iter_best = float(np.min(values))
        if iter_best < best_val:
            best_val = iter_best

    return best_val
