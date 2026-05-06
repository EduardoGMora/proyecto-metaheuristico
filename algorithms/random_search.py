import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        **kwargs) -> float:
    """
    n-dimensional Random Search for minimization.

    Parameters
    ----------
    func   : callable, receives np.ndarray shape (dim,) and returns float
    dim    : number of dimensions
    bounds : (lo, hi) — search domain
    n_iter : total number of evaluations (init points count toward this budget)
    init   : np.ndarray shape (pop_size, dim) — pre-generated initial points

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(kwargs.get("seed", 0))
    lo, hi = bounds

    # Evaluate init points
    values = np.array([func(p) for p in init])
    best_val = float(np.min(values))

    # Additional random evaluations up to n_iter budget
    restantes = n_iter - len(init)
    for _ in range(max(restantes, 0)):
        p = rng.uniform(lo, hi, size=dim)
        v = func(p)
        if v < best_val:
            best_val = v

    return best_val
