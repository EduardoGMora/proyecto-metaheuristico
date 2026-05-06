import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        step_size: float = 0.5,
        step_decay: float = 0.5,
        min_step: float = 1e-3,
        **kwargs) -> float:
    """
    n-dimensional Local Search (hill climbing with step decay) for minimization.
    Starts from the best point in `init`.

    Parameters
    ----------
    func       : callable, receives np.ndarray shape (dim,) and returns float
    dim        : number of dimensions
    bounds     : (lo, hi) — search domain
    n_iter     : not used directly (stopping criterion is min_step);
                 included to keep a uniform interface across algorithms
    init       : np.ndarray shape (pop_size, dim) — initial points;
                 the one with the lowest value is used as the starting point
    step_size  : initial step size
    step_decay : factor by which the step is reduced when no improvement is found
    min_step   : minimum step size (stopping criterion)

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    lo, hi = bounds

    # Start from the best point in init
    values = np.array([func(p) for p in init])
    best_idx = int(np.argmin(values))
    pos = init[best_idx].copy().astype(float)
    val = values[best_idx]

    # Unit directions: +1 and -1 along each axis
    dirs = []
    for d in range(dim):
        for sign in (-1, 1):
            delta = np.zeros(dim)
            delta[d] = sign
            dirs.append(delta)

    while step_size >= min_step:
        mejor_vecino = None
        mejor_val = val

        for delta in dirs:
            vecino = np.clip(pos + delta * step_size, lo, hi)
            v = func(vecino)
            if v < mejor_val:
                mejor_val = v
                mejor_vecino = vecino

        if mejor_vecino is not None:
            pos = mejor_vecino
            val = mejor_val
        else:
            step_size *= step_decay

    return float(val)
