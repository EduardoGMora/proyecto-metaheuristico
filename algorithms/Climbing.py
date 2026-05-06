import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        step_size: float = 0.1,
        seed: int = 0,
        **kwargs) -> float:
    """
    Stochastic Hill Climbing Algorithm for n-dimensional minimization.

    Evaluates neighbors around current solutions by adding random noise.
    If a neighbor has a better (lower) objective value, the current 
    solution moves to that neighbor.

    Parameters
    ----------
    func      : callable, receives np.ndarray shape (dim,) and returns float
    dim       : number of dimensions
    bounds    : (lo, hi) — search domain
    n_iter    : number of iterations (steps to climb)
    init      : np.ndarray shape (num_agents, dim) — initial starting points
    step_size : standard deviation of the Gaussian noise for neighbor generation
    seed      : seed for internal random generator

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    num_agents = len(init)

    positions = init.copy().astype(float)
    values = np.array([func(p) for p in positions])

    best_val = float(np.min(values))

    for _ in range(n_iter):
        for i in range(num_agents):
            curr_pos = positions[i]
            curr_val = values[i]

            noise = rng.normal(loc=0.0, scale=step_size, size=dim)
            neighbor_pos = curr_pos + noise
            
            neighbor_pos = np.clip(neighbor_pos, lo, hi)

            neighbor_val = func(neighbor_pos)

            if neighbor_val < curr_val:
                positions[i] = neighbor_pos
                values[i] = neighbor_val

                if neighbor_val < best_val:
                    best_val = float(neighbor_val)

    return best_val