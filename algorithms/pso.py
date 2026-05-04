import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 1.5,
        c2: float = 1.5,
        v_max_ratio: float = 0.2,
        seed: int = 0,
        **kwargs) -> float:
    """
    Particle Swarm Optimization (PSO) for n-dimensional minimization.

    Uses linearly decreasing inertia weight from w_max to w_min.

    Parameters
    ----------
    func        : callable, receives np.ndarray shape (dim,) and returns float
    dim         : number of dimensions
    bounds      : (lo, hi) — search domain
    n_iter      : number of iterations
    init        : np.ndarray shape (swarm_size, dim) — initial particle positions
    w_max       : initial inertia weight
    w_min       : final inertia weight
    c1          : cognitive coefficient (personal best)
    c2          : social coefficient (global best)
    v_max_ratio : max velocity as a fraction of the search range
    seed        : seed for internal operators (does not affect init)

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    vmax = v_max_ratio * (hi - lo)
    swarm_size = len(init)

    pos = init.copy().astype(float)
    vel = rng.uniform(-vmax, vmax, size=(swarm_size, dim))

    pbest_pos = pos.copy()
    pbest_val = np.array([func(p) for p in pbest_pos])

    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = float(pbest_val[gbest_idx])

    for t in range(1, n_iter + 1):
        # Linearly decreasing inertia
        w = w_max - (w_max - w_min) * (t - 1) / max(n_iter - 1, 1)

        r1 = rng.random(size=(swarm_size, dim))
        r2 = rng.random(size=(swarm_size, dim))

        vel = (w * vel
               + c1 * r1 * (pbest_pos - pos)
               + c2 * r2 * (gbest_pos - pos))
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lo, hi)

        fitness = np.array([func(p) for p in pos])

        # Update personal bests (minimization)
        improved = fitness < pbest_val
        pbest_pos[improved] = pos[improved]
        pbest_val[improved] = fitness[improved]

        # Update global best
        idx = np.argmin(pbest_val)
        if pbest_val[idx] < gbest_val:
            gbest_val = float(pbest_val[idx])
            gbest_pos = pbest_pos[idx].copy()

    return gbest_val
