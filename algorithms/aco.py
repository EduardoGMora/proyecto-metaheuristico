import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        q: float = 0.5,
        xi: float = 0.85,
        num_ants: int = 10,
        seed: int = 0,
        **kwargs) -> float:
    """
    Ant Colony Optimization for continuous domains (ACO-R)
    for n-dimensional minimization.

    Parameters
    ----------
    func     : callable, receives np.ndarray shape (dim,) and returns float
    dim      : number of dimensions
    bounds   : (lo, hi) — search domain
    n_iter   : number of iterations
    init     : np.ndarray shape (archive_size, dim) — initial archive population
    q        : locality of the search (smaller → more focused on best solutions)
    xi       : pheromone evaporation / deviation scaling factor
    num_ants : number of ants generated per iteration
    seed     : seed for internal operators (does not affect init)

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    archive_size = len(init)

    # Build archive from init, sorted ascending (best = lowest value first)
    archive = init.copy().astype(float)
    archive_vals = np.array([func(s) for s in archive])
    order = np.argsort(archive_vals)
    archive = archive[order]
    archive_vals = archive_vals[order]

    best_val = float(archive_vals[0])

    def compute_weights(m, q):
        k = np.arange(1, m + 1)
        w = np.exp(-((k - 1) ** 2) / (2 * (q * m) ** 2))
        w /= w.sum()
        return w

    def compute_sigmas(arch):
        m = arch.shape[0]
        sigmas = np.empty_like(arch)
        for k in range(m):
            diffs = np.abs(arch[k] - arch)          # shape (m, dim)
            sigmas[k] = xi * diffs.sum(axis=0) / max(m - 1, 1)
        sigmas = np.where(sigmas < 1e-6, 1e-6, sigmas)
        return sigmas

    for _ in range(n_iter):
        weights = compute_weights(archive_size, q)
        sigmas = compute_sigmas(archive)
        cum_w = np.cumsum(weights)

        new_sols = np.empty((num_ants, dim))
        new_vals = np.empty(num_ants)

        for a in range(num_ants):
            k = min(np.searchsorted(cum_w, rng.random(), side='right'), archive_size - 1)
            sample = rng.normal(loc=archive[k], scale=sigmas[k])
            sample = np.clip(sample, lo, hi)
            new_sols[a] = sample
            new_vals[a] = func(sample)

        # Merge and keep best archive_size solutions
        combined = np.vstack((archive, new_sols))
        combined_vals = np.concatenate((archive_vals, new_vals))
        order = np.argsort(combined_vals)
        archive = combined[order][:archive_size]
        archive_vals = combined_vals[order][:archive_size]

        if archive_vals[0] < best_val:
            best_val = float(archive_vals[0])

    return best_val
