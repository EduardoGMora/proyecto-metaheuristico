import numpy as np


def run_experiment(algo_fn: callable,
                   func: callable,
                   dim: int,
                   bounds: tuple,
                   n_iter: int,
                   init_fn: callable,
                   pop_size: int,
                   n_runs: int = 30,
                   **algo_kwargs) -> dict:
    """
    Run an algorithm n_runs times and return AB, MD, SD metrics.

    Parameters
    ----------
    algo_fn    : algorithm function with signature run(func, dim, bounds, n_iter, init, **kwargs)
    func       : objective function, receives np.ndarray shape (dim,) and returns float
    dim        : number of dimensions
    bounds     : (lo, hi) — search domain
    n_iter     : number of iterations passed to the algorithm
    init_fn    : initialization function with signature f(pop_size, dim, bounds, rng) -> np.ndarray
    pop_size   : number of individuals in the initial population
    n_runs     : number of independent runs (default 30)
    algo_kwargs: extra keyword arguments forwarded to algo_fn

    Returns
    -------
    dict with keys "AB" (mean), "MD" (median), "SD" (std deviation)
    """
    results = []
    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        init = init_fn(pop_size, dim, bounds, rng)
        best = algo_fn(func, dim, bounds, n_iter, init, seed=seed, **algo_kwargs)
        results.append(best)

    arr = np.array(results)
    return {"AB": float(np.mean(arr)),
            "MD": float(np.median(arr)),
            "SD": float(np.std(arr))}