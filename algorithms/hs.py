import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        HMCR: float = 0.95,
        PAR: float = 0.3,
        BW: float = 0.1,
        seed: int = 0,
        **kwargs) -> float:
    """
    Harmony Search (HS) Algorithm for n-dimensional minimization.

    Improvises new harmonies by combining memory consideration, pitch 
    adjustment, and random selection. If a new harmony is better (lower) 
    than the worst in the memory, it replaces it.

    Parameters
    ----------
    func    : callable, receives np.ndarray shape (dim,) and returns float
    dim     : number of dimensions
    bounds  : (lo, hi) — search domain
    n_iter  : number of iterations (improvisations)
    init    : np.ndarray shape (HMS, dim) — initial Harmony Memory
    HMCR    : Harmony Memory Considering Rate (probability of choosing from HM)
    PAR     : Pitch Adjust Rate (probability of adjusting the chosen note)
    BW      : Bandwidth for pitch adjustment (fraction of the domain size)
    seed    : seed for internal random generator

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    HMS = len(init)

    HM = init.copy().astype(float)
    HM_vals = np.array([func(p) for p in HM])

    best_val = float(np.min(HM_vals))

    for _ in range(n_iter):
        new = np.empty(dim)

        for d in range(dim):
            if rng.random() < HMCR:
                idx = rng.integers(0, HMS)
                val = HM[idx, d]

                if rng.random() < PAR:
                    delta = BW * (hi - lo)
                    val = val + rng.uniform(-delta, delta)
            else:

                val = rng.uniform(lo, hi)
                

            new[d] = np.clip(val, lo, hi)


        new_val = func(new)


        worst_idx = np.argmax(HM_vals)
        
        if new_val < HM_vals[worst_idx]:
            HM[worst_idx] = new
            HM_vals[worst_idx] = new_val

        iter_best = float(np.min(HM_vals))
        if iter_best < best_val:
            best_val = iter_best

    return best_val