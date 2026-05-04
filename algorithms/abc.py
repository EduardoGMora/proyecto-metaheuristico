import numpy as np

def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        limit: int = 15,
        seed: int = 0,
        **kwargs) -> float:
    """
    Artificial Bee Colony (ABC) for n-dimensional minimization.

    Parameters
    ----------
    func   : callable, receives np.ndarray shape (dim,) and returns float
    dim    : number of dimensions
    bounds : (lo, hi) — search domain
    n_iter : number of iterations
    init   : np.ndarray shape (num_sources, dim) — initial population
    limit  : trials without improvement before a source becomes a scout
    seed   : seed for internal operators (does not affect init)

    Returns
    -------
    best_val : float — best value found (minimum)
    """

    def fitness_to_prob(vals):
        # Invert for minimization: lower value → higher selection probability
        inv = 1.0 / (1.0 + vals - vals.min())
        return inv / inv.sum()

    rng = np.random.default_rng(seed)
    lo, hi = bounds
    num_sources = init.shape[0]

    sources = init.copy().astype(float)
    values  = np.array([func(s) for s in sources])
    trials  = np.zeros(num_sources, dtype=int)

    best_val = float(np.min(values))

    for _ in range(n_iter):
        # ---------- 1) Employed bees ----------
        for i in range(num_sources):
            k = rng.integers(0, num_sources - 1)
            if k >= i:
                k += 1
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v = np.clip(sources[i] + phi * (sources[i] - sources[k]), lo, hi)
            fv = func(v)
            if fv < values[i]:
                sources[i] = v
                values[i] = fv
                trials[i] = 0
            else:
                trials[i] += 1

        idx = np.argmin(values)
        if values[idx] < best_val:
            best_val = values[idx]

        # ---------- 2) Onlooker bees ----------
        probs = fitness_to_prob(values)
        for _ in range(num_sources):
            sel = rng.choice(num_sources, p=probs)
            k = rng.integers(0, num_sources - 1)
            if k >= sel:
                k += 1
            phi = rng.uniform(-1.0, 1.0, size=dim)
            v = np.clip(sources[sel] + phi * (sources[sel] - sources[k]), lo, hi)
            fv = func(v)
            if fv < values[sel]:
                sources[sel] = v
                values[sel] = fv
                trials[sel] = 0
            else:
                trials[sel] += 1

        idx = np.argmin(values)
        if values[idx] < best_val:
            best_val = values[idx]

        # ---------- 3) Scout bees ----------
        for i in range(num_sources):
            if trials[i] >= limit:
                sources[i] = rng.uniform(lo, hi, size=dim)
                values[i] = func(sources[i])
                trials[i] = 0

        idx = np.argmin(values)
        if values[idx] < best_val:
            best_val = values[idx]

    return best_val