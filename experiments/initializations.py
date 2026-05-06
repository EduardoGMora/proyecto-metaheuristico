from PIL.ImageChops import constant
import numpy as np

def uniform(pop_size: int, dim: int, bounds: tuple, rng: np.random.Generator) -> np.ndarray:
    """
    Uniform random initialization.
    Returns np.ndarray shape (pop_size, dim).

    Parameters
    ----------
    pop_size : int — number of points to generate
    dim      : int — number of dimensions
    bounds   : (lo, hi) — search domain
    rng      : np.random.Generator — random number generator for reproducibility

    Returns
    -------
    np.ndarray : shape (pop_size, dim)
    """
    lo: float
    hi: float
    lo, hi = bounds
    return rng.uniform(lo, hi, size=(pop_size, dim))


def latin_hypercube(pop_size: int, dim: int, bounds: tuple, rng: np.random.Generator) -> np.ndarray:
    """
    Latin Hypercube Sampling initialization.
    Divides each dimension into pop_size equal strata and samples one point per stratum.
    Returns np.ndarray shape (pop_size, dim).

    Parameters
    ----------
    pop_size : int — number of points to generate
    dim      : int — number of dimensions
    bounds   : (lo, hi) — search domain
    rng      : np.random.Generator — random number generator for reproducibility

    Returns
    -------
    np.ndarray : shape (pop_size, dim)
    """
    lo: float
    hi: float
    lo, hi = bounds
    samples: np.ndarray = np.empty((pop_size, dim))
    for d in range(dim):
        perm: np.ndarray = rng.permutation(pop_size)
        samples[:, d] = lo + (perm + rng.random(pop_size)) * (hi - lo) / pop_size
    return samples
