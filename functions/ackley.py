import numpy as np

def ackley(x: np.ndarray) -> float:
    """
    Function Ackley n-dimensional.
    Global minimum at x=(0,...,0) with f=0.
    Recommended domain: [-5, 5]^n
    
    Parameters
    ----------
    x : np.ndarray shape (dim,)

    Returns
    -------
    float
    """
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    return float(
        -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
        - np.exp(sum_cos / n)
        + 20.0
        + np.e
    )
