import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """
    Function Rastrigin n-dimensional.
    Global minimum at x=(0,...,0) with f=0.
    Recommended domain: [-5.12, 5.12]^n

    Parameters
    ----------
    x : np.ndarray shape (dim,)
    
    Returns
    -------
    float
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))