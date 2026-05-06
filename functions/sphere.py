import numpy as np


def sphere(x: np.ndarray) -> float:
    """
    Function Sphere n-dimensional.
    Global minimum at x=0 with f(0)=0.
    Recommended domain: [-5.12, 5.12]^n

    Parameters
    ----------
    x : np.ndarray shape (dim,)
    
    Returns
    -------
    float
    """
    return float(np.sum(x**2))