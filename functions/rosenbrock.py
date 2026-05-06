import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """
    Function Rosenbrock n-dimensional (banana function).
    Global minimum at x=(1,...,1) with f=0.
    Recommended domain: [-5, 10]^n

    Parameters
    ----------
    x : np.ndarray shape (dim,)
    
    Returns
    -------
    float
    """
    xi = x[:-1]
    xi1 = x[1:]
    return float(np.sum(100.0 * (xi1 - xi**2)**2 + (1.0 - xi)**2))
