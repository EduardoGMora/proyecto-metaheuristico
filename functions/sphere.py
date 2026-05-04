import numpy as np


def sphere(x: np.ndarray) -> float:
    """
    Función Sphere n-dimensional. Mínimo global en x=0 con f(0)=0.
    x : np.ndarray shape (dim,)
    """
    return float(np.sum(x**2))