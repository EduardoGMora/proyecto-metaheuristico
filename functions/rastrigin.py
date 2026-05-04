import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """
    Función Rastrigin n-dimensional. Mínimo global en x=0 con f(0)=0.
    x : np.ndarray shape (dim,)
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))