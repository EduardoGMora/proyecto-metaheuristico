import numpy as np

def ackley(x: np.ndarray) -> float:
    """
    Función Ackley n-dimensional. Mínimo global en x=0 con f=0.
    Dominio recomendado: [-32.768, 32.768]^n
    x : np.ndarray shape (dim,)
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
