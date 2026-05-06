import numpy as np

def rosenbrock(x: np.ndarray) -> float:
    """
    Función Rosenbrock n-dimensional (función banana de Valley).
    Mínimo global en x=(1,...,1) con f=0.
    Dominio recomendado: [-5, 10]^n
    x : np.ndarray shape (dim,)
    """
    xi = x[:-1]
    xi1 = x[1:]
    return float(np.sum(100.0 * (xi1 - xi**2)**2 + (1.0 - xi)**2))
