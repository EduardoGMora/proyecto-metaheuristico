import numpy as np

def latin_hypercube(n_samples: int, dim: int, bounds: tuple) -> np.ndarray:
    """Genera una muestra de puntos utilizando el método de Latin Hypercube Sampling."""
    lower, upper = bounds
    # Crear una matriz de índices para cada dimensión
    indices = np.array([np.random.permutation(n_samples) for _ in range(dim)])
    # Escalar los índices a los límites del espacio de búsqueda
    samples = lower + (indices + np.random.rand(*indices.shape)) * (upper - lower) / n_samples
    return samples.T  # Transponer para que cada fila sea un punto


def uniform(n_samples: int, dim: int, bounds: tuple) -> np.ndarray:
    """Genera una muestra de puntos utilizando distribución uniforme."""
    lower, upper = bounds
    return np.random.uniform(lower, upper, size=(n_samples, dim))
