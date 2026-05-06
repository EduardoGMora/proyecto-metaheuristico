import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        t_initial: float = 100.0,
        cooling_rate: float = 0.95,
        step_size: float = 0.1,
        seed: int = 0,
        **kwargs) -> float:
    """
    Simulated Annealing (SA) Algorithm for n-dimensional minimization.

    Explores the search space by occasionally accepting worse solutions 
    to escape local minima. The probability of accepting a worse solution 
    decreases over time as the "temperature" cools down.

    Parameters
    ----------
    func         : callable, receives np.ndarray shape (dim,) and returns float
    dim          : number of dimensions
    bounds       : (lo, hi) — search domain
    n_iter       : number of iterations (cooling steps)
    init         : np.ndarray shape (num_agents, dim) — initial starting points
    t_initial    : initial temperature
    cooling_rate : multiplicative decay factor for the temperature (0 < cooling_rate < 1)
    step_size    : standard deviation of the Gaussian noise for neighbor generation
    seed         : seed for internal random generator

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    num_agents = len(init)

    positions = init.copy().astype(float)
    values = np.array([func(p) for p in positions])
    best_val = float(np.min(values))

    T = t_initial

    for _ in range(n_iter):
        for i in range(num_agents):
            curr_pos = positions[i]
            curr_val = values[i]

            noise = rng.normal(loc=0.0, scale=step_size, size=dim)
            neighbor_pos = curr_pos + noise
            neighbor_pos = np.clip(neighbor_pos, lo, hi)

            neighbor_val = func(neighbor_pos)

            # Calcular la diferencia (Delta E)
            delta_e = neighbor_val - curr_val

            # Criterio de aceptacion de Metropolis
            # Si es mejor (delta_e < 0), lo aceptamos siempre
            # Si es peor, lo aceptamos con una probabilidad basada en la temperatura
            if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                positions[i] = neighbor_pos
                values[i] = neighbor_val

                # Actualizar el mejor global si encontramos un nuevo mínimo absoluto
                if neighbor_val < best_val:
                    best_val = float(neighbor_val)
        

        T *= cooling_rate

    return best_val