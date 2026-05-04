import numpy as np


def run(func: callable,
        dim: int,
        bounds: tuple,
        n_iter: int,
        init: np.ndarray,
        prob_crossover: float = 0.8,
        prob_mutation: float = 0.2,
        mutation_scale: float = 0.3,
        seed: int = 0,
        **kwargs) -> float:
    """
    Genetic Algorithm (GA) for n-dimensional minimization.

    Parameters
    ----------
    func            : callable, receives np.ndarray shape (dim,) and returns float
    dim             : number of dimensions
    bounds          : (lo, hi) — search domain
    n_iter          : number of generations
    init            : np.ndarray shape (pop_size, dim) — initial population
    prob_crossover  : crossover probability
    prob_mutation   : per-individual mutation probability
    mutation_scale  : uniform noise half-range applied during mutation
    seed            : seed for internal operators (does not affect init)

    Returns
    -------
    best_val : float — best value found (minimum)
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds

    population = init.copy().astype(float)
    pop_size = len(population)

    fitness = np.array([func(ind) for ind in population])
    best_val = float(np.min(fitness))

    for _ in range(n_iter):
        # --- Selection: roulette wheel (inverted for minimization) ---
        inv = 1.0 / (1.0 + fitness - fitness.min())
        probs = inv / inv.sum()
        parent_idx = rng.choice(pop_size, size=pop_size, p=probs)
        parents = population[parent_idx]

        # --- Crossover: blend (BLX-alpha style with alpha=random) ---
        offspring = []
        for i in range(0, pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % pop_size]
            if rng.random() < prob_crossover:
                alpha = rng.random()
                h1 = alpha * p1 + (1 - alpha) * p2
                h2 = alpha * p2 + (1 - alpha) * p1
                offspring.extend([h1, h2])
            else:
                offspring.extend([p1.copy(), p2.copy()])
        population = np.array(offspring[:pop_size])

        # --- Mutation ---
        for ind in population:
            if rng.random() < prob_mutation:
                ind += rng.uniform(-mutation_scale, mutation_scale, size=dim)
                np.clip(ind, lo, hi, out=ind)

        # --- Evaluate ---
        fitness = np.array([func(ind) for ind in population])
        gen_best = float(np.min(fitness))
        if gen_best < best_val:
            best_val = gen_best

    return best_val
