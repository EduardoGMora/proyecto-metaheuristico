import functions as f
from algorithms import (
    abc, 
    aco, 
    ga, 
    fa, 
    pso, 
    local_search, 
    random_search, 
    climbing, 
    sa, 
    hs
)
from experiments.initializations import uniform, latin_hypercube

BOUNDS   = (-3, 3)
POP_SIZE = 20
N_ITER   = 80
N_RUNS   = 30

ALGORITHMS = {
    "ABC":           abc.run,
    "ACO":           aco.run,
    "GA":            ga.run,
    "FA":            fa.run,
    "PSO":           pso.run,
    "Local Search":  local_search.run,
    "Random Search": random_search.run,
    "Climbing":      climbing.run,
    "SA":            sa.run,
    "HS":            hs.run,
}

FUNCTIONS = {
    "Rastrigin":  f.rastrigin,
    "Sphere":     f.sphere,
    "Rosenbrock": f.rosenbrock,
    "Ackley":     f.ackley,
}

INITS = {
    "Uniform":         uniform,
    "Latin Hypercube": latin_hypercube,
}

DIMS = [3, 10]
