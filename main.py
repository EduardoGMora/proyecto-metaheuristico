import pandas as pd

from functions.rastrigin import rastrigin
from functions.sphere import sphere

from algorithms import abc, aco, ga, fa, pso, local_search, random_search

from experiments.runner import run_experiment
from experiments.initializations import uniform, latin_hypercube

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BOUNDS    = (-3, 3)
POP_SIZE  = 20
N_ITER    = 80
N_RUNS    = 30

ALGORITHMS = {
    "ABC":           abc.run,
    # "ACO":           aco.run,
    # "GA":            ga.run,
    # "FA":            fa.run,
    # "PSO":           pso.run,
    # "Local Search":  local_search.run,
    # "Random Search": random_search.run,
}

FUNCTIONS = {
    "Rastrigin": rastrigin,
    "Sphere":    sphere,
}

INITS = {
    "Uniform":         uniform,
    "Latin Hypercube": latin_hypercube,
}

DIMS = [3, 10]

# ---------------------------------------------------------------------------
# Generate tables: 2 dims × 2 initializations = 4 tables
# ---------------------------------------------------------------------------
def build_table(dim: int, init_name: str, init_fn: callable) -> pd.DataFrame:
    rows = []
    for algo_name, algo_fn in ALGORITHMS.items():
        row = {"Algorithm": algo_name}
        for fn_name, fn in FUNCTIONS.items():
            metrics = run_experiment(
                algo_fn=algo_fn,
                func=fn,
                dim=dim,
                bounds=BOUNDS,
                n_iter=N_ITER,
                init_fn=init_fn,
                pop_size=POP_SIZE,
                n_runs=N_RUNS,
            )
            row[f"{fn_name} AB"] = round(metrics["AB"], 4)
            row[f"{fn_name} MD"] = round(metrics["MD"], 4)
            row[f"{fn_name} SD"] = round(metrics["SD"], 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Algorithm")


def main():
    tables = {}
    for dim in DIMS:
        for init_name, init_fn in INITS.items():
            key = f"Dimensiones = {dim} | Inicialización = {init_name}"
            print(f"\n{'='*60}")
            print(f"  {key}")
            print(f"{'='*60}")
            df = build_table(dim, init_name, init_fn)
            tables[key] = df
            print(df.to_string())
    return tables


if __name__ == "__main__":
    main()