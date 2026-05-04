import pandas as pd
from functions.rastrigin import rastrigin
from functions.sphere import sphere
import algorithms
from experiments.runner import run_experiment
from experiments.initializations import uniform, latin_hypercube

ALGORITHMS = {"ABC": algorithms.run_abc, "Local Search": algorithms.run_local_search, "Random Search": algorithms.run_random_search}
FUNCTIONS  = {"Rastrigin": rastrigin, "Sphere": sphere}
INITS      = {"Uniforme": uniform, "Latin HC": latin_hypercube}
DIMS       = [3, 10]

# genera las 4 tablas: 2 dims × 2 inits
for dim in DIMS:
    for init_name, init_fn in INITS.items():
        rows = []
        for algo_name, algo_fn in ALGORITHMS.items():
            row = {"Algoritmo": algo_name}
            for fn_name, fn in FUNCTIONS.items():
                metrics = run_experiment(algo_fn, fn, dim, ...)
                row[f"{fn_name} AB"] = metrics["AB"]
                row[f"{fn_name} MD"] = metrics["MD"]
                row[f"{fn_name} SD"] = metrics["SD"]
            rows.append(row)
        df = pd.DataFrame(rows).set_index("Algoritmo")
        print(f"\n--- dim={dim}, init={init_name} ---")
        print(df.to_string())

def main():
    for algo_name, algo_fn in ALGORITHMS.items():
        for fn_name, fn in FUNCTIONS.items():
            print(f"Running {algo_name} on {fn_name}...")
            metrics = run_experiment(algo_fn, fn, dim=10, bounds=(-5.12, 5.12), n_iter=1000, init_fn=uniform)
            print(f"AB: {metrics['AB']:.4f}, MD: {metrics['MD']:.4f}, SD: {metrics['SD']:.4f}")

if __name__ == "__main__":
    main()