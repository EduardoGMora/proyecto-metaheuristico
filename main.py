import pandas as pd

from experiments.runner import run_experiment
from config import BOUNDS, POP_SIZE, N_ITER, N_RUNS, ALGORITHMS, FUNCTIONS, INITS, DIMS

def build_table(dim: int, init_fn: callable) -> pd.DataFrame:
    """
    Build a results table for a given dimension and initialization method.

    Parameters
    ----------
    dim : int — dimension of the problem to optimize
    init_fn : callable — initialization function that generates initial points
    
    Returns
    -------
    pd.DataFrame : A DataFrame containing the results for each algorithm and function.

    """
    rows: list[dict[str, float]] = []
    for algo_name, algo_fn in ALGORITHMS.items():
        row: dict[str, any] = {"Algorithm": algo_name}
        for fn_name, fn in FUNCTIONS.items():
            metrics: dict[str, float] = run_experiment(
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

def get_tables() -> dict:
    """
    Generate results tables for all combinations of dimensions and initialization methods.
    
    Returns    
    -------  
    dict : A dictionary containing the results tables for each configuration.
    """
    tables: dict[str, dict[str, any]] = {}
    for dim in DIMS:
        for init_name, init_fn in INITS.items():
            key = f"Dimensiones = {dim} | Inicialización = {init_name}"
            print(f"\n{'-'*50}")
            print(f"  {key}")
            print(f"{'-'*50}")
            df = build_table(dim, init_fn)
            tables[key] = df
            print(df.to_string())
            store_table(df, dim, init_name)
    return tables

def store_table(table: pd.DataFrame, dim: int, init: str) -> None:
    """
    Store a results table in an Excel file.

    Parameters
    ----------
    table : pd.DataFrame — the results table to store
    dim : int — dimension of the problem
    init : str — name of the initialization method

    Returns
    -------
    None
    """
    filename: str = f"results_dim{dim}_{init.lower().replace(' ', '_')}.xlsx"
    table.to_excel(filename)
    print(f"\nTabla guardada en {filename}")

def main():
    print("Iniciando experimentos...")
    tables = get_tables()
    print(f"Experimentos completados. {len(tables)} Tablas generadas y almacenadas.")

if __name__ == "__main__":
    main()