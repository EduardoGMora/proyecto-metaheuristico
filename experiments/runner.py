import numpy as np

def run_experiment(algo_fn, func, dim, init_fn, num_runs=30, num_iter=None, **kwargs):
    results = []
    for _ in range(num_runs):
        init_pos = init_fn(dim)
        result = algo_fn(func=func, init_pos=init_pos, num_iter=num_iter, **kwargs)
        results.append(result)

    arr = np.array(results)
    return {
        "AB": np.mean(arr),
        "MD": np.median(arr),
        "SD": np.std(arr)
    }