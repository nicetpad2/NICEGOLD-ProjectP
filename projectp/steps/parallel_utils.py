
from joblib import Parallel, delayed
import multiprocessing
def parallel_fit(models, X, y):
    n_jobs = min(len(models), multiprocessing.cpu_count())
    results = Parallel(n_jobs = n_jobs)(delayed(lambda m: m.fit(X, y))(model) for model in models)
    return results