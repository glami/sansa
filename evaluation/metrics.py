"""
Metrics used for evaluation.
"""

from functools import wraps
from logging import Logger
import numpy as np
from time import perf_counter


# Accuracy-based recommendation quality metrics
def recall(ids_true: np.ndarray, ids_top_k: np.ndarray) -> float:
    k = len(ids_top_k)
    num_true = len(ids_true)
    num_positive = np.sum(np.isin(ids_top_k, ids_true), dtype=np.float32)
    return num_positive / min(k, num_true)


# recall used by openbenchmark BARS
def recall_BARS(ids_true: np.ndarray, ids_top_k: np.ndarray) -> float:
    num_true = len(ids_true)
    num_positive = np.sum(np.isin(ids_top_k, ids_true), dtype=np.float32)
    return num_positive / num_true


def ndcg(ids_true, ids_top_k) -> float:
    k = len(ids_top_k)
    num_true = len(ids_true)
    relevances_top_k = np.isin(ids_top_k, ids_true).astype(np.float32)
    with np.errstate(
        divide="ignore"
    ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
        tp = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (relevances_top_k * tp).sum()
    idcg = tp[: min(num_true, k)].sum()
    return dcg / idcg


# Wrapper for measuring execution time
def execution_time(logger: Logger) -> callable:
    def decorate(func: callable) -> callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            run_time = end_time - start_time
            logger.info(f"Execution of {func.__name__} took at {run_time:.3f} seconds.")
            return result, run_time

        return wrapper

    return decorate
