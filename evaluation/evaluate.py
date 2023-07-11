"""
Evaluation functions.
"""

import numpy as np
import pandas as pd

from datasets.split import DatasetSplit
from evaluation.logs import evaluation_logger
from evaluation.metrics import ndcg, recall, recall_BARS
from models.model import Model


def get_stats(metrics: list[float]) -> dict:
    """
    Summary statistics of a list of metrics.
    """
    stats = {
        "mean": np.average(metrics),
        "std": np.std(metrics),
        "se": np.std(metrics) / np.sqrt(len(metrics)),
        "min": np.min(metrics),
        "max": np.max(metrics),
        "percentiles": {
            1: np.percentile(metrics, 1),
            5: np.percentile(metrics, 5),
            10: np.percentile(metrics, 10),
            25: np.percentile(metrics, 25),
            50: np.percentile(metrics, 50),
            75: np.percentile(metrics, 75),
            90: np.percentile(metrics, 90),
            95: np.percentile(metrics, 95),
            99: np.percentile(metrics, 99),
        },
    }
    return stats


def evaluate(
    model: Model,
    split: DatasetSplit,
    metrics: list[str] = ["recall", "ndcg", "coverage"],
    ks: list[int] = [20, 50, 100],
    batch_size: int = 2000,
) -> dict[int : dict[str:dict]]:
    """
    Batched evaluation of a model on a split of a dataset.
    """
    stats = {
        k: {
            "coverage": 0.0,
        }
        for k in ks
    }
    max_k = max(ks)

    total_users = len(split.user_encoder.classes_)

    recalls = {k: [] for k in ks}
    recalls_BARS = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}

    for batch_start in range(0, total_users, batch_size):
        if batch_start + batch_size > total_users:
            batch_end = total_users
        else:
            batch_end = batch_start + batch_size
        evaluation_logger.info(f"Evaluating model at batch {batch_start}:{batch_end}")
        batch_users = list(split.user_encoder.classes_)[batch_start:batch_end]
        batch_data = split.get_rated_items(batch_users)
        batch_targets = split.get_target_items(batch_users)
        batch_target_ids_dict = (
            batch_targets.groupby("user_id", group_keys=True)["item_id"]
            .apply(list)
            .to_dict()
        )
        batch_keys = list(batch_target_ids_dict.keys())
        users_to_arange = {user: i for i, user in enumerate(batch_keys)}
        pd.options.mode.chained_assignment = None  # suppress irrelevant warning
        batch_data["user_id"] = batch_data["user_id"].map(users_to_arange)
        pd.options.mode.chained_assignment = "warn"
        batch_top_maxk_ids, batch_top_maxk_scores = model.recommend(batch_data, k=max_k)

        for k in ks:
            recalls_batch = []
            recalls_BARS_batch = []
            ndcgs_batch = []

            for i in range(len(batch_users)):
                target_ids = batch_target_ids_dict[batch_keys[i]]
                top_k_ids = batch_top_maxk_ids[i, :k]
                recalls_batch.append(recall(target_ids, top_k_ids))
                recalls_BARS_batch.append(recall_BARS(target_ids, top_k_ids))
                ndcgs_batch.append(ndcg(target_ids, top_k_ids))

            recalls[k] += recalls_batch
            recalls_BARS[k] += recalls_BARS_batch
            ndcgs[k] += ndcgs_batch

    for k in ks:
        if "recall" in metrics:
            stats[k]["recall"] = get_stats(metrics=recalls[k])
        if "recall BARS" in metrics:
            stats[k]["recall BARS"] = get_stats(metrics=recalls_BARS[k])
        if "ndcg" in metrics:
            stats[k]["ndcg"] = get_stats(metrics=ndcgs[k])

    return stats
