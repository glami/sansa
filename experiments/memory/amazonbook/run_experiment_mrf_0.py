import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from evaluation.pipeline import Pipeline

mrf_config = {
    "general": {
        "model": "MRF",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        "l2": 2.0,
        "alpha": 0.75,  # from MRF paper
        "threshold": 0.480,  # results in sparsity 0.01%
        "maxInColumn": 1000,  # from MRF paper
        "rr": 0,
        "sparse_evaluation": False,
    },
}

evaluation_config = {
    "dataset": {
        "name": "amazonbook",
        "rewrite": False,
    },
    "split": {
        "seed": 42,
        "val_target_proportion": 0.0,
    },
    "evaluation": {
        "split": "test",
        "metrics": ["recall BARS", "ndcg"],
        "ks": [20],
        "experiment_folder": "experiments/memory/",
    },
}

# train MRF
pipe = Pipeline(model_config=mrf_config, eval_config=evaluation_config)
data, _ = pipe.run()
