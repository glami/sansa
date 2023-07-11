import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from evaluation.pipeline import Pipeline

model_config_mrf = {
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
        "maxInColumn": 1000,
        "rr": 0.0,
        "sparse_evaluation": False,
    },
}

model_config_sansa = {
    "general": {
        "model": "SANSA",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        "l2": 20.0,
        "target_density": 5e-5,  # of factor L and L_inv; also controls memory overhead in UMR
        "ainv_params": {
            "umr_scans": 3,  # number of scans across the whole approximate inverse matrix
            "umr_finetune_steps": 10,  # default 1, can be any non-negative integer
            "umr_loss_threshold": 1e-4,
        },  # all ainv parameters control the length of the iterative process. More scans and steps and higher lower threshold = better result.
        "ldlt_method": "icf",  # cholmod
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
        "experiment_folder": "experiments/sandbox",
    },
}

pipe = Pipeline(model_config=model_config_sansa, eval_config=evaluation_config)

data, _ = pipe.run()
results = data["results"]

print()
print(
    f"Recall@20: {results[20]['recall BARS']['mean']:.6f} +- {results[20]['recall BARS']['se']:.6f}"
)
print(f"nDCG@20: {results[20]['ndcg']['mean']:.6f} +- {results[20]['ndcg']['se']:.6f}")
print()
