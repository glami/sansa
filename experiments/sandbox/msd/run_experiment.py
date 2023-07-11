import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from evaluation.pipeline import Pipeline

model_config_ease = {
    "general": {
        "model": "EASE",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {"l2": 200},
}


model_config_mrf = {
    "general": {
        "model": "MRF",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        "l2": 1.0,  # from MRF paper
        # parameter for pre-computation of the training data (see the paper for details):
        #   exponent for re-scaling of the training data
        #   if alpha=1, then XtX becomes the correlation matrix below
        # parameters for the sparse solution of the MSD data:
        # choose a sparsity level
        "alpha": 0.75,  # from MRF paper
        # "threshold": 0.375,  # results in sparsity 0.1 % (for alpha=0.75)
        # "threshold": 0.21,  # results in sparsity 0.2 % (for alpha=0.75)
        # "threshold": 0.11,    # results in sparsity 0.5 % (for alpha=0.75)
        # "threshold": 0.069,    # results in sparsity 1.0 % (for alpha=0.75)
        # "threshold": 0.05,    # results in sparsity 1.5 % (for alpha=0.75)
        "threshold": 0.038,  # results in sparsity 2.0 % (for alpha=0.75)
        "maxInColumn": 1000,  # from MRF paper
        "rr": 0.5,  # hyper-parameter r in the paper, which determines the trade-off between approximation-accuracy and training-time
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
        "l2": 200,  # for cholmod variant; same as EASE
        # "l2": 2.5,  # for icf variant
        "target_density": 1e-2,
        "ainv_params": {
            "umr_scans": 4,
            "umr_finetune_steps": 10,
            "umr_loss_threshold": 1e-4,
        },
        "ldlt_method": "cholmod",
        "ldlt_params": {},  # hyperparameters for cholmod (other reorderings, etc.). Default parameters are fine (uses COLAMD). For larger systems, it may be necessary to set "use_long": True.
    },
}

evaluation_config = {
    "dataset": {
        "name": "msd",
        "rewrite": False,
    },
    "split": {
        "n_val_users": 50000,
        "n_test_users": 50000,
        "seed": 42,
        "target_proportion": 0.2,
        "targets_newest": False,
    },
    "evaluation": {
        "split": "test",
        "metrics": ["recall", "ndcg"],
        "ks": [20, 50, 100],
        "experiment_folder": "experiments/sandbox",
    },
}

pipe = Pipeline(model_config=model_config_ease, eval_config=evaluation_config)
data, _ = pipe.run()
results = data["results"]

print()
print(
    f"Recall@20: {results[20]['recall']['mean']:.6f} +- {results[20]['recall']['se']:.6f}"
)
print(
    f"Recall@50: {results[50]['recall']['mean']:.6f} +- {results[50]['recall']['se']:.6f}"
)
print(
    f"nDCG@100: {results[100]['ndcg']['mean']:.6f} +- {results[100]['ndcg']['se']:.6f}"
)
print()
print(
    results[20]["recall"]["mean"]
    + results[50]["recall"]["mean"]
    + results[100]["ndcg"]["mean"]
)
print()
