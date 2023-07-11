import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../../")

from evaluation.pipeline import Pipeline

sansa_config = {
    "general": {
        "model": "SANSA",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        # "l2": 200,  # same as EASE
        "l2": 2.5,  # for icf variant
        "target_density": None,
        "ainv_params": {
            "umr_scans": 4,
            "umr_finetune_steps": 10,
            "umr_loss_threshold": 1e-4,
        },
        "ldlt_method": "icf",
        "ldlt_params": {},
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
        "experiment_folder": "experiments/accuracy",
    },
}

# train SANSA icf with various densities
densities = [5e-4, 2.5e-3, 1e-2]
sansa_results_dict = {}
for density in densities:
    sansa_config["model_specific"]["target_density"] = density
    pipe = Pipeline(model_config=sansa_config, eval_config=evaluation_config)
    data, _ = pipe.run()
    sansa_results_dict[density] = data["results"]

# print results in human readable form
print("\n\nRecall @ 20:")
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][20]['recall']['mean']:.4f} +- {sansa_results_dict[density][20]['recall']['se']:.4f}"
    )

print("\n\nRecall @ 50:")
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][50]['recall']['mean']:.4f} +- {sansa_results_dict[density][50]['recall']['se']:.4f}"
    )

print("\n\nnDCG @ 100:")
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][100]['ndcg']['mean']:.4f} +- {sansa_results_dict[density][100]['ndcg']['se']:.4f}"
    )
