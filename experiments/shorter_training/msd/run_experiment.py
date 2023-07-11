import itertools
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
        "l2": 200,
        "target_density": None,
        "ainv_params": {
            "umr_scans": None,
            "umr_finetune_steps": None,
            "umr_loss_threshold": 1e-4,
        },
        "ldlt_method": "cholmod",
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
        "experiment_folder": "experiments/shorter_training",
    },
}

# train SANSA models with various densities
densities = [5e-4, 2.5e-3, 1e-2]
# with various number of training steps
nscans = [0, 1, 2, 3, 4]
# and with various number of finetuning steps
nfinetune = [0, 5, 10, 20]

sansa_results_dict = {}
for density, ns, nf in list(itertools.product(densities, nscans, nfinetune)):
    print(f"d={density}, ns={ns}, nf={nf}\n\n")
    sansa_config["model_specific"]["target_density"] = density
    sansa_config["model_specific"]["ainv_params"]["umr_scans"] = ns
    sansa_config["model_specific"]["ainv_params"]["umr_finetune_steps"] = nf
    pipe = Pipeline(model_config=sansa_config, eval_config=evaluation_config)
    data, _ = pipe.run()
    sansa_results_dict[(density, float(ns), float(nf))] = data["results"]


# display results
print("\n\nRecall @ 50:")
for density in densities:
    print(f"density: {density:.2%}")
    print("SANSA")
    for ns, nf in zip(nscans, nfinetune):
        print(
            f"sc={ns}, ft={nf}: {sansa_results_dict[(density, float(ns), float(nf))][50]['recall']['mean']:.3f}"
        )
    print("\n\n")
