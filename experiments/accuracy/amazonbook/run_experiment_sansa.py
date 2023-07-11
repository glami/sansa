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
        "l2": 20.0,
        "target_density": 5e-5,
        "ainv_params": {
            "umr_scans": 3,
            "umr_finetune_steps": 10,
            "umr_loss_threshold": 1e-4,
        },
        "ldlt_method": "icf",
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
        "experiment_folder": "experiments/accuracy",
    },
}


# train SANSA
pipe = Pipeline(model_config=sansa_config, eval_config=evaluation_config)
pipe.run()
