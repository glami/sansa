#
# Copyright 2023 Inspigroup s.r.o.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# https://github.com/glami/sansa/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
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
    "model_specific": {"l2": 500},
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
        "alpha": 0.75,
        # parameters for the sparse solution of the MSD data:
        # choose a sparsity level
        "threshold": 0.175,  # results in sparsity 2 % (for alpha=0.75)
        # "threshold": 0.3,  # results in sparsity 1 % (for alpha=0.75)
        # "threshold": 0.458,  # results in sparsity 0.5 % (for alpha=0.75)
        "maxInColumn": 1000,
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
        "l2": 500,  # same as EASE
        "target_density": 1e-2,  # of factor L and L_inv; also controls memory overhead in UMR
        "ainv_params": {
            "umr_scans": 4,  # number of scans across the whole approximate inverse matrix
            "umr_finetune_steps": 10,  # default 1, can be any non-negative integer
            "umr_loss_threshold": 1e-4,
        },  # all ainv parameters control the length of the iterative process. More scans and steps and higher lower threshold = better result.
        "ldlt_method": "cholmod",  # currently only cholmod is supported
        "ldlt_params": {},  # hyperparameters for cholmod (other reorderings, etc.). Default parameters are fine (uses COLAMD). For larger systems, it may be necessary to set "use_long": True.
    },
}

evaluation_config = {
    "dataset": {
        "name": "movielens20",
        "rewrite": False,
    },
    "split": {
        "n_val_users": 10000,
        "n_test_users": 10000,
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
