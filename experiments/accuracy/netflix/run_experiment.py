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

ease_config = {
    "general": {
        "model": "EASE",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {"l2": 1200},
}

mrf_config = {
    "general": {
        "model": "MRF",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        "l2": 1.0,  # from MRF paper
        "alpha": 0.75,  # from MRF paper
        "threshold": 0.3,  # results in sparsity 2 % (for alpha=0.75)
        # "threshold": 0.64,  # results in sparsity 1 % (for alpha=0.75)
        # "threshold": 0.999,  # results in sparsity 0.5 % (for alpha=0.75)
        "maxInColumn": 1000,  # from MRF paper
        "rr": None,
        "sparse_evaluation": False,
    },
}

sansa_config = {
    "general": {
        "model": "SANSA",
        "checkpoints_folder": "models/checkpoints/",
        "retrain": True,
        "save": False,
    },
    "model_specific": {
        "l2": 1200,  # same as EASE
        "target_density": None,
        "ainv_params": {
            "umr_scans": 4,
            "umr_finetune_steps": 10,
            "umr_loss_threshold": 1e-4,
        },
        "ldlt_method": "cholmod",
    },
}

evaluation_config = {
    "dataset": {
        "name": "netflix",
        "rewrite": False,
    },
    "split": {
        "n_val_users": 40000,
        "n_test_users": 40000,
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

# train EASE
pipe = Pipeline(model_config=ease_config, eval_config=evaluation_config)
data, _ = pipe.run()
ease_results = data["results"]

# train MRF
thresholds = [0.999, 0.64, 0.3]
rs = [0.5, 0.1, 0]
mrf_results_dict = {}
for th in thresholds:
    for r in rs:
        mrf_config["model_specific"]["threshold"] = th
        mrf_config["model_specific"]["rr"] = r
        pipe = Pipeline(model_config=mrf_config, eval_config=evaluation_config)
        data, _ = pipe.run()
        mrf_results_dict[(th, r)] = data["results"]

# train SANSA models with various densities
densities = [2.5e-3, 5e-3, 1e-2]
sansa_results_dict = {}
for density in densities:
    sansa_config["model_specific"]["target_density"] = density
    pipe = Pipeline(model_config=sansa_config, eval_config=evaluation_config)
    data, _ = pipe.run()
    sansa_results_dict[density] = data["results"]

# print results in human readable form
print("\n\nRecall @ 20:")
print(
    f" EASE : {ease_results[20]['recall']['mean']:.4f} +- {ease_results[20]['recall']['se']:.4f}"
)
print(f"MRF")
for th in thresholds:
    for r in rs:
        print(
            f"th={th:.3f}, r={r:.1f}: {mrf_results_dict[(th, r)][20]['recall']['mean']:.4f} +- {mrf_results_dict[(th, r)][20]['recall']['se']:.4f}"
        )
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][20]['recall']['mean']:.4f} +- {sansa_results_dict[density][20]['recall']['se']:.4f}"
    )

print("\n\nRecall @ 50:")
print(
    f" EASE : {ease_results[50]['recall']['mean']:.4f} +- {ease_results[50]['recall']['se']:.4f}"
)
print(f"MRF")
for th in thresholds:
    for r in rs:
        print(
            f"th={th:.3f}, r={r:.1f}: {mrf_results_dict[(th, r)][50]['recall']['mean']:.4f} +- {mrf_results_dict[(th, r)][50]['recall']['se']:.4f}"
        )
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][50]['recall']['mean']:.4f} +- {sansa_results_dict[density][50]['recall']['se']:.4f}"
    )

print("\n\nnDCG @ 100:")
print(
    f" EASE : {ease_results[100]['ndcg']['mean']:.4f} +- {ease_results[100]['ndcg']['se']:.4f}"
)
print(f"MRF")
for th in thresholds:
    for r in rs:
        print(
            f"th={th:.3f}, r={r:.1f}: {mrf_results_dict[(th, r)][100]['ndcg']['mean']:.4f} +- {mrf_results_dict[(th, r)][100]['ndcg']['se']:.4f}"
        )
print(f"SANSA")
for density in densities:
    print(
        f"{density:.4f}: {sansa_results_dict[density][100]['ndcg']['mean']:.4f} +- {sansa_results_dict[density][100]['ndcg']['se']:.4f}"
    )
