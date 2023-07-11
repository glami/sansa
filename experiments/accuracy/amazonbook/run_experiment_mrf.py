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
        "rr": None,
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
        "experiment_folder": "experiments/accuracy",
    },
}

# train MRF
rs = [0.5, 0]
for r in rs:
    mrf_config["model_specific"]["rr"] = r
    pipe = Pipeline(model_config=mrf_config, eval_config=evaluation_config)
    data, _ = pipe.run()
