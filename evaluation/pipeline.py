"""
Evaluation pipeline used to run experiments.
"""

import datetime
import json
import os
from pathlib import Path

from datasets.dataset import Dataset
from datasets.amazonbook import Amazonbook
from datasets.goodbooks10 import Goodbooks
from datasets.movielens20 import Movielens
from datasets.msd import MSD
from datasets.netflix import Netflix
from datasets.split import DatasetSplit
from evaluation.evaluate import evaluate
from evaluation.logs import (
    start_logger,
    dataset_logger,
    training_logger,
    evaluation_logger,
    end_logger,
)
from evaluation.metrics import execution_time
from evaluation.steps import PIPELINE_STEPS
from models.model import Model
from models.ease import EASE
from models.sansa import SANSA
from models.mrf import MRF


class Pipeline:
    def __init__(self, model_config: dict, eval_config: dict):
        self.train_split: DatasetSplit = None
        self.val_split: DatasetSplit = None
        self.test_split: DatasetSplit = None
        self.model: Model = None
        self.model_config = model_config
        self.eval_config = eval_config

        self.step = PIPELINE_STEPS["start"]
        self.logger = start_logger
        self.experiment_data = {}

    @execution_time(logger=dataset_logger)
    def _create_dataset_splits(self) -> None:
        """
        Create train, validation and test dataset splits.
        """

        def _log_dataset_info(info_dict: dict) -> None:
            msg = f"Dataset info | "
            for key, value in info_dict.items():
                msg += f"{key}: {value}, "
            msg = msg[:-2]
            self.logger.info(msg=msg)

        def _config_to_name(config: dict) -> str:
            name = ""
            for k, v in config.items():
                name += f"{k}={v}_"
            return name[:-1]

        def _evaluate_load_splits(splits_folder) -> bool:
            if not os.path.exists(splits_folder):
                os.makedirs(splits_folder)
                return False
            if (
                not os.path.exists(os.path.join(splits_folder, "train.pickle"))
                or not os.path.exists(os.path.join(splits_folder, "val.pickle"))
                or not os.path.exists(os.path.join(splits_folder, "test.pickle"))
            ):
                return False
            if self.eval_config["dataset"]["rewrite"]:
                return False
            return True

        @execution_time(logger=dataset_logger)
        def _load_dataset_from_config(config: dict) -> Dataset:
            name = config["name"]
            if name == "amazonbook":
                dataset = Amazonbook.from_config(config)
            elif name == "goodbooks10":
                dataset = Goodbooks.from_config(config)
            elif name == "movielens20":
                dataset = Movielens.from_config(config)
            elif name == "msd":
                dataset = MSD.from_config(config)
            elif name == "netflix":
                dataset = Netflix.from_config(config)
            else:
                raise ValueError(
                    f"Invalid dataset name {name}, use one of available datasets: amazonbook, goodbooks10, movielens20, msd, netflix"
                )
            return dataset

        def _get_splits_from_eval_config(eval_config: dict) -> float:
            name_from_config = _config_to_name(eval_config["split"])
            splits_folder = f"{eval_config['dataset'].get('folder', 'datasets/data')}/{eval_config['dataset']['name']}/{name_from_config}"

            if _evaluate_load_splits(splits_folder=splits_folder):
                self.train_split = DatasetSplit.load(
                    os.path.join(splits_folder, "train.pickle")
                )
                self.val_split = DatasetSplit.load(
                    os.path.join(splits_folder, "val.pickle")
                )
                self.test_split = DatasetSplit.load(
                    os.path.join(splits_folder, "test.pickle")
                )
                self.logger.info(msg=f"Loaded dataset splits from {splits_folder}.")
                return 0.0

            (
                self.train_split,
                self.val_split,
                self.test_split,
            ), split_time = dataset.create_splits(config=eval_config["split"])
            self.train_split.save(os.path.join(splits_folder, "train.pickle"))
            self.val_split.save(os.path.join(splits_folder, "val.pickle"))
            self.test_split.save(os.path.join(splits_folder, "test.pickle"))
            self.logger.info(msg=f"Splits saved to {splits_folder}.")

            return split_time

        self.step = PIPELINE_STEPS["dataset"]
        self.logger = dataset_logger

        dataset, dataset_load_time = _load_dataset_from_config(
            config=self.eval_config["dataset"]
        )
        _log_dataset_info(dataset.info())
        dataset_split_time = _get_splits_from_eval_config(eval_config=self.eval_config)

        return dataset_load_time, dataset_split_time

    @execution_time(logger=training_logger)
    def _get_model(self) -> None:
        def _construct_filename(model_config: dict, eval_config: dict) -> str:
            """Construct path to model checkpoint based on model_config and eval_config."""

            def _name_from_config(model_specific_config: dict) -> str:
                name = ""
                for k, v in model_specific_config.items():
                    if isinstance(v, dict):
                        for k2, v2 in v.items():
                            name += f"{k2}_{v2}_"
                    else:
                        name += f"{k}_{v}_"
                name = name[:-1] + ".pickle"
                return name

            checkpoint_file = _name_from_config(model_config["model_specific"])
            filename = os.path.join(
                model_config["general"]["checkpoints_folder"],
                eval_config["dataset"]["name"],
                model_config["general"]["model"],
                checkpoint_file,
            )
            return filename

        def _create_model() -> None:
            """Initialize model based on model_config, train it and save it to path specified in model_config."""
            if self.model_config["general"]["model"] == "EASE":
                self.model = EASE.from_config(self.model_config["model_specific"])
            elif self.model_config["general"]["model"] == "SANSA":
                self.model = SANSA.from_config(self.model_config["model_specific"])
            elif self.model_config["general"]["model"] == "MRF":
                self.model = MRF.from_config(self.model_config["model_specific"])
            else:
                raise ValueError(
                    f"Invalid model name: {self.model_config['general']['model']}, must be one of EASE, SANSA, MRF"
                )
            self.model.train(self.train_split)
            if self.model_config["general"]["save"]:
                self.model.save(filename)

        def _load_model() -> None:
            """Load model from path specified in model_config."""
            self.logger.info(msg=f"Loading model from {filename}...")
            if self.model_config["general"]["model"] == "EASE":
                self.model = EASE.load(filename)
            elif self.model_config["general"]["model"] == "SANSA":
                self.model = SANSA.load(filename)
            elif self.model_config["general"]["model"] == "MRF":
                self.model = MRF.load(filename)
            else:
                raise ValueError(
                    f"Invalid model name: {self.model_config['general']['model']}, must be one of EASE, SANSA, MRF"
                )

        def _log_model_info() -> None:
            msg = (
                f"Model: {self.model_config['general']['model']}, "
                + f"number of weights: {self.model.get_num_weights()}, "
                + f"weights size: {self.model.get_weights_size() / (1024*1024):.3f} MB"
            )
            self.logger.info(msg=msg)

        self.step = PIPELINE_STEPS["training"]
        self.logger = training_logger

        filename = _construct_filename(
            model_config=self.model_config, eval_config=self.eval_config
        )
        if os.path.exists(filename) and not self.model_config["general"]["retrain"]:
            _load_model()
        else:
            _create_model()
        _log_model_info()

    @execution_time(logger=evaluation_logger)
    def _evaluate_model(self) -> dict:
        def _get_performance(model: Model) -> dict:
            performance_dict = {"time": {}, "memory": {}}
            for k, v in model.stats_trace.items():
                if "time" in k:
                    performance_dict["time"][k] = v
                else:
                    performance_dict["memory"][k] = v
            return performance_dict

        def _get_experiment_data(results: dict) -> dict:
            data = {
                "dataset": self.eval_config["dataset"],
                "split": self.eval_config["split"],
                "model": self.model_config,
                "evaluation": self.eval_config["evaluation"],
                "performance": _get_performance(self.model),
                "results": results,
            }
            return data

        self.step = PIPELINE_STEPS["evaluation"]
        self.logger = evaluation_logger

        if self.eval_config["evaluation"]["split"] == "train":
            evaluation_split = self.train_split
        elif self.eval_config["evaluation"]["split"] == "val":
            evaluation_split = self.val_split
        elif self.eval_config["evaluation"]["split"] == "test":
            evaluation_split = self.test_split
        else:
            raise ValueError(
                "Evaluation split must be either 'train', 'val' or 'test'."
            )

        results = evaluate(
            model=self.model,
            split=evaluation_split,
            metrics=self.eval_config["evaluation"]["metrics"],
            ks=[int(k) for k in self.eval_config["evaluation"]["ks"]],
            batch_size=int(self.eval_config["evaluation"].get("batch_size", 2000)),
        )

        data = _get_experiment_data(results)

        return data

    def _save_experiment_data(self, data: dict) -> None:
        filename = os.path.join(
            self.eval_config["evaluation"]["experiment_folder"],
            self.eval_config["dataset"]["name"],
            "results",
            f"{self.model_config['general']['model']}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
        )
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    @execution_time(logger=end_logger)
    def run(self) -> dict:
        self.logger.info(msg="Starting evaluation pipeline.")
        (
            dataset_load_time,
            dataset_split_time,
        ), data_preparation_time = self._create_dataset_splits()
        _, model_preparation_time = self._get_model()
        experiment_data, evaluation_time = self._evaluate_model()

        experiment_data["performance"]["time"]["pipeline"] = {
            "dataset_load_time": dataset_load_time,
            "dataset_split_time": dataset_split_time,
            "data_preparation_time": data_preparation_time,
            "model_preparation_time": model_preparation_time,
            "evaluation_time": evaluation_time,
        }

        self._save_experiment_data(experiment_data)

        self.step = PIPELINE_STEPS["end"]

        return experiment_data
