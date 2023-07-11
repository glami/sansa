#
# Copyright 2023 Inspigroup s.r.o.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# https://github.com/glami/sansa/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
"""
Abstract base class for models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pickle

from datasets.split import DatasetSplit


class Model(ABC):
    @abstractmethod
    def train(self, train_split: DatasetSplit, val_split=None) -> None:
        """
        Fit the model to the train data.
        Optionally evaluate the model on the validation data.

        Args:
            train: The training set.
            val: The validation set.
        """
        ...

    @abstractmethod
    def recommend(self, inputs, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Given input data of several users, recommend the top k items.

        Args:
            inputs: The input data.
            k: The number of items to recommend.
        """
        ...

    def save(self, path: str) -> None:
        """
        Save the model to the given path.

        Args:
            path: Path to save the model to.
        """
        pickle.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "Model":
        """
        Load the model from the given path.

        Args:
            path (str): Path to load the model from.
        """
        model = pickle.load(open(path, "rb"))
        return model

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "Model":
        """
        Create a model from a config.

        Args:
            config: The config.
        """
        ...

    @abstractmethod
    def get_num_weights(self) -> int:
        """Return number of weights in model."""
        ...

    @abstractmethod
    def get_weights_size(self) -> int:
        """Return size of weights in bytes."""
        ...
