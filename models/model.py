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
