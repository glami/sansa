"""
Movielens20 dataset.
"""

import numpy as np
import os
import pandas as pd

from datasets.dataset import Dataset


class Movielens(Dataset):
    def _read_raw_data(self) -> None:
        """
        Loads raw data and stores it in self.dataset.
        """
        raw_data_file = f"{self.path}/rating.csv"
        if not os.path.isfile(raw_data_file):
            raise ValueError(f"Raw data file {raw_data_file} not found.")
        self.dataset = pd.read_csv(
            raw_data_file,
            delimiter=",",
            engine="python",
        )

    def _unify_dataset(self) -> None:
        """
        Unifies the dataset to a common format.
        A common format is a dataframe with the following columns:
        - user_id: int64
        - item_id: int64
        - feedback: float64
        - timestamp: datetime64
        """
        # Rename columns
        self.dataset.rename(
            columns={"userId": "user_id", "movieId": "item_id", "rating": "feedback"},
            inplace=True,
        )
        # Change user_id and item_id to int64
        self.dataset.user_id = self.dataset.user_id.astype(np.int64)
        self.dataset.item_id = self.dataset.item_id.astype(np.int64)
        # Change feedback to float64
        self.dataset.feedback = self.dataset.feedback.astype(np.float64)
        # Change timestamp to datetime
        self.dataset.timestamp = pd.to_datetime(self.dataset.timestamp)

    def _preprocess_data(self) -> None:
        """
        Preprocesses the dataset according to the experiment requirements.
        Returns True if the dataset was successfully preprocessed.
        """
        # Only keep logs with feedback >= 4
        self.dataset = self.dataset[self.dataset.feedback >= 4]
        # Binarize feedback
        self.dataset.feedback = 1
        # Change feedback to float64
        self.dataset.feedback = self.dataset.feedback.astype(np.float64)
        # Only keep users with at least 5 interactions
        self.dataset = self.dataset.groupby("user_id").filter(lambda x: len(x) >= 5)
        # Set self.processed to True to indicate that the dataset was successfully preprocessed
        self.processed = True
