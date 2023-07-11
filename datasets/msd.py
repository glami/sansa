"""
Million Song Dataset (MSD).
"""

import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets.dataset import Dataset


class MSD(Dataset):
    def _read_raw_data(self) -> None:
        """
        Loads raw data and stores it in self.dataset.
        """
        raw_data_file = f"{self.path}/train_triplets.txt"
        if not os.path.isfile(raw_data_file):
            raise ValueError(f"Raw data file {raw_data_file} not found.")
        with open(raw_data_file) as f:
            data = f.read()
        data = data.replace("\t", ",")
        data = data.replace("\n", ",")
        data = data.split(",")
        data = [d for d in data if d != ""]

        l = len(data)
        users_all = [data[i] for i in range(0, l, 3)]
        items_all = [data[i] for i in range(1, l, 3)]
        ratings_all = [data[i] for i in range(2, l, 3)]
        timestamps_all = [0 for _ in range(0, l, 3)]

        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        users_all = user_encoder.fit_transform(users_all)
        items_all = item_encoder.fit_transform(items_all)

        self.dataset = pd.DataFrame(
            {
                "user_id": users_all,
                "item_id": items_all,
                "feedback": ratings_all,
                "timestamp": timestamps_all,
            }
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
        # Binarize feedback
        self.dataset.feedback = 1
        # Change feedback to float64
        self.dataset.feedback = self.dataset.feedback.astype(np.float64)
        # Only keep songs with at least 200 listeners
        self.dataset = self.dataset.groupby("item_id").filter(lambda x: len(x) >= 200)
        # Only keep users with at least 20 songs
        self.dataset = self.dataset.groupby("user_id").filter(lambda x: len(x) >= 20)
        # Set self.processed to True to indicate that the dataset was successfully preprocessed
        self.processed = True
