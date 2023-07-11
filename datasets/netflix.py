"""
Netflix dataset.
"""

import numpy as np
import os
import pandas as pd

from datasets.dataset import Dataset


class Netflix(Dataset):
    def _read_raw_data(self) -> None:
        """
        Loads raw data and stores it in self.dataset.
        """
        filename_pattern = "combined_data_#.txt"
        data = ""
        for i in range(1, 5):
            filename = f"{self.path}/{filename_pattern}".replace("#", str(i))
            if not os.path.isfile(filename):
                raise ValueError(f"Raw data file {filename} not found.")
            with open(filename) as f:
                data += f.read()
                data += ","
        data = data.replace("\n", ",")
        data = data.split(",")
        data = [d for d in data if d != ""]

        item_id_mask = [d.endswith(":") for d in data]
        item_id_idx = np.where(item_id_mask)[0]

        item_ids = [data[i][:-1] for i in item_id_idx]

        item_data = [
            data[i + 1 : j] if j is not None else data[i + 1 :]
            for i, j in zip(item_id_idx, np.concatenate((item_id_idx[1:], [None])))
        ]

        items_all = []
        users_all = []
        ratings_all = []
        timestamps_all = []

        for id, d in zip(item_ids, item_data):
            l = len(d)
            items_all.append([id for _ in range(0, l, 3)])
            users_all.append([d[i] for i in range(0, l, 3)])
            ratings_all.append([d[i] for i in range(1, l, 3)])
            timestamps_all.append([d[i] for i in range(2, l, 3)])

        self.dataset = pd.DataFrame(
            {
                "user_id": np.concatenate(users_all),
                "item_id": np.concatenate(items_all),
                "feedback": np.concatenate(ratings_all),
                "timestamp": np.concatenate(timestamps_all),
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
