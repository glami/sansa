"""
Amazonbook dataset.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets.dataset import Dataset

from datasets.split import (
    DatasetSplit,
    FixedVerticalDatasetSplit,
    fixed_vertical_split,
)
from evaluation.logs import dataset_logger
from evaluation.metrics import execution_time


class Amazonbook(Dataset):
    def _read_raw_data(self) -> None:
        """
        Loads raw data and stores it in self.dataset.
        """
        train_file = f"{self.path}/train.txt"
        test_file = f"{self.path}/test.txt"

        # Selected train interactions
        users_train = []
        items_train = []
        target_train = []
        if not os.path.isfile(train_file):
            raise ValueError(f"Train file {train_file} not found.")
        with open(train_file) as f:
            lines = f.read().splitlines()
        for line in lines:
            entries = line.split()
            user = entries[0]
            items = entries[1:]
            users_train += [user] * len(items)
            items_train += items
            target_train += [False] * len(items)

        # Selected targets
        users_test = []
        items_test = []
        target_test = []
        if not os.path.isfile(test_file):
            raise ValueError(f"Test file {test_file} not found.")
        with open(test_file) as f:
            lines = f.read().splitlines()
        for line in lines:
            entries = line.split()
            user = entries[0]
            items = entries[1:]
            users_test += [user] * len(items)
            items_test += items
            target_test += [True] * len(items)

        train_dataset = pd.DataFrame(
            {
                "user_id": users_train,
                "item_id": items_train,
                "feedback": [1] * len(items_train),
                "timestamp": [0] * len(items_train),
                "target": target_train,
            }
        )
        test_dataset = pd.DataFrame(
            {
                "user_id": users_test,
                "item_id": items_test,
                "feedback": [1] * len(items_test),
                "timestamp": [0] * len(items_test),
                "target": target_test,
            }
        )

        self.dataset = pd.concat([train_dataset, test_dataset], ignore_index=True)

    def _unify_dataset(self) -> None:
        """
        Unifies the dataset to a common format.
        A common format is a dataframe with the following columns:
        - user_id: int64
        - item_id: int64
        - feedback: float64
        - timestamp: datetime64
        - target: bool      # specific for datasets with fixed targets
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
        # Set self.processed to True to indicate that the dataset was successfully preprocessed
        self.processed = True

    @execution_time(logger=dataset_logger)
    def create_splits(
        self, config: dict
    ) -> tuple[DatasetSplit, DatasetSplit, FixedVerticalDatasetSplit]:
        """
        Creates train, validation and test split objects from the dataset.
        Split is done VERTICALLY, using pre-defined interactions for test.
        """
        # To continue, the dataset must be processed.
        if not self.processed:
            raise ValueError("Dataset not processed yet.")

        # Get config parameters (all are optional).
        train_target_proportion = config.get("train_target_proportion", 0.1)
        val_target_proportion = config.get("val_target_proportion", 0.1)
        seed = config.get("seed", 42)
        extract_train_targets = config.get("extract_train_targets", False)

        # Fit item encoder and transform item IDs.
        item_encoder = LabelEncoder().fit(self.dataset["item_id"])
        self.dataset["item_id"] = item_encoder.transform(self.dataset["item_id"])

        # Shuffle dataset.
        self.dataset = self.dataset.sample(frac=1, random_state=seed)

        # Split dataset.
        train_df, val_df, test_df = fixed_vertical_split(
            self.dataset, val_target_proportion, seed
        )

        # Log dataset information.
        dataset_logger.info(
            f"Dataframe lengths | train_df: {len(train_df)}, val_df: {len(val_df)}, test_df: {len(test_df)}"
        )

        # Create split objects (test split is fixed)
        train_split = DatasetSplit(
            train_df,
            item_encoder,
            extract_targets=extract_train_targets,
            target_proportion=train_target_proportion,  # only used if extract_targets
            newest=False,
        )
        val_split = DatasetSplit(
            val_df,
            item_encoder,
            extract_targets=True,
            target_proportion=val_target_proportion,
            newest=False,
        )
        test_split = FixedVerticalDatasetSplit(
            test_df,
            item_encoder,
        )

        # Log dataset information.
        dataset_logger.info(msg="Splits information:")
        dataset_logger.info(msg=f"Train split info | {train_split.info()}")
        dataset_logger.info(msg=f"Validation split info | {val_split.info()}")
        dataset_logger.info(msg=f"Test split info | {test_split.info()}")

        return train_split, val_split, test_split
