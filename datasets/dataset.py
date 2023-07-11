"""
Abstract class for datasets.
"""

from abc import ABC, abstractmethod
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from datasets.split import DatasetSplit, horizontal_split
from evaluation.logs import dataset_logger
from evaluation.metrics import execution_time


class Dataset(ABC):
    def __init__(
        self,
        name: str,
        folder: str = "./data",
        processed_file: str = "ratings.parquet",
    ):
        self.name = name
        self.folder = folder
        self.processed_file = processed_file

        self.path = os.path.join(self.folder, self.name)
        self.dataset: pd.DataFrame = None
        self.processed: bool = False

    def _read_raw_data(self) -> None:
        """
        Loads raw data and stores it in self.dataset.
        """
        ...

    @abstractmethod
    def _unify_dataset(self) -> None:
        """
        Unifies the dataset to a common format.
        A common format is a dataframe with the following columns:
        - user_id: int64
        - item_id: int64
        - feedback: float64
        - timestamp: datetime64
        """
        ...

    @abstractmethod
    def _preprocess_data(self) -> None:
        """
        Preprocesses the dataset according to the experiment requirements.
        Sets self.preprocessed = True if the dataset was successfully preprocessed.
        """
        ...

    def save_processed(self) -> None:
        self.dataset.to_parquet(f"{self.path}/{self.processed_file}", index=False)

    def load_processed(self) -> None:
        """
        Load a processed dataset from a parquet file.
        """
        dataset_logger.info(
            f"Loading processed dataset {self.path}/{self.processed_file}."
        )
        self.dataset = pd.read_parquet(f"{self.path}/{self.processed_file}")
        self.processed = True

    def preprocess_and_save(self) -> None:
        dataset_logger.info(f"Creating new dataset {self.name}:")
        dataset_logger.info(f"Loading raw dataset files from {self.path}/ ...")
        self._read_raw_data()
        dataset_logger.info("Unifying dataset format...")
        self._unify_dataset()
        dataset_logger.info("Preprocessing dataset...")
        self._preprocess_data()
        dataset_logger.info(
            f"Saving processed dataset {self.path}/{self.processed_file}..."
        )
        self.save_processed()

    @classmethod
    def from_config(cls, config: dict) -> "Dataset":
        """Instantiate a dataset according to the config."""
        # Get config parameters.
        dataset_name = config["name"]  # must be specified
        folder = config.get("folder", "datasets/data")

        # Instantiate dataset.
        processed_file = "dataset.parquet"
        dataset = cls(
            name=dataset_name,
            folder=folder,
            processed_file=processed_file,
        )

        # Check if processed dataset is available and rewrite is not requested.
        f = os.path.join(dataset.path, dataset.processed_file)
        if os.path.isfile(f) and not config.get("rewrite", False):
            dataset.load_processed()
            return dataset

        # Preprocess raw data and save dataset.
        dataset.preprocess_and_save()
        return dataset

    @execution_time(logger=dataset_logger)
    def create_splits(
        self, config: dict
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Creates train, validation and test split objects from the dataset.
        Split is done horizontally, i.e. users are split into train, validation and test users.
        Targets are extracted from validation and test users, train users only if needed (e.g. EASE-like models do not need train targets).
        """
        # To continue, the dataset must be processed.
        if not self.processed:
            raise ValueError("Dataset not processed yet.")

        # Get config parameters.
        n_val_users = config.get("n_val_users", 10000)
        n_test_users = config.get("n_test_users", 10000)
        test_target_proportion = config["target_proportion"]  # must be specified
        val_target_proportion = config.get(
            "val_target_proportion", test_target_proportion
        )
        train_target_proportion = config.get(
            "train_target_proportion", test_target_proportion
        )
        newest = config.get("targets_newest", False)
        seed = config.get("seed", 42)
        extract_train_targets = config.get("extract_train_targets", False)

        # Fit item encoder and transform item IDs.
        item_encoder = LabelEncoder().fit(self.dataset["item_id"])
        self.dataset["item_id"] = item_encoder.transform(self.dataset["item_id"])

        # Shuffle dataset.
        self.dataset = self.dataset.sample(frac=1, random_state=seed)

        # Split dataset.
        train_df, val_df, test_df = horizontal_split(
            self.dataset, n_val_users, n_test_users, seed
        )

        # Log split information.
        dataset_logger.info(
            f"Dataframe lengths | train_df: {len(train_df)}, val_df: {len(val_df)}, test_df: {len(test_df)}"
        )

        # Create split objects.
        train_split = DatasetSplit(
            train_df,
            item_encoder,
            extract_targets=extract_train_targets,
            target_proportion=train_target_proportion,  # only used if extract_targets
            newest=newest,  # only used if extract_targets
        )
        val_split = DatasetSplit(
            val_df,
            item_encoder,
            extract_targets=True,
            target_proportion=val_target_proportion,
            newest=newest,
        )
        test_split = DatasetSplit(
            test_df,
            item_encoder,
            extract_targets=True,
            target_proportion=test_target_proportion,
            newest=newest,
        )

        # Log dataset information.
        dataset_logger.info(msg="Splits information:")
        dataset_logger.info(msg=f"Train split info | {train_split.info()}")
        dataset_logger.info(msg=f"Validation split info | {val_split.info()}")
        dataset_logger.info(msg=f"Test split info | {test_split.info()}")

        return train_split, val_split, test_split

    def info(self) -> dict | None:
        """
        Returns dict of information about the dataset.
        """
        dataset_name_str = f"Dataset: {self.name}\n"
        if self.dataset is None:
            dataset_logger.warning(dataset_name_str + "Not loaded yet.")
            return None
        num_users = self.dataset.user_id.nunique()
        num_items = self.dataset.item_id.nunique()

        info_dict = {
            "Dataset name": self.name,
            "Number of users": num_users,
            "Number of items": num_items,
            "Number of interactions": len(self.dataset),
            "Interaction density": f"{len(self.dataset) / (num_users * num_items):.4%}",
        }

        return info_dict
