"""
Utility functions for splitting datasets into train, validation, and test sets.
"""

import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder

from evaluation.logs import dataset_logger


def df_to_csr(df: pd.DataFrame, shape: tuple[int, int]) -> sp.csr_matrix:
    """Returns a sparse matrix from a dataframe."""
    return sp.coo_matrix(
        (df.feedback, (df.user_id, df.item_id)),
        shape=shape,
        dtype=np.float64,
    ).tocsr()


def extract_targets_from_split(
    df: pd.DataFrame,
    extract_targets: bool,
    target_proportion: float = 0.2,
    newest: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts targets from a (horizontal) split of a dataframe."""
    if not extract_targets:
        return df, None
    tmp_df = df.copy()
    if newest:
        tmp_df.sort_values(by=["timestamp"], ascending=[False], inplace=True)
    user_rating_counts = tmp_df["user_id"].value_counts()
    target_sizes = np.floor(target_proportion * user_rating_counts).astype(int)
    user_id_groups = tmp_df.groupby("user_id", group_keys=True)
    inputs_df = user_id_groups.apply(lambda x: x[target_sizes[x.name] :]).reset_index(
        drop=True
    )
    target_df = user_id_groups.apply(lambda x: x[: target_sizes[x.name]]).reset_index(
        drop=True
    )
    return inputs_df, target_df


def extract_marked_targets_from_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extracts marked targets from a split of a dataframe."""
    tmp_df = df.copy()
    inputs_df = tmp_df[~tmp_df["target"]]
    target_df = tmp_df[tmp_df["target"]]
    return inputs_df, target_df


def horizontal_split(
    df: pd.DataFrame,
    n_val_users: int,
    n_test_users: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe horizontally (=disjoint user sets) into train, validation, and test sets."""
    np.random.seed(seed)
    unique_users = df["user_id"].unique()
    np.random.shuffle(unique_users)
    test_users = unique_users[:n_test_users]
    val_users = unique_users[n_test_users : n_test_users + n_val_users]
    train_users = unique_users[n_test_users + n_val_users :]
    train_df = df[df["user_id"].isin(train_users)]
    val_df = df[df["user_id"].isin(val_users)]
    test_df = df[df["user_id"].isin(test_users)]
    return train_df, val_df, test_df


def fixed_vertical_split(
    df: pd.DataFrame,
    val_target_proportion: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataframe according to predefined train and test sets. Creates train, validation, and test dataframes."""
    np.random.seed(seed)
    # df has column "target", used to identify test targets
    val_inputs_df, val_targets_df = extract_targets_from_split(
        df=df[~df["target"]],
        extract_targets=True,
        target_proportion=val_target_proportion,
        newest=False,
    )
    val_inputs_df.drop(columns=["target"], inplace=True)
    val_targets_df.drop(columns=["target"], inplace=True)
    val_df = pd.concat([val_inputs_df, val_targets_df])
    train_df = val_inputs_df
    return train_df, val_df, df


class DatasetSplit:
    """Class for storing and manipulating implicit feedback data."""

    def __init__(
        self,
        df: pd.DataFrame,
        item_encoder: LabelEncoder,
        extract_targets: bool = False,
        target_proportion: float = 0.2,
        newest: bool = True,
    ) -> None:
        # Fit user encoder and transform user IDs.
        pd.options.mode.chained_assignment = None  # suppress irrelevant warning
        self.user_encoder = LabelEncoder()
        df.user_id = self.user_encoder.fit_transform(df.user_id.values)
        pd.options.mode.chained_assignment = "warn"  # reset to default

        self.item_encoder = item_encoder
        self.data, self.targets = extract_targets_from_split(
            df=df,
            extract_targets=extract_targets,
            target_proportion=target_proportion,
            newest=newest,
        )
        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        self.n_ratings = self.data.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the dataset."""
        return (self.n_users, self.n_items)

    @property
    def sparsity(self) -> float:
        """Return the sparsity of the dataset."""
        if (self.n_users == 0) or (self.n_items == 0):
            return 0
        return 1 - self.n_ratings / (self.n_users * self.n_items)

    @property
    def density(self) -> float:
        """Return the density of the dataset."""
        return 1 - self.sparsity

    def get_csr_matrix(self) -> sp.csr_matrix:
        df = self.data
        return df_to_csr(df=df, shape=(self.n_users, self.n_items))

    def get_csc_t_matrix(self) -> sp.csr_matrix:
        return self.get_csr_matrix().T

    def get_rated_items(self, user_ids: list[int]) -> pd.DataFrame:
        user_ids = self.user_encoder.transform(user_ids)
        rated_items_df = self.data[self.data.user_id.isin(user_ids)]

        return rated_items_df

    def get_target_items(self, user_ids: list[int]) -> pd.DataFrame:
        user_ids = self.user_encoder.transform(user_ids)
        targets_df = self.targets[self.targets.user_id.isin(user_ids)]

        return targets_df

    def info(self) -> str:
        return f"n_users = {self.n_users}, n_items = {self.n_items}, n_ratings = {self.n_ratings}, sparsity = {self.sparsity:.2%}"

    def item_item_info(self) -> str:
        return f"shape = ({self.n_items},{self.n_items})"

    def save(self, path: str) -> None:
        """Saves dataset split to pickle"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DatasetSplit":
        """Loads dataset split from pickle"""
        with open(path, "rb") as f:
            return pickle.load(f)


class FixedVerticalDatasetSplit(DatasetSplit):
    """
    Class preserving fixed splits, for storing and manipulating implicit feedback data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        item_encoder: LabelEncoder,
    ) -> None:
        self.item_encoder = item_encoder
        # df has column "target" with 1 if item is target, 0 otherwise
        self.data, self.targets = extract_marked_targets_from_split(df=df)

        # Find users with empty targets and remove them from data.
        users_with_empty_targets = list(
            set(self.data.user_id) - set(self.targets.user_id)
        )
        dataset_logger.info(
            f"Removing users {users_with_empty_targets} from test inputs."
        )
        self.data = self.data[self.data.user_id.isin(self.targets.user_id)]
        # Fit user encoder and transform user IDs.
        pd.options.mode.chained_assignment = None  # suppress irrelevant warning
        self.user_encoder = LabelEncoder()
        self.data.user_id = self.user_encoder.fit_transform(self.data.user_id.values)
        self.targets.user_id = self.user_encoder.transform(self.targets.user_id.values)
        pd.options.mode.chained_assignment = "warn"  # reset to default

        self.data.drop(columns=["target"], inplace=True)
        self.targets.drop(columns=["target"], inplace=True)

        self.n_users = len(self.user_encoder.classes_)
        self.n_items = len(self.item_encoder.classes_)
        self.n_ratings = self.data.shape[0]
