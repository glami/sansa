"""
Embarrassingly Shallow Autoencoder (EASE)

https://dl.acm.org/doi/10.1145/3308558.3313710
"""


import numpy as np
import pandas as pd
import scipy.sparse as sp

from datasets.split import DatasetSplit, df_to_csr
from evaluation.logs import training_logger, evaluation_logger
from evaluation.metrics import execution_time
from models.model import Model
from sparseinv.matmat import _matmat


class EASE(Model):
    """
    EASE Recommender System

    Paper:
    Embarrassingly Shallow Autoencoders for Sparse Data
    Harald Steck
    arXiv:1905.03375v1
    """

    def __init__(self, l2: float) -> None:
        """
        Initialize EASE model.

        Args:
            l2: L2 regularization parameter.
        """
        super().__init__()
        self.l2 = l2
        self.n_items = None
        self.weights = None  # List of matrices
        self.stats_trace = dict()

    def set_params(self, **kwargs) -> "EASE":
        """Set model parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @execution_time(logger=training_logger)
    def _construct_weights(self, X: sp.spmatrix) -> list[np.ndarray]:
        """Construct weights for dense EASE."""
        # 1. Construct item-item matrix G
        training_logger.info("Creating item-item matrix...")
        G = _matmat(X.T, X)
        training_logger.info(
            f"nnz of G: {G.nnz}, density: {G.nnz / (G.shape[0] ** 2)}, size: {(G.data.nbytes + G.indices.nbytes + G.indptr.nbytes) / 1e6:.3f} MB"
        )
        # 2. Transform to dense matrix
        training_logger.info("Converting to dense matrix...")
        G = G.toarray()
        # 3. Add regularization
        di = np.diag_indices(self.n_items)
        G[di] += self.l2
        # 4. Invert G
        training_logger.info("Inverting matrix...")
        P = np.linalg.inv(G)
        # 5. Divide columns of the inverse by negative diagonal entries
        training_logger.info("Constructing weight matrix...")
        B = P / (-np.diag(P))
        # 6. Set diagonal entries to zero
        B[di] = 0
        # 7. Return list of weight matrices
        training_logger.info("Done.")
        return [B]

    def train(self, train_split: DatasetSplit) -> None:
        """Fit the model to the train_split."""
        training_logger.info(f"Training EASE with L2={self.l2}...")
        training_logger.info(f"Matrix size: {train_split.item_item_info()}")
        # 1. Prepare user-item matrix
        X = train_split.get_csr_matrix()
        # 2. Store number of items
        self.n_items = train_split.n_items
        # 3. Compute weights
        training_logger.info("Constructing weights:")
        self.weights, construct_weights_time = self._construct_weights(X)
        # we report the time it took to construct the weights as training time, as we want to ignore data loading
        self.stats_trace["construct_weights_time"] = construct_weights_time
        training_logger.debug("Training done.")

    @execution_time(logger=evaluation_logger)
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings for a batch of users."""
        # Run prediction to obtain Predicted Score Matrix
        P = X @ self.weights[0]
        # Mask out items already seen/rated by the user by setting their scores to -inf
        P[X.nonzero()] = -np.inf
        return P

    def recommend(
        self, inputs_df: pd.DataFrame, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recommend for a batch of users.
        Dense prediction, because calculated predictions are dense.
        """
        # Create sparse user-item matrix of feedbacks
        n_users = inputs_df.user_id.nunique()
        X = df_to_csr(df=inputs_df, shape=(n_users, self.n_items)).toarray()

        # Run prediction to obtain Predicted Score Matrix
        evaluation_logger.debug(f"Predicting scores (n_users = {n_users}):")
        P, predict_time = self._predict(X)
        self.stats_trace["predict_time"] = predict_time

        evaluation_logger.debug("Filtering, sorting...")

        # Get indices of top k items for each user
        evaluation_logger.debug("Filtering top k items...")
        top_k_ids = np.argpartition(-P, k, axis=1)[:, :k]
        # Get corresponding scores
        top_k_scores = P[np.arange(n_users)[:, np.newaxis], top_k_ids]

        evaluation_logger.debug("Sorting top k items...")
        # Create sorting array to sort top_k_idx_mat and top_k_scores_mat in descending order of scores
        sorting = np.argsort(-top_k_scores, axis=1)
        # sort top_k_idx_mat and top_k_scores_mat
        top_k_ids = top_k_ids[np.arange(n_users)[:, np.newaxis], sorting]
        top_k_scores = top_k_scores[np.arange(n_users)[:, np.newaxis], sorting]

        return top_k_ids, top_k_scores

    @classmethod
    def from_config(cls, config: dict) -> "EASE":
        model = cls(l2=config["l2"])
        return model

    def get_num_weights(self) -> int:
        """Return number of weights in model."""
        return self.weights[0].size

    def get_weights_size(self) -> int:
        """Return size of weights in bytes."""
        return self.weights[0].nbytes
