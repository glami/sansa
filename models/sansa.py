"""
Scalable Approximate NonSymmetric Autoencoder (SANSA) model.

Our submission to RecSys 2023 conference.
"""

import gc
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.utils.sparsefuncs as spfuncs
import warnings

from datasets.split import DatasetSplit, df_to_csr
from evaluation.logs import training_logger, evaluation_logger
from evaluation.metrics import execution_time
from models.ease import EASE
from sparseinv.ainv import ainv_L
from sparseinv.ldlt import ldlt
from sparseinv.matmat import _matmat as _m


@execution_time(logger=evaluation_logger)
def _matmat(A: sp.spmatrix, B: sp.spmatrix) -> sp.spmatrix:
    """Execution time wrapper for sparseinv.matmat._matmat during evaluation."""
    return _m(A, B)


def row_top_k(data, indices, indptr, k):
    m = indptr.shape[0] - 1
    top_k_ids = np.zeros((m, k), dtype=indices.dtype)
    top_k_scores = np.zeros((m, k), dtype=data.dtype)
    for i in nb.prange(m):
        row_entries = data[indptr[i] : indptr[i + 1]]
        # "Safety feature": the following line will crash if k > len(row_entries): kth out of bounds. Fix if it crashes during the experiments.
        ids = np.argpartition(row_entries, -k)[-k:]
        top_k_ids[i] = indices[indptr[i] : indptr[i + 1]][ids]
        top_k_scores[i] = data[indptr[i] : indptr[i + 1]][ids]
    return top_k_ids, top_k_scores


class SANSA(EASE):
    """
    Scalable Approximate NonSymmetric Autoencoder
    """

    def __init__(
        self,
        l2: float,
        target_density: float,
        ainv_method: str,
        ainv_params: dict,
        ldlt_method: str,
        ldlt_params: dict,
    ) -> None:
        """
        Initialize SANSA model.

        Args:
            l2: L2 regularization parameter.
        """
        super().__init__(l2=l2)
        self.target_density = target_density
        if ainv_method not in ["exact", "s1", "umr"]:
            raise ValueError(
                f"Invalid ainv_method {ainv_method}, must be one of 'exact', 's1', 'umr'."
            )
        self.ainv_method = ainv_method
        self.ainv_params = ainv_params
        if ldlt_method not in ["cholmod", "icf"]:
            raise ValueError(
                f"Invalid ldlt_method {ldlt_method}, must be one of 'cholmod', 'icf'."
            )
        self.ldlt_method = ldlt_method
        self.ldlt_params = ldlt_params
        self.weights = None  # List of matrices
        self.stats_trace = dict()

    @execution_time(logger=training_logger)
    def _construct_weights(self, X_T: sp.csc_matrix) -> list[sp.spmatrix]:
        """Construct weights for SANSA."""
        # 1. Compute LDL^T decomposition of P(X^TX + self.l2 * I)P^T without constructing it explicitly
        (L, D, p, memory_stats), ldlt_time = ldlt(
            X_T,
            l2=self.l2,
            target_density=self.target_density,
            method=self.ldlt_method,
            method_params=self.ldlt_params,
        )
        del X_T

        self.stats_trace["ldlt_time"] = ldlt_time
        for k, v in memory_stats.items():
            self.stats_trace[f"{k}_memory"] = v
        training_logger.info(
            f"nnz of L: {L.nnz}, size: {(L.data.nbytes + L.indices.nbytes + L.indptr.nbytes) / 1e6:.3f} MB"
        )

        # 2. Compute approximate inverse of L using selected method
        training_logger.info("Computing approximate inverse of L:")

        (L_inv, umr_scans_times, umr_finetune_step_times), ainv_time = ainv_L(
            L,
            target_density=self.target_density,
            method=self.ainv_method,
            method_params=self.ainv_params,
        )  # this returns a pruned matrix
        # Garbage collect L
        del L

        L_inv_nnz = L_inv.nnz
        L_inv_mbytes = (
            L_inv.data.nbytes + L_inv.indices.nbytes + L_inv.indptr.nbytes
        ) / 1e6
        L_inv_density = L_inv_nnz / (L_inv.shape[0] * L_inv.shape[1])
        self.stats_trace["L_inv_memory"] = {
            "nnz": L_inv_nnz,
            "mbytes": L_inv_mbytes,
            "density": L_inv_density,
        }
        self.stats_trace["umr_scans_times"] = umr_scans_times
        self.stats_trace["umr_finetune_step_times"] = umr_finetune_step_times
        self.stats_trace["ainv_time"] = ainv_time
        training_logger.info(f"nnz of L_inv: {L_inv_nnz}, size: {L_inv_mbytes:.3f} MB")

        # 3. Construct W = L_inv @ P
        training_logger.info("Constructing W = L_inv @ P...")
        inv_p = np.argsort(p)
        W = L_inv[:, inv_p]
        # Garbage collect L_inv
        del L_inv

        # 4. Construct W_r (A^{-1} = W.T @ W_r)
        W_r = W.copy()
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_row_scale(W_r, 1 / D.diagonal())

        # 5. Extract diagonal entries
        training_logger.info("Extracting diagonal of W.T @ D_inv @ W...")
        diag = W.copy()
        diag.data = diag.data**2
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_row_scale(diag, 1 / D.diagonal())
        diagsum = diag.sum(axis=0)  # original
        del diag
        diag = np.asarray(diagsum)[0]

        # 6. Divide columns of the inverse by negative diagonal entries
        training_logger.info("Dividing columns of W by diagonal entries...")
        # Due to associativity of matrix multiplication, this is equivalent to dividing the columns of W by negative diagonal entries
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely caused by endianness
            spfuncs.inplace_column_scale(W_r, -1 / diag)

        # 7. Set diagonal entries to zero (B[di] = 0)
        # --> trick during prediction

        gc.collect()

        # Return list of weight matrices [W.T, W_r]
        return [W.T.tocsr(), W_r.tocsr()]

    def train(self, train_split: DatasetSplit) -> None:
        """Fit the model to the train_split."""
        training_logger.info(f"Train user-item matrix info | {train_split.info()}")
        training_logger.info(f"Item-item matrix info | {train_split.item_item_info()}")
        training_logger.info(
            f"Training SANSA with L2={self.l2}, target density={self.target_density:%}, LDL^T method={self.ldlt_method}, approx. inverse method={self.ainv_method}..."
        )
        # 1. Prepare item_user matrix (transpose of user-item matrix)
        training_logger.info("Loading item-user matrix...")
        X_T = train_split.get_csc_t_matrix()
        # 2. Store number of items
        self.n_items = train_split.n_items
        # 3. Compute weights
        training_logger.info("Constructing weights:")
        self.weights, construct_weights_time = self._construct_weights(X_T)
        del X_T
        # we report the time it took to construct the weights as training time, as we want to ignore data loading
        self.stats_trace["construct_weights_time"] = construct_weights_time
        training_logger.debug("Training done.")

    @execution_time(logger=evaluation_logger)
    def _predict(self, X: sp.csr_matrix) -> tuple[sp.csr_matrix, list[float]]:
        """Predict ratings for a user."""
        # 1. Compute predicted ratings vector
        evaluation_logger.debug("Matrix-matrix multiplication 1/2...")
        XW_T, matmat1_time = _matmat(X, self.weights[0])

        evaluation_logger.debug("Matrix-matrix multiplication 2/2...")
        P, matmat2_time = _matmat(XW_T, self.weights[1])
        self.stats_trace["matmat_times"] = [matmat1_time, matmat2_time]
        # We need to add operation simulating B[di] = 0 (where B = W.T @ W_r) -> TRICK:
        #
        # Let K = B - diag(B) (that is, K is the matrix with all diagonal entries set to zero: B = K + diag(B) )
        # Prediction: P = X @ K = X @ B - X @ diag(B)
        # But: diag(B) = -I (because we normalized the columns of W by negative diagonal entries)
        # So: P = X @ K = X @ B - X @ -I = X @ B + X
        # But we mask items that have been interacted with anyway with -inf, so we can just skip this P += X
        evaluation_logger.debug("Masking weights...")
        with warnings.catch_warnings():  # ignore warning about changing sparsity pattern, here it has basically no effect on performance
            warnings.simplefilter("ignore")
            P[X.nonzero()] = -np.inf
        evaluation_logger.debug("Prediction done.")
        return P

    def recommend(
        self, inputs_df: pd.DataFrame, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recommend for a batch of users."""
        # Create sparse user-item matrix of feedbacks
        n_users = inputs_df.user_id.nunique()
        X = df_to_csr(df=inputs_df, shape=(n_users, self.n_items))

        # Run prediction to obtain Predicted Score Matrix
        evaluation_logger.debug(f"Predicting scores (n_users = {n_users}):")
        P, predict_time = self._predict(X)
        self.stats_trace["predict_time"] = predict_time

        evaluation_logger.debug("Filtering, sorting...")

        # Get indices of top k items for each user and corresponding scores
        evaluation_logger.debug("Filtering top k items...")
        top_k_ids, top_k_scores = row_top_k(P.data, P.indices, P.indptr, k)

        evaluation_logger.debug("Sorting top k items...")
        # Create sorting array to sort top_k_idx_mat and top_k_scores_mat in descending order of scores
        sorting = np.argsort(-top_k_scores, axis=1)
        # sort top_k_idx_mat and top_k_scores_mat
        top_k_ids = top_k_ids[np.arange(n_users)[:, np.newaxis], sorting]
        top_k_scores = top_k_scores[np.arange(n_users)[:, np.newaxis], sorting]

        evaluation_logger.debug("Recommendation done.")

        return top_k_ids, top_k_scores

    def get_config(self) -> dict:
        """Return model configuration."""
        config = {
            "l2": self.l2,
            "target_density": self.target_density,
            "ainv_method": self.ainv_method,
            "ainv_params": self.ainv_params,
            "ldlt_method": self.ldlt_method,
            "ldlt_params": self.ldlt_params,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> "SANSA":
        l2 = config["l2"]  # must be specified
        target_density = config["target_density"]  # must be specified
        ldlt_method = config["ldlt_method"]  # must be specified
        ldlt_params = config.get(
            "ldlt_params", {}
        )  # hyperparameters for cholmod (other reorderings, etc.). Default parameters are fine (uses COLAMD). For larger systems, it may be necessary to set "use_long": True.
        ainv_method = config.get("ainv_method", "umr")
        ainv_params = config["ainv_params"]  # must be specified
        model = cls(
            l2=l2,
            target_density=target_density,
            ainv_method=ainv_method,
            ainv_params=ainv_params,
            ldlt_method=ldlt_method,
            ldlt_params=ldlt_params,
        )
        return model

    def get_num_weights(self) -> int:
        """Return number of parameters in model."""
        return self.weights[0].nnz + self.weights[1].nnz

    def get_weights_size(self) -> int:
        """Return size of weights in bytes."""
        size = 0
        for i in range(2):
            size += (
                self.weights[i].data.nbytes
                + self.weights[i].indices.nbytes
                + self.weights[i].indptr.nbytes
            )
        return size
