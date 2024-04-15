import logging
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from .core import (
    FactorizationMethod,
    GramianFactorizer,
    GramianFactorizerConfig,
    UnitLowerTriangleInverter,
    UnitLowerTriangleInverterConfig,
)
from .utils import (
    get_squared_norms_along_compressed_axis,
    inplace_scale_along_compressed_axis,
    inplace_scale_along_uncompressed_axis,
    matmat,
    top_k_along_compressed_axis,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _apply_icf_scaling(X: sp.csr_matrix, compute_gramian: bool) -> None:
    if compute_gramian:
        # Inplace scale columns of X by square roots of column norms of X^TX.
        logger.info(f"Computing column norms of X^TX...")
        da = np.sqrt(np.sqrt(get_squared_norms_along_compressed_axis(matmat(X.T, X))))
        # Divide columns of X by the computed square roots of row norms of X^TX
        da[da == 0] = 1  # ignore zero elements
        logger.info(f"Scaling columns of X by computed norms...")
        inplace_scale_along_uncompressed_axis(X, 1 / da)  # CSR column scaling
        del da
    else:
        # Inplace scale rows and columns of X by square roots of row norms of X.
        logger.info(f"Computing row norms of X...")
        da = np.sqrt(np.sqrt(get_squared_norms_along_compressed_axis(X)))
        # Divide rows and columns of X by the computed square roots of row norms of X
        da[da == 0] = 1  # ignore zero elements
        logger.info(f"Scaling rows and columns of X by computed norms...")
        inplace_scale_along_uncompressed_axis(X, 1 / da)  # CSR column scaling
        inplace_scale_along_compressed_axis(X, 1 / da)  # CSR row scaling
        del da



@dataclass
class SANSAConfig:
    l2: float
    weight_matrix_density: float
    gramian_factorizer_config: GramianFactorizerConfig
    lower_triangle_inverter_config: UnitLowerTriangleInverterConfig


class SANSA:
    def __init__(self, config: SANSAConfig) -> None:
        self.l2 = config.l2
        self.weight_matrix_density = config.weight_matrix_density
        self.factorizer = GramianFactorizer.from_config(config.gramian_factorizer_config)
        self.factorization_method = config.gramian_factorizer_config.factorization_method
        self.inverter = UnitLowerTriangleInverter.from_config(config.lower_triangle_inverter_config)
        self.weights = (None, None)

    @property
    def config(self) -> SANSAConfig:
        return SANSAConfig(
            self.l2,
            self.weight_matrix_density,
            self.factorizer.config,
            self.inverter.config,
        )

    def load_weights(self, weights: Tuple[sp.csr_matrix, sp.csr_matrix]) -> "SANSA":
        self.weights = weights
        return self

    def fit(self, training_matrix: sp.csr_matrix, compute_gramian=True) -> "SANSA":
        """
        Fit SANSA model with user-item or item-item matrix.
        """
        # create a working copy of user_item_matrix
        X = training_matrix.copy()
        X = X.astype(np.float32)

        if self.factorization_method == FactorizationMethod.ICF:
            # scale matrix X
            _apply_icf_scaling(X, compute_gramian)

        # Compute LDL^T decomposition of
        # - P(X^TX + self.l2 * I)P^T if compute_gramian=True
        # - P(X + self.l2 * I)P^T if compute_gramian=False
        logger.info("Computing LDL^T decomposition of permuted item-item matrix...")
        L, D, p = self.factorizer.approximate_ldlt(
            X,
            self.l2,
            self.weight_matrix_density,
            compute_gramian=compute_gramian,
        )
        del X

        # Compute approximate inverse of L using selected method
        logger.info("Computing approximate inverse of L...")
        L_inv = self.inverter.invert(L)
        del L

        # Construct W = L_inv @ P
        logger.info("Constructing W = L_inv @ P...")
        inv_p = np.argsort(p)
        W = L_inv[:, inv_p]
        del L_inv

        # Construct W_r (A^{-1} = W.T @ W_r)
        W_r = W.copy()
        inplace_scale_along_uncompressed_axis(W_r, 1 / D.diagonal())

        # Extract diagonal entries
        logger.info("Extracting diagonal of W.T @ D_inv @ W...")
        diag = W.copy()
        diag.data = diag.data**2
        inplace_scale_along_uncompressed_axis(diag, 1 / D.diagonal())
        diagsum = diag.sum(axis=0)  # original
        del diag
        diag = np.asarray(diagsum)[0]

        # Divide columns of the inverse by negative diagonal entries
        logger.info("Dividing columns of W by diagonal entries...")
        # equivalent to dividing the columns of W by negative diagonal entries
        inplace_scale_along_compressed_axis(W_r, -1 / diag)
        self.weights = (W.T.tocsr(), W_r.tocsr())

        return self

    def forward(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """
        Forward pass.
        """
        latent = X @ self.weights[0]
        out = latent @ self.weights[1]
        return out

    def recommend(self, interactions: sp.csr_matrix, k: int, mask_input: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recommend top k items for a batch of users given as a CSR matrix.
        Fails with "kth out of bounds" error if for some user the model can't recommend k items (=model is too sparse).
        """
        n_users = interactions.shape[0]
        predictions = self.forward(interactions)
        if mask_input:
            with warnings.catch_warnings():  # ignore warning about changing sparsity pattern
                warnings.simplefilter("ignore")
                predictions[interactions.nonzero()] = 0

        # Get indices of top k items for each user and corresponding scores
        top_k_ids, top_k_scores = top_k_along_compressed_axis(predictions, k)
        # sort top_k_idx matrix and top_k_scores matrix
        sorting = np.argsort(-top_k_scores, axis=1)
        top_k_ids = top_k_ids[np.arange(n_users)[:, np.newaxis], sorting]
        top_k_scores = top_k_scores[np.arange(n_users)[:, np.newaxis], sorting]

        return top_k_ids, top_k_scores
