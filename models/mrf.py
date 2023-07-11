"""
Markov Random Field (MRF) model; sparse modification of EASE.

https://dl.acm.org/doi/10.5555/3454287.3454778
"""

import gc
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import warnings

from copy import deepcopy

from datasets.split import DatasetSplit, df_to_csr
from evaluation.logs import training_logger, evaluation_logger
from evaluation.metrics import execution_time
from models.ease import EASE
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
        # "Safety feature": this will crash if k > len(row_entries): kth out of bounds. Fix if it crashes during the experiments.
        ids = np.argpartition(row_entries, -k)[-k:]
        top_k_ids[i] = indices[indptr[i] : indptr[i + 1]][ids]
        top_k_scores[i] = data[indptr[i] : indptr[i + 1]][ids]
    return top_k_ids, top_k_scores


def dense_row_top_k(predictions, k):
    top_k_ids = np.argpartition(predictions, -k, axis=1)[:, -k:]
    top_k_scores = np.take_along_axis(predictions, top_k_ids, axis=1)
    return top_k_ids, top_k_scores


# unchanged from the original implementation, except for logging
@execution_time(logger=training_logger)
def calculate_sparsity_pattern(XtX, threshold, maxInColumn):
    # this implements section 3.1 in the paper.

    training_logger.info("sparsifying the data-matrix (section 3.1 in the paper) ...")
    # apply threshold
    ix = np.where(np.abs(XtX) > threshold)
    AA = sp.csc_matrix((XtX[ix], ix), shape=XtX.shape, dtype=np.float32)
    # enforce maxInColumn, see section 3.1 in paper
    countInColumns = AA.getnnz(axis=0)
    iiList = np.where(countInColumns > maxInColumn)[0]
    training_logger.info(
        f"number of items with more than {maxInColumn} entries in column: {len(iiList)}"
    )
    for ii in iiList:
        jj = AA[:, ii].nonzero()[0]
        kk = np.argpartition(
            -np.abs(np.asarray(AA[jj, ii].todense()).flatten()), maxInColumn
        )[maxInColumn:]
        AA[jj[kk], ii] = 0.0
    AA.eliminate_zeros()

    density = AA.nnz * 1.0 / AA.shape[0] / AA.shape[0]
    training_logger.info(f"resulting density of AA: {density}")
    return AA, density


# unchanged from the original implementation, except for the diagonal zeroing and logging
@execution_time(logger=training_logger)
def sparse_parameter_estimation(rr, XtX, AA):
    # this implements section 3.2 in the paper

    # list L in the paper, sorted by item-counts per column, ties broken by item-popularities as reflected by np.diag(XtX)
    AAcountInColumns = AA.getnnz(axis=0)
    sortedList = np.argsort(
        AAcountInColumns + np.diag(XtX) / 2.0 / np.max(np.diag(XtX))
    )[::-1]

    training_logger.info(
        "iterating through steps 1,2, and 4 in section 3.2 of the paper ..."
    )
    todoIndicators = np.ones(AAcountInColumns.shape[0])
    blockList = (
        []
    )  # list of blocks. Each block is a list of item-indices, to be processed in step 3 of the paper
    for ii in sortedList:
        if todoIndicators[ii] == 1:
            nn, _, vals = sp.find(
                AA[:, ii]
            )  # step 1 in paper: set nn contains item ii and its neighbors N
            kk = np.argsort(np.abs(vals))[::-1]
            nn = nn[kk]
            blockList.append(
                nn
            )  # list of items in the block, to be processed in step 3 below
            # remove possibly several items from list L, as determined by parameter rr (r in the paper)
            dd_count = max(1, int(np.ceil(len(nn) * rr)))
            dd = nn[:dd_count]  # set D, see step 2 in the paper
            todoIndicators[dd] = 0  # step 4 in the paper

    training_logger.info("now step 3 in section 3.2 of the paper: iterating ...")
    # now the (possibly heavy) computations of step 3:
    # given that steps 1,2,4 are already done, the following for-loop could be implemented in parallel.
    BBlist_ix1, BBlist_ix2, BBlist_val = [], [], []
    for nn in blockList:
        # calculate dense solution for the items in set nn
        BBblock = np.linalg.inv(XtX[np.ix_(nn, nn)])
        BBblock /= -np.diag(BBblock)
        # determine set D based on parameter rr (r in the paper)
        dd_count = max(1, int(np.ceil(len(nn) * rr)))
        dd = nn[:dd_count]  # set D in paper
        # store the solution regarding the items in D
        blockix = np.meshgrid(dd, nn)
        BBlist_ix1.extend(blockix[1].flatten().tolist())
        BBlist_ix2.extend(blockix[0].flatten().tolist())
        BBlist_val.extend(BBblock[:, :dd_count].flatten().tolist())

    training_logger.info(
        "final step: obtaining the sparse matrix BB by averaging the solutions regarding the various sets D ..."
    )
    BBsum = sp.csc_matrix(
        (BBlist_val, (BBlist_ix1, BBlist_ix2)), shape=XtX.shape, dtype=np.float32
    )
    BBcnt = sp.csc_matrix(
        (np.ones(len(BBlist_ix1), dtype=np.float32), (BBlist_ix1, BBlist_ix2)),
        shape=XtX.shape,
        dtype=np.float32,
    )
    b_div = sp.find(BBcnt)[2]
    b_3 = sp.find(BBsum)
    BBavg = sp.csc_matrix(
        (b_3[2] / b_div, (b_3[0], b_3[1])), shape=XtX.shape, dtype=np.float32
    )

    # ONLY MODIFICATION WE MADE IN THIS FUNCTION (apart from logging):
    BBavg.setdiag(0)  # instead of BBavg[ii_diag] = 0.0

    training_logger.info("forcing the sparsity pattern of AA onto BB ...")
    BBavg = sp.csr_matrix(
        (np.asarray(BBavg[AA.nonzero()]).flatten(), AA.nonzero()),
        shape=BBavg.shape,
        dtype=np.float32,
    )

    training_logger.info(
        "resulting sparsity of learned BB: {}".format(
            BBavg.nnz * 1.0 / AA.shape[0] / AA.shape[0]
        )
    )

    return BBavg


@execution_time(logger=training_logger)
def sparse_solution(XtX, XtXdiag, ii_diag, rr, threshold, maxInColumn, L2reg):
    # sparsity pattern, see section 3.1 in the paper
    XtX[ii_diag] = XtXdiag
    (AA, density), sparsity_pattern_time = calculate_sparsity_pattern(
        XtX, threshold, maxInColumn
    )

    # parameter-estimation, see section 3.2 in the paper
    XtX[ii_diag] = XtXdiag + L2reg
    BBsparse, sparse_estimation_time = sparse_parameter_estimation(rr, XtX, AA)

    return BBsparse, density, sparsity_pattern_time, sparse_estimation_time


class MRF(EASE):
    """
    MRF, a sparse approximation of EASE via Markov Random Fields.

    Copied and reorganized from https://github.com/hasteck/MRF_NeurIPS_2019/blob/master/mrf_for_cf_NeurIPS_2019.ipynb
    """

    # TODO:
    def __init__(
        self,
        l2: float,
        alpha: float,
        threshold: float,
        maxInColumn: int,
        rr: float,
        sparse_evaluation: bool = False,
    ) -> None:
        """
        Initialize SANSA model.

        Args:
            l2: L2 regularization parameter.
        """
        super().__init__(l2=l2)
        self.alpha = alpha
        self.threshold = threshold
        self.maxInColumn = maxInColumn
        self.rr = rr
        self.sparse_evaluation = sparse_evaluation

        self.weights = None  # List of matrices
        self.stats_trace = dict()

    # Training process from the MRF notebook
    @execution_time(logger=training_logger)
    def _construct_weights(self, X: sp.csr_matrix) -> list[sp.spmatrix]:
        """Construct weights for MRF."""
        # input: X ... user-item interaction matrix (sparse, binary), gets loaded in step 1 above.
        userCount = X.shape[0]
        XtX = np.asarray(_m(X.T, X).todense(), dtype=np.float32)
        del X

        mu = (
            np.diag(XtX) / userCount
        )  # the mean of the columns in train_data (for binary train_data)
        variance_times_userCount = (
            np.diag(XtX) - mu * mu * userCount
        )  # variances of columns in train_data (scaled by userCount)

        # standardizing the data-matrix XtX (if alpha=1, then XtX becomes the correlation matrix)
        # WARNING: THIS MAKES MATRIX XTX DENSE. REQUIRES BIG DATA OPERATIONS ON LARGE DATASETS = EXPENSIVE
        XtX -= mu[:, None] * (mu * userCount)
        rescaling = np.power(variance_times_userCount, self.alpha / 2.0)
        with np.errstate(
            divide="ignore"
        ):  # can raise divide by zero warning when using intel MKL numpy. Most likely cause by endianness
            scaling = 1.0 / rescaling
        XtX = scaling[:, None] * XtX * scaling
        gc.collect()

        XtXdiag = deepcopy(np.diag(XtX))
        ii_diag = np.diag_indices(XtX.shape[0])

        training_logger.info("Training the sparse model:")
        (
            B,
            density,
            sparsity_pattern_time,
            sparse_estimation_time,
        ), sparse_solution_time = sparse_solution(
            XtX, XtXdiag, ii_diag, self.rr, self.threshold, self.maxInColumn, self.l2
        )
        del XtX
        gc.collect()
        self.stats_trace["density"] = density
        self.stats_trace["sparsity_pattern_time"] = sparsity_pattern_time
        self.stats_trace["sparse_estimation_time"] = sparse_estimation_time
        self.stats_trace["sparse_solution_time"] = sparse_solution_time
        training_logger.info("Re-scaling BB back to the original item-popularities ...")
        # assuming that mu.T.dot(BB) == mu, see Appendix in paper
        B = sp.diags(scaling).dot(B).dot(sp.diags(rescaling))
        gc.collect()

        return [B.astype(np.float64)]

    def train(self, train_split: DatasetSplit) -> None:
        """Fit the model to the train_split."""
        training_logger.info(f"Train user-item matrix info | {train_split.info()}")
        training_logger.info(f"Item-item matrix info | {train_split.item_item_info()}")
        training_logger.info(
            f"Training MRF with L2={self.l2}, alpha={self.alpha}, threshold={self.threshold}, maxInColumn={self.maxInColumn}, rr={self.rr}"
        )
        # 1. Prepare item_user matrix (transpose of user-item matrix)
        training_logger.info("Loading item-user matrix...")
        X = train_split.get_csr_matrix()
        # 2. Store number of items
        self.n_items = train_split.n_items
        # 3. Compute weights
        training_logger.info("Constructing weights:")
        self.weights, construct_weights_time = self._construct_weights(X)
        del X
        # we report the time it took to construct the weights as training time, as we want to ignore data loading
        self.stats_trace["construct_weights_time"] = construct_weights_time
        training_logger.debug("Training done.")

    @execution_time(logger=evaluation_logger)
    def _predict(self, X: sp.csr_matrix) -> tuple[sp.csr_matrix, list[float]]:
        """Predict ratings for a user."""
        # 1. Compute predicted ratings vector
        evaluation_logger.debug("Matrix-matrix multiplication 1/1...")
        P, matmat1_time = _matmat(X, self.weights[0])
        self.stats_trace["matmat_times"] = [matmat1_time]
        # 2. Mask out already rated items
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
        if (
            self.sparse_evaluation
        ):  # sparse evaluation can crash on k>40, because the model will not produce enough recommendations
            top_k_ids, top_k_scores = row_top_k(P.data, P.indices, P.indptr, k)
        else:  # dense evaluation will take some random items
            top_k_ids, top_k_scores = dense_row_top_k(P.toarray(), k)

        evaluation_logger.debug("Sorting top k items...")
        # Create sorting array to sort top_k_idx_mat and top_k_scores_mat in descending order of scores
        sorting = np.argsort(-top_k_scores, axis=1)
        # sort top_k_idx_mat and top_k_scores_mat
        top_k_ids = top_k_ids[np.arange(n_users)[:, np.newaxis], sorting]
        top_k_scores = top_k_scores[np.arange(n_users)[:, np.newaxis], sorting]

        return top_k_ids, top_k_scores

    def get_config(self) -> dict:
        """Return model configuration."""
        config = {
            "l2": self.l2,
            "alpha": self.alpha,
            "threshold": self.threshold,
            "maxInColumn": self.maxInColumn,
            "rr": self.rr,
        }
        return config

    @classmethod
    def from_config(cls, config: dict) -> "MRF":
        l2 = config["l2"]
        alpha = config["alpha"]
        threshold = config["threshold"]
        maxInColumn = config["maxInColumn"]
        rr = config["rr"]
        model = cls(
            l2=l2,
            alpha=alpha,
            threshold=threshold,
            maxInColumn=maxInColumn,
            rr=rr,
        )
        return model

    def get_num_weights(self) -> int:
        """Return number of parameters in model."""
        return self.weights[0].nnz

    def get_weights_size(self) -> int:
        """Return size of weights in bytes."""
        size = (
            self.weights[0].data.nbytes
            + self.weights[0].indices.nbytes
            + self.weights[0].indptr.nbytes
        )
        return size
