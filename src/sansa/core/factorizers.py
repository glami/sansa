import gc
import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.sparse as sp
import sksparse.cholmod as cholmod

from ..utils import inplace_scale_along_compressed_axis, inplace_sparsify, matmat
from ._ops import icf

logger = logging.getLogger(__name__)


class ReorderingMode(Enum):
    SIMPLICIAL = "simplicial"
    SUPERNODAL = "supernodal"


class ReorderingMethod(Enum):
    COLAMD = "colamd"


class FactorizationMethod(Enum):
    CHOLMOD = "CHOLMOD"
    ICF = "ICF"


@dataclass
class GramianFactorizerConfig(ABC):
    reordering_use_long: bool = False
    reordering_method: ReorderingMethod = ReorderingMethod.COLAMD

    @abstractmethod
    def __post_init__(self) -> None:
        self.reordering_mode: ReorderingMode
        self.factorization_method: FactorizationMethod
        raise NotImplementedError("Implement this")


@dataclass
class CHOLMODGramianFactorizerConfig(GramianFactorizerConfig):

    def __post_init__(self) -> None:
        self.reordering_mode = ReorderingMode.SUPERNODAL
        self.factorization_method = FactorizationMethod.CHOLMOD


@dataclass
class ICFGramianFactorizerConfig(GramianFactorizerConfig):
    factorization_shift_step: float = 1e-3
    factorization_shift_multiplier: float = 2.0

    def __post_init__(self) -> None:
        self.reordering_mode = ReorderingMode.SIMPLICIAL
        self.factorization_method = FactorizationMethod.ICF


class GramianFactorizer(ABC):

    def __init__(self, config: GramianFactorizerConfig):
        self.reordering_mode = config.reordering_mode
        self.reordering_use_long = config.reordering_use_long
        self.reordering_method = config.reordering_method
        self.factorization_method = config.factorization_method

    @property
    @abstractmethod
    def config(self) -> GramianFactorizerConfig:
        raise NotImplementedError("Implement this")

    @classmethod
    def from_config(cls, config: GramianFactorizerConfig) -> "GramianFactorizer":
        module = importlib.import_module(__name__)
        class_name = "".join([config.factorization_method.value, cls.__name__])
        factorizer_class = getattr(module, class_name)
        return factorizer_class(config)

    @staticmethod
    def _clip_density_to_lower_bound(factor_density: float, minimum_density: float) -> float:
        if factor_density < minimum_density:
            logger.warning(
                f"""
                Too Sparse Warning: 
                    Selected density {factor_density:%} is too low, clipping to {minimum_density:%}. 
                    Minimum density might result in worse quality of the approximate factor.
                """
            )
            return minimum_density
        return factor_density

    @abstractmethod
    def approximate_cholesky(
        self, X: sp.csr_matrix, l2: float, factor_density: float, compute_gramian: bool
    ) -> tuple[sp.csc_matrix, np.ndarray]:
        raise NotImplementedError("Implement this")

    def approximate_ldlt(
        self,
        X: sp.csr_matrix,
        l2: float,
        factor_density: float,
        compute_gramian: bool,
    ) -> tuple[sp.csc_matrix, sp.dia_matrix, np.ndarray]:

        # 1. Compute incomplete Cholesky decomposition of
        # - P(X^TX + self.l2 * I)P^T if compute_gramian=True
        # - P(X + self.l2 * I)P^T if compute_gramian=False
        logger.info(f"Computing incomplete Cholesky decomposition of X^TX + {l2}*I...")
        L, p = self.approximate_cholesky(X, l2, factor_density, compute_gramian)

        # 2. Compute LDL^T decomposition of A' from LL^T decomposition
        logger.info("Scaling columns and creating diagonal matrix D (LL^T -> L'DL'^T)...")
        d = L.diagonal()
        # The following operation raises divide by zero warning when using intel MKL numpy (caused by endianness)
        with np.errstate(divide="ignore"):
            inplace_scale_along_compressed_axis(L, 1 / d)
        d **= 2
        D = sp.dia_matrix(([d], [0]), shape=L.shape)

        return L, D, p


class CHOLMODGramianFactorizer(GramianFactorizer):

    def __init__(self, config: CHOLMODGramianFactorizerConfig):
        super().__init__(config)

    @property
    def config(self) -> CHOLMODGramianFactorizerConfig:
        return CHOLMODGramianFactorizerConfig(
            reordering_use_long=self.reordering_use_long,
            reordering_method=self.reordering_method,
        )

    @staticmethod
    def _suggest_icf_if_desired_density_too_low(desired_density: float) -> None:
        if desired_density <= 0.05:
            logger.warning(
                f"""
                For low desired desired ({desired_density:%}), computing exact factorization (CHOLMOD) 
                followed by sparsification may be inefficient.
                You may want to try {ICFGramianFactorizer.__name__} instead of {CHOLMODGramianFactorizer.__name__} 
                (requires less memory and may be faster).
                """
            )

    def approximate_cholesky(
        self,
        X: sp.csr_matrix,
        l2: float,
        factor_density: float,
        compute_gramian: bool,
    ) -> tuple[sp.csc_matrix, np.ndarray]:

        # 0. Clip density to a reasonable minimum
        minimum_density = 2 / X.shape[1]
        factor_density = self._clip_density_to_lower_bound(factor_density, minimum_density)
        self._suggest_icf_if_desired_density_too_low(factor_density)

        # 1. Compute symbolic Cholesky factorization
        # - of X^TX if compute_gramian=True
        # - of X if compute_gramian=False
        # along with fill-in reducing ordering
        logger.info(f"Finding a fill-in reducing ordering (method = {self.reordering_method.value})...")
        if compute_gramian:
            factor = cholmod.analyze_AAt(
                X.transpose(),
                mode=self.reordering_mode.value,
                use_long=self.reordering_use_long,
                ordering_method=self.reordering_method.value,
            )
        else:
            factor = cholmod.analyze(
                X.transpose(),
                mode=self.reordering_mode.value,
                use_long=self.reordering_use_long,
                ordering_method=self.reordering_method.value,
            )
        p = factor.P()

        # 2. Compute numerical factorization
        logger.info(f"Computing approximate Cholesky decomposition (method = {self.factorization_method.value})...")
        if compute_gramian:
            factor.cholesky_AAt_inplace(X.transpose(), beta=l2)
        else:
            factor.cholesky_inplace(X.transpose(), beta=l2)
        L = factor.L().tocsc()
        del factor
        gc.collect()

        # 3. Drop small values from L
        current_density = L.nnz / (L.shape[0] ** 2)
        logger.info(f"Dropping small entries in L ({current_density:%} dense, target = {factor_density:%})...")
        inplace_sparsify(L, factor_density)

        return L, p


class ICFGramianFactorizer(GramianFactorizer):

    def __init__(self, config: ICFGramianFactorizerConfig):
        super().__init__(config)
        self.factorization_shift_step = config.factorization_shift_step
        self.factorization_shift_multiplier = config.factorization_shift_multiplier

    @property
    def config(self) -> ICFGramianFactorizerConfig:
        return ICFGramianFactorizerConfig(
            factorization_shift_step=self.factorization_shift_step,
            factorization_shift_multiplier=self.factorization_shift_multiplier,
            reordering_use_long=self.reordering_use_long,
            reordering_method=self.reordering_method,
        )

    @staticmethod
    def _index_dtypes_to_int64(A: sp.csc_matrix):
        if not A.indptr.dtype == np.int64:
            logger.info("Casting indptr of A to int64...")
            A.indptr = A.indptr.astype(np.int64)
        if not A.indices.dtype == np.int64:
            logger.info("Casting indices of A to int64...")
            A.indices = A.indices.astype(np.int64)

    @staticmethod
    def _suggest_cholmod_if_A_too_dense(A: sp.csc_matrix) -> None:
        density = A.nnz / (A.shape[0] ** 2)
        if density > 0.05:
            logger.warning(
                f"""
                Attempting incomplete factorization of a relatively dense matrix ({density:%} dense). 
                This is unstable:
                 - the factorization might fail and automatically restart with additional regularization
                 - the resulting approximate factor might be of lesser quality
                You may want to try {CHOLMODGramianFactorizer.__name__} instead of {ICFGramianFactorizer.__name__} 
                (requires more memory but is likely faster and more accurate).
                """
            )

    @staticmethod
    def _suggest_cholmod_if_desired_density_too_high(desired_density: float) -> None:
        if desired_density > 0.05:
            logger.warning(
                f"""
                Attempting incomplete factorization with high desired density ({desired_density:%}). 
                This may be inefficient:
                 - the factorization might fail and automatically restart with additional regularization
                 - the resulting approximate factor might be of lesser quality
                You may want to try {CHOLMODGramianFactorizer.__name__} instead of {ICFGramianFactorizer.__name__} 
                (requires more memory but is likely faster and more accurate).
                """
            )

    def approximate_cholesky(
        self,
        X: sp.csr_matrix,
        l2: float,
        factor_density: float,
        compute_gramian: bool,
    ) -> tuple[sp.csc_matrix, np.ndarray]:

        # 0. Clip density to a reasonable minimum
        minimum_density = 2 / X.shape[1]
        factor_density = self._clip_density_to_lower_bound(factor_density, minimum_density)
        self._suggest_cholmod_if_desired_density_too_high(factor_density)

        # 1. Compute COLAMD permutation of A ( A' = [p, :]A[:, p]] )
        logger.info(f"Finding a fill-in reducing ordering (method = {self.reordering_method.value})...")
        if compute_gramian:
            p = cholmod.analyze_AAt(
                X.transpose(),
                mode=self.reordering_mode.value,
                use_long=self.reordering_use_long,
                ordering_method=self.reordering_method.value,
            ).P()
        else:
            p = cholmod.analyze(
                X.transpose(),
                mode=self.reordering_mode.value,
                use_long=self.reordering_use_long,
                ordering_method=self.reordering_method.value,
            ).P()
        gc.collect()


        if compute_gramian:
            # 2. permute columns of X
            X_P = X[:, p]  # column permutation
        else:
            X_P = X[:, p]  # column permutation
            # 2. permute row and columns of X
            X_P = X_P[p, :]  # row permutation

        if compute_gramian:
            # 3. Compute A = P(X^TX)P^T
            logger.info("Computing X^TX...")
            A = matmat(X_P.transpose(), X_P)
            logger.info(
                f"""
                X^TX info:
                    shape = {A.shape} 
                    nnz = {A.nnz} 
                    density = {A.nnz / (A.shape[0] ** 2):%} 
                    size = {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6:.1f} MB
                """
            )
            self._suggest_cholmod_if_A_too_dense(A)
        else:
            A = X_P
            logger.info(
                f"""
                X info:
                    shape = {A.shape} 
                    nnz = {A.nnz} 
                    density = {A.nnz / (A.shape[0] * A.shape[1]):%} 
                    size = {(A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / 1e6:.1f} MB
                """
            )
        del X_P

        # 4. Prepare A for ICF algorithm
        logger.info("Sorting indices of A...")
        A.sort_indices()
        self._index_dtypes_to_int64(A)
        gc.collect()

        # 5. Compute incomplete Cholesky factorization of A
        logger.info(f"Computing approximate Cholesky decomposition (method = {self.factorization_method.value})...")
        max_nnz = int(np.ceil(factor_density * A.shape[0] ** 2))
        L = icf(
            A,
            l2=l2,
            max_nnz=max_nnz,
            shift_step=self.factorization_shift_step,
            shift_multiplier=self.factorization_shift_multiplier,
        )

        # 6. Delete A, we don't need it anymore
        del A
        gc.collect()

        return L, p
