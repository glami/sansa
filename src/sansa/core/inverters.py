import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import scipy.sparse as sp

from ..utils import get_norms_along_compressed_axis
from ._ops import get_residual_matrix, s1, umr

logger = logging.getLogger(__name__)


@dataclass
class ApproximateInverseMethod(Enum):
    UMR = "UMR"


@dataclass
class UnitLowerTriangleInverterConfig:

    @abstractmethod
    def __post_init__(self) -> None:
        self.approximate_inverse_method: ApproximateInverseMethod
        raise NotImplementedError("Implement this")


class UnitLowerTriangleInverter(ABC):

    def __init__(self, config: UnitLowerTriangleInverterConfig):
        self.approximate_inverse_method = config.approximate_inverse_method

    @property
    @abstractmethod
    def config(self) -> UnitLowerTriangleInverterConfig:
        raise NotImplementedError("Implement this")

    @classmethod
    def from_config(cls, config: UnitLowerTriangleInverterConfig) -> "UnitLowerTriangleInverter":
        module = importlib.import_module(__name__)
        class_name = "".join([config.approximate_inverse_method.value, cls.__name__])
        factorizer_class = getattr(module, class_name)
        return factorizer_class(config)

    @abstractmethod
    def invert(self, L: sp.csc_matrix) -> sp.csc_matrix:
        raise NotImplementedError("Implement this")


@dataclass
class UMRUnitLowerTriangleInverterConfig(UnitLowerTriangleInverterConfig):
    scans: int = 1
    finetune_steps: int = 10
    log_norm_threshold: float = -7  # log10 of the norm threshold

    def __post_init__(self) -> None:
        self.approximate_inverse_method = ApproximateInverseMethod.UMR


class UMRUnitLowerTriangleInverter(UnitLowerTriangleInverter):
    def __init__(self, config: UMRUnitLowerTriangleInverterConfig):
        super().__init__(config)
        self.scans = config.scans
        self.finetune_steps = config.finetune_steps
        self.log_norm_threshold = config.log_norm_threshold

    @property
    def config(self) -> UMRUnitLowerTriangleInverterConfig:
        return UMRUnitLowerTriangleInverterConfig(
            self.scans,
            self.finetune_steps,
            self.log_norm_threshold,
        )

    def invert(self, L: sp.csc_matrix) -> sp.csc_matrix:
        density = L.nnz / (L.shape[0] * L.shape[1])
        logger.info("Calculating initial guess using 1 step of Schultz method...")
        M_0 = s1(L)  # initial guess
        logger.info("Calculating approximate inverse using Uniform Minimal Residual algorithm...")
        L_inv = umr(
            A=L,
            M_0=M_0,
            target_density=density,
            num_scans=self.scans,
            num_finetune_steps=self.finetune_steps,
            log_norm_threshold=self.log_norm_threshold,
        )
        del M_0

        # Final residual norm evaluation:
        # Compute residual matrix
        R = get_residual_matrix(L, L_inv)
        # Compute column norms of R
        norms = get_norms_along_compressed_axis(R)
        del R
        max_res = np.max(norms)  # Maximum residual
        loss = np.mean(norms**2)  # Loss = n * MSE = mean (column norm)^2 = (relative Frobenius norm)^2
        logger.info(f"Current maximum residual: {max_res}, relative Frobenius norm squared: {loss}")

        return L_inv
