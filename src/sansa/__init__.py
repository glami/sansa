import importlib.metadata

__version__ = importlib.metadata.version("sansa")

import sansa.core
import sansa.utils
from sansa.core import CHOLMODGramianFactorizerConfig, ICFGramianFactorizerConfig, UMRUnitLowerTriangleInverterConfig
from sansa.model import SANSA, SANSAConfig
