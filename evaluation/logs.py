"""
Logging setup for the evaluation pipeline.
"""

import logging

from evaluation.steps import PIPELINE_STEPS

LOG_FORMAT = "%(asctime)s : %(STEP)s : %(message)s"

logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")

start_logger = logging.LoggerAdapter(logger, {"STEP": PIPELINE_STEPS["start"]})
dataset_logger = logging.LoggerAdapter(logger, {"STEP": PIPELINE_STEPS["dataset"]})
training_logger = logging.LoggerAdapter(logger, {"STEP": PIPELINE_STEPS["training"]})
evaluation_logger = logging.LoggerAdapter(
    logger, {"STEP": PIPELINE_STEPS["evaluation"]}
)
end_logger = logging.LoggerAdapter(logger, {"STEP": PIPELINE_STEPS["end"]})
