#
# Copyright 2023 Inspigroup s.r.o.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
#
# https://github.com/glami/sansa/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
#
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
