# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .embedding_evaluation import EmbeddingEvaluator
from .mae import MAETask
from .regression import RegressionTask
from .segmentation import SemanticSegmentationTask
from .tile2vec import Tile2VecTask
from .vicreg import VICRegTask

__all__ = (
    "Tile2VecTask",
    "BYOLTask",
    "ClassificationTask",
    "MultiLabelClassificationTask",
    "RegressionTask",
    "SemanticSegmentationTask",
    "EmbeddingEvaluator",
    "VICRegTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
