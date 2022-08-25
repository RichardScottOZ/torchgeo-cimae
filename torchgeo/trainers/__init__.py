# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .cae import CAETask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .embedding_evaluation import EmbeddingEvaluator
from .mae import MAETask
from .msae import MSAETask
from .msn import MSNTask
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
    "MAETask",
    "MSNTask",
    "MSAETask",
    "CAETask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
