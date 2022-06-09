# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

from .batch import (
    BatchGeoSampler,
    RandomBatchGeoSampler,
    TripletBatchGeoSampler,
    TripletTileBatchGeoSampler,
)
from .block import (
    RandomBlockBatchGeoSampler,
    TripletBlockBatchGeoSampler,
    TripletTileBlockBatchGeoSampler,
)
from .constants import Units
from .single import GeoSampler, GridGeoSampler, PreChippedGeoSampler, RandomGeoSampler

__all__ = (
    # Samplers
    "GridGeoSampler",
    "PreChippedGeoSampler",
    "RandomGeoSampler",
    # Batch samplers
    "RandomBatchGeoSampler",
    "TripletBatchGeoSampler",
    "TripletTileBatchGeoSampler",
    # Base classes
    "GeoSampler",
    "BatchGeoSampler",
    # Constants
    "Units",
    # Block
    "RandomBlockBatchGeoSampler",
    "TripletBlockBatchGeoSampler",
    "TripletTileBlockBatchGeoSampler",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.samplers"
