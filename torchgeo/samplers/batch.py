# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo batch samplers."""

import abc
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


class BatchGeoSampler(Sampler[List[BoundingBox]], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.BatchSampler`, :class:`BatchGeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Optional[Union[BoundingBox, Sequence[BoundingBox]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if roi is None or not roi:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)
        else:
            if not isinstance(roi, Sequence):
                roi = [roi]

            self.index = Index(interleaved=False, properties=Property(dimension=3))
            for region in roi:
                hits = dataset.index.intersection(tuple(region), objects=True)
                for hit in hits:
                    bbox = BoundingBox(*hit.bounds) & region
                    bbox_time = BoundingBox(
                        bbox.minx,
                        bbox.maxx,
                        bbox.miny,
                        bbox.maxy,
                        region.mint,
                        region.maxt,
                    )
                    self.index.insert(hit.id, tuple(bbox_time), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[Union[BoundingBox, Sequence[BoundingBox]]] = None,
        units: Units = Units.PIXELS,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.batch_size = batch_size
        self.length = length
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(dataset.bounds), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)
                areas.append(bounds.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose random indices within that tile
            batch = []
            for _ in range(self.batch_size):

                bounding_box = get_random_bounding_box(bounds, self.size, self.res)
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


class TripletBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        neighborhood: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[Union[BoundingBox, Sequence[BoundingBox]]] = None,
        units: Units = Units.PIXELS,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.neighborhood = _to_tuple(neighborhood)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.neighborhood = (
                self.neighborhood[0] * self.res,
                self.neighborhood[1] * self.res,
            )

        self.batch_size = batch_size
        self.length = length
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(dataset.bounds), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)
                areas.append(bounds.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            batch = []
            for _ in range(self.batch_size):
                # Choose a random tile, weighted by area
                anchor_idx, distant_idx = torch.multinomial(
                    self.areas, 2, replacement=True
                )
                anchor_hit = self.hits[anchor_idx]
                anchor_bounds = BoundingBox(*anchor_hit.bounds)

                # Choose a random index within that tile
                anchor_bbox = get_random_bounding_box(
                    anchor_bounds, self.size, self.res
                )

                mid_x = anchor_bbox.minx + (anchor_bbox.maxx - anchor_bbox.minx) / 2
                mid_y = anchor_bbox.miny + (anchor_bbox.maxy - anchor_bbox.miny) / 2

                # Neighborhood based on midpoint of anchor
                # Contains positive midpoint => +/- half the patch size
                neighborhood_bounds = BoundingBox(
                    max(
                        anchor_bounds.minx,
                        mid_x - self.neighborhood[0] - self.size[0] / 2,
                    ),
                    min(
                        anchor_bounds.maxx,
                        mid_x + self.neighborhood[0] + self.size[0] / 2,
                    ),
                    max(
                        anchor_bounds.miny,
                        mid_y - self.neighborhood[1] - self.size[1] / 2,
                    ),
                    min(
                        anchor_bounds.maxy,
                        mid_y + self.neighborhood[1] + self.size[1] / 2,
                    ),
                    anchor_bounds.mint,
                    anchor_bounds.maxt,
                )

                neighbor_bbox = get_random_bounding_box(
                    neighborhood_bounds, self.size, self.res
                )

                distant_hit = self.hits[distant_idx]
                distant_bounds = BoundingBox(*distant_hit.bounds)

                distant_bbox = get_random_bounding_box(
                    distant_bounds, self.size, self.res
                )

                for _ in range(3):
                    if distant_bbox not in neighborhood_bounds:
                        break

                    distant_idx = torch.multinomial(self.areas, 1)
                    distant_hit = self.hits[distant_idx]
                    distant_bounds = BoundingBox(*distant_hit.bounds)

                    distant_bbox = get_random_bounding_box(
                        distant_bounds, self.size, self.res
                    )

                batch.append(anchor_bbox)
                batch.append(neighbor_bbox)
                batch.append(distant_bbox)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


class TripletTileBatchGeoSampler(BatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        neighborhood: Union[Tuple[float, float], float],
        batch_size: int,
        length: int,
        roi: Optional[Union[BoundingBox, Sequence[BoundingBox]]] = None,
        units: Units = Units.PIXELS,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            batch_size: number of samples per batch
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.neighborhood = _to_tuple(neighborhood)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.neighborhood = (
                self.neighborhood[0] * self.res,
                self.neighborhood[1] * self.res,
            )

        self.batch_size = batch_size
        self.length = length
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(dataset.bounds), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx >= self.size[1]
                and bounds.maxy - bounds.miny >= self.size[0]
            ):
                self.hits.append(hit)
                areas.append(bounds.area)

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return the indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            batch = []

            # Choose a random tile, weighted by area
            anchor_idx, distant_idx = torch.multinomial(self.areas, 2, replacement=True)

            anchor_hit = self.hits[anchor_idx]
            anchor_bounds = BoundingBox(*anchor_hit.bounds)

            distant_hit = self.hits[distant_idx]
            distant_bounds = BoundingBox(*distant_hit.bounds)

            for _ in range(self.batch_size):
                # Choose a random index within that tile
                anchor_bbox = get_random_bounding_box(
                    anchor_bounds, self.size, self.res
                )

                mid_x = anchor_bbox.minx + (anchor_bbox.maxx - anchor_bbox.minx) / 2
                mid_y = anchor_bbox.miny + (anchor_bbox.maxy - anchor_bbox.miny) / 2

                # Neighborhood based on midpoint of anchor
                # Contains positive midpoint => +/- half the patch size
                neighborhood_bounds = BoundingBox(
                    max(
                        anchor_bounds.minx,
                        mid_x - self.neighborhood[0] - self.size[0] / 2,
                    ),
                    min(
                        anchor_bounds.maxx,
                        mid_x + self.neighborhood[0] + self.size[0] / 2,
                    ),
                    max(
                        anchor_bounds.miny,
                        mid_y - self.neighborhood[1] - self.size[1] / 2,
                    ),
                    min(
                        anchor_bounds.maxy,
                        mid_y + self.neighborhood[1] + self.size[1] / 2,
                    ),
                    anchor_bounds.mint,
                    anchor_bounds.maxt,
                )

                neighbor_bbox = get_random_bounding_box(
                    neighborhood_bounds, self.size, self.res
                )

                distant_bbox = get_random_bounding_box(
                    distant_bounds, self.size, self.res
                )

                for _ in range(3):
                    if distant_bbox not in neighborhood_bounds:
                        break

                    distant_idx = torch.multinomial(self.areas, 1)
                    distant_hit = self.hits[distant_idx]
                    distant_bounds = BoundingBox(*distant_hit.bounds)

                    distant_bbox = get_random_bounding_box(
                        distant_bounds, self.size, self.res
                    )

                batch.append(anchor_bbox)
                batch.append(neighbor_bbox)
                batch.append(distant_bbox)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size
