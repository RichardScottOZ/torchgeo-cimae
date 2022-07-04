# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo block batch samplers."""

import abc
from math import ceil, floor
from typing import Any, Iterator, List, Optional, Sequence, Tuple, Union

import rasterio
import torch
from rasterio.windows import from_bounds
from rtree.index import Index, Property
from torch.utils.data import Sampler

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import (
    _to_tuple,
    get_bounds_from_grid,
    get_random_bounding_box,
    get_random_bounding_box_from_grid,
)

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


class BlockGeoSampler(Sampler[BoundingBox], abc.ABC):
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
                    bounds = BoundingBox(*hit.bounds)
                    region_bounded = bounds & region

                    window_start: Optional[rasterio.windows.Window] = None
                    window_end: Optional[rasterio.windows.Window] = None
                    with rasterio.open(hit.object) as src:
                        [row_start], [col_start] = rasterio.transform.rowcol(
                            src.transform,
                            [region_bounded.minx],
                            [region_bounded.maxy],
                            op=ceil,
                        )

                        [row_end], [col_end] = rasterio.transform.rowcol(
                            src.transform,
                            [region_bounded.maxx],
                            [region_bounded.miny],
                            op=floor,
                        )

                        window_current: Optional[rasterio.windows.Window] = None
                        for _, window in src.block_windows():
                            if (
                                window.col_off < col_start
                                or window.row_off < row_start
                                or window.width < src.block_shapes[1][1]
                            ):
                                continue
                            if (
                                window.height < src.block_shapes[1][0]
                                or window.row_off > row_end
                            ):
                                break

                            if window_start is None:
                                window_start = window

                            if window.col_off <= col_end:
                                window_end = window_current

                            window_current = window

                    if window_start is None or window_end is None:
                        continue

                    minx, _, _, maxy = src.window_bounds(window_start)
                    _, miny, maxx, _ = src.window_bounds(window_end)

                    region_block = BoundingBox(
                        minx, maxx, miny, maxy, bounds.mint, bounds.maxt
                    )

                    self.index.insert(hit.id, tuple(region_block), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class BlockBatchGeoSampler(Sampler[List[BoundingBox]], abc.ABC):
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
                    bounds = BoundingBox(*hit.bounds)
                    region_bounded = bounds & region

                    window_start: Optional[rasterio.windows.Window] = None
                    window_end: Optional[rasterio.windows.Window] = None
                    with rasterio.open(hit.object) as src:
                        [row_start], [col_start] = rasterio.transform.rowcol(
                            src.transform,
                            [region_bounded.minx],
                            [region_bounded.maxy],
                            op=ceil,
                        )

                        [row_end], [col_end] = rasterio.transform.rowcol(
                            src.transform,
                            [region_bounded.maxx],
                            [region_bounded.miny],
                            op=floor,
                        )

                        window_current: Optional[rasterio.windows.Window] = None
                        for _, window in src.block_windows():
                            if (
                                window.col_off < col_start
                                or window.row_off < row_start
                                or window.width < src.block_shapes[1][1]
                            ):
                                continue
                            if (
                                window.height < src.block_shapes[1][0]
                                or window.row_off > row_end
                            ):
                                break

                            if window_start is None:
                                window_start = window

                            if window.col_off <= col_end:
                                window_end = window_current

                            window_current = window

                    if window_start is None or window_end is None:
                        continue

                    minx, _, _, maxy = src.window_bounds(window_start)
                    _, miny, maxx, _ = src.window_bounds(window_end)

                    region_block = BoundingBox(
                        minx, maxx, miny, maxy, bounds.mint, bounds.maxt
                    )

                    self.index.insert(hit.id, tuple(region_block), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[List[BoundingBox]]:
        """Return a batch of indices of a dataset.

        Returns:
            batch of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomBlockGeoSampler(BlockGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        block_size: Union[Tuple[float, float], float],
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
            length: number of samples per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset, roi)

        self.size = _to_tuple(size)
        self.block_size = _to_tuple(block_size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

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

    def __iter__(self) -> Iterator[BoundingBox]:
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
            bounding_box = get_random_bounding_box_from_grid(
                bounds, self.size, self.res, self.block_size
            )

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length


class RandomBlockBatchGeoSampler(BlockBatchGeoSampler):
    """Samples batches of elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        block_size: Union[Tuple[float, float], float],
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
        self.block_size = _to_tuple(block_size)

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
                bounding_box = get_random_bounding_box_from_grid(
                    bounds, self.size, self.res, self.block_size
                )
                batch.append(bounding_box)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches in a single epoch.

        Returns:
            number of batches in an epoch
        """
        return self.length // self.batch_size


class TripletBlockBatchGeoSampler(BlockBatchGeoSampler):
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
        block_size: Union[Tuple[float, float], float] = 128,
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
        self.size = _to_tuple(size)
        self.block_size = _to_tuple(block_size)
        self.neighborhood = _to_tuple(neighborhood)
        self.res = dataset.res

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.block_size = (
                self.block_size[0] * self.res,
                self.block_size[1] * self.res,
            )
            self.neighborhood = (
                self.neighborhood[0] * self.res,
                self.neighborhood[1] * self.res,
            )

        super().__init__(dataset, roi)

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
                anchor_bbox = get_random_bounding_box_from_grid(
                    anchor_bounds, self.size, self.res, self.block_size
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

                neighborhood_block_bounds = get_bounds_from_grid(
                    anchor_bounds, neighborhood_bounds, self.size, self.block_size
                )

                neighbor_bbox = get_random_bounding_box_from_grid(
                    neighborhood_block_bounds,
                    self.size,
                    self.block_size,
                    neighborhood_bounds,
                )

                neighbor_bbox = anchor_bbox

                distant_hit = self.hits[distant_idx]
                distant_bounds = BoundingBox(*distant_hit.bounds)

                distant_bbox = get_random_bounding_box_from_grid(
                    distant_bounds, self.size, self.block_size
                )

                for _ in range(3):
                    if distant_bbox not in neighborhood_bounds:
                        break

                    distant_idx = torch.multinomial(self.areas, 1)
                    distant_hit = self.hits[distant_idx]
                    distant_bounds = BoundingBox(*distant_hit.bounds)

                    distant_bbox = get_random_bounding_box_from_grid(
                        distant_bounds, self.size, self.block_size
                    )

                distant_bbox = anchor_bbox

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


class TripletTileBlockBatchGeoSampler(BlockBatchGeoSampler):
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
        block_size: int = 128,
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
        self.size = _to_tuple(size)
        self.block_size = _to_tuple(block_size)
        self.neighborhood = _to_tuple(neighborhood)
        self.res = dataset.res

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.block_size = (
                self.block_size[0] * self.res,
                self.block_size[1] * self.res,
            )
            self.neighborhood = (
                self.neighborhood[0] * self.res,
                self.neighborhood[1] * self.res,
            )

        super().__init__(dataset, self.block_size, roi)

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
            anchor_idx, distant_idx = torch.multinomial(self.areas, 2, replacement=True)
            anchor_hit = self.hits[anchor_idx]
            anchor_bounds = BoundingBox(*anchor_hit.bounds)

            distant_hit = self.hits[distant_idx]
            distant_bounds = BoundingBox(*distant_hit.bounds)

            batch = []
            for _ in range(self.batch_size):
                # Choose a random index within that tile
                anchor_bbox = get_random_bounding_box_from_grid(
                    anchor_bounds, self.size, self.block_size
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

                # neighborhood_block_bounds = get_bounds_from_grid(
                #     anchor_bounds, neighborhood_bounds, self.size, self.block_size
                # )

                # neighbor_bbox = get_random_bounding_box_from_grid(
                #     neighborhood_block_bounds,
                #     self.size,
                #     self.block_size,
                #     neighborhood_bounds,
                # )

                neighbor_bbox = anchor_bbox
                distant_bbox = anchor_bbox

                # distant_bbox = get_random_bounding_box_from_grid(
                #     distant_bounds, self.size, self.block_size
                # )

                # for _ in range(3):
                #     if distant_bbox not in neighborhood_bounds:
                #         break

                #     distant_idx = torch.multinomial(self.areas, 1)
                #     distant_hit = self.hits[distant_idx]
                #     distant_bounds = BoundingBox(*distant_hit.bounds)

                #     distant_bbox = get_random_bounding_box_from_grid(
                #         distant_bounds, self.size, self.block_size
                #     )

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
