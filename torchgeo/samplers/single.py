# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
from typing import Callable, Iterable, Iterator, Optional, Tuple, Union

import torch
from rtree.index import Index, Property
from torch.utils.data import Sampler

from ..datasets import BoundingBox, GeoDataset
from .constants import Units
from .utils import _to_tuple, get_random_bounding_box

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    def __init__(self, dataset: GeoDataset, roi: Optional[Union[BoundingBox, Sequence[BoundingBox]]] = None) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest or multiple region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
        """
        if not isinstance(roi, Sequence):
            roi = [roi]

        if None in roi:
            self.index = dataset.index
            roi = BoundingBox(*self.index.bounds)            
        else:
            for region in roi:
                self.index = Index(interleaved=False, properties=Property(dimension=3))
                hits = dataset.index.intersection(tuple(region), objects=True)
                for hit in hits:
                    bbox = BoundingBox(*hit.bounds) & region
                    self.index.insert(hit.id, tuple(bbox), hit.object)

        self.res = dataset.res
        self.roi = roi

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
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
            length: number of random samples to draw per epoch
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

        self.length = length
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
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
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` and ``stride`` are in pixel or CRS units

        .. versionchanged:: 0.3
           Added ``units`` parameter, changed default to pixel units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.stride = (self.stride[0] * self.res, self.stride[1] * self.res)

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)
            if (
                bounds.maxx - bounds.minx > self.size[1]
                and bounds.maxy - bounds.miny > self.size[0]
            ):
                self.hits.append(hit)

        self.length: int = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1]) + 1
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1]) + 1

            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


class PreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.VisionDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch

        .. versionadded:: 0.3
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            self.hits.append(hit)

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm

        for idx in generator(len(self)):
            yield BoundingBox(*self.hits[idx].bounds)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.hits)


class TripletGeoSampler(GeoSampler):
    """Samples triplets from a region of interest randomly."""

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        neighborhood: int,
        length: int,
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
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
            neighborhood: size of positive sampling neighborhood
            length: number of random samples to draw per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """
        super().__init__(dataset, roi)
        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            self.size = (self.size[0] * self.res, self.size[1] * self.res)

        self.neighborhood = neighborhood
        self.length = length
        self.hits = []
        areas = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
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
        """Return triplet indices of a dataset.

        Returns:
            list of (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            anchor_bbox = get_random_bounding_box(bounds, self.size, self.res)

            mid_x, mid_y = (
                anchor_bbox.minx + (anchor_bbox.maxx - anchor_bbox.minx) / 2,
                anchor_bbox.miny + (anchor_bbox.maxy - anchor_bbox.miny) / 2,
            )

            # Neighborhood based on midpoint of anchor
            # Contains positive midpoint => +/- half the patch size
            neighborhood_bounds = BoundingBox(
                max(bounds.minx, mid_x - self.neighborhood - self.size[0] // 2),
                min(bounds.maxx, mid_x + self.neighborhood + self.size[0] // 2),
                max(bounds.miny, mid_y - self.neighborhood - self.size[1] // 2),
                min(bounds.maxy, mid_y + self.neighborhood + self.size[1] // 2),
                bounds.mint,
                bounds.maxt,
            )

            neighbor_bbox = get_random_bounding_box(
                neighborhood_bounds, self.size, self.res
            )

            # Choose another random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]
            bounds = BoundingBox(*hit.bounds)

            distant_bbox = get_random_bounding_box(bounds, self.size, self.res)
            while distant_bbox in neighborhood_bounds:
                distant_bbox = get_random_bounding_box(bounds, self.size, self.res)

            yield from [anchor_bbox, neighbor_bbox, distant_bbox]

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length
