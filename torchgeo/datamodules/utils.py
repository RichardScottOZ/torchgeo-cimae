# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common datamodule utilities."""

from typing import Any, List, Optional, Sequence, Tuple, Union

from torch import float64, linspace, randperm
from torch.utils.data import Subset, TensorDataset, random_split

from ..datasets import BoundingBox, VisionDataset


def dataset_split(
    dataset: Union[TensorDataset, VisionDataset],
    val_pct: float,
    test_pct: Optional[float] = None,
) -> List[Subset[Any]]:
    """Split a torch Dataset into train/val/test sets.

    If ``test_pct`` is not set then only train and validation splits are returned.

    Args:
        dataset: dataset to be split into train/val or train/val/test subsets
        val_pct: percentage of samples to be in validation set
        test_pct: (Optional) percentage of samples to be in test set

    Returns:
        a list of the subset datasets. Either [train, val] or [train, val, test]
    """
    if test_pct is None:
        val_length = int(len(dataset) * val_pct)
        train_length = len(dataset) - val_length
        return random_split(dataset, [train_length, val_length])
    else:
        val_length = int(len(dataset) * val_pct)
        test_length = int(len(dataset) * test_pct)
        train_length = len(dataset) - (val_length + test_length)
        return random_split(dataset, [train_length, val_length, test_length])


def roi_split_half(roi: BoundingBox, **kwargs: Any) -> Sequence[BoundingBox]:
    """Splits the roi based on the midpoint of the `BoundingBox`.

    Args:
        roi: Used to specify the region of interest.

    Returns:
        A list of train, validation, and test `BoundingBox`es.
    """
    midx = roi.minx + (roi.maxx - roi.minx) / 2
    midy = roi.miny + (roi.maxy - roi.miny) / 2
    train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
    val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
    test_roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

    return train_roi, val_roi, test_roi


def roi_split_grid(roi: BoundingBox, **kwargs: Any) -> Sequence[Sequence[BoundingBox]]:
    """The roi_split_grid function splits a bounding box into a grid of smaller bounding boxes.

    Args:
        roi: Used to specify the region of interest.

    Returns:
        A list of bounding boxes of the grid.
    """
    number_of_cells: Tuple[int, int] = _to_tuple(kwargs["number_of_cells"])

    xs = linspace(roi.minx, roi.maxx, number_of_cells[0] + 1, dtype=float64)
    ys = linspace(roi.miny, roi.maxy, number_of_cells[1] + 1, dtype=float64)

    # TODO: Replace for loop with something??
    bounding_boxes = []
    miny = float(ys[0])

    for maxy in ys[1:]:
        minx = float(xs[0])
        maxy = float(maxy)

        for maxx in xs[1:]:
            maxx = float(maxx)

            bbox = BoundingBox(minx, maxx, miny, maxy, roi.mint, roi.maxt)
            bounding_boxes.append(bbox)

            minx = maxx
        miny = maxy

    val_pct: float = kwargs.get("val_pct", 0.0)
    test_pct: float = kwargs.get("test_pct", 0.0)

    indices = randperm(len(bounding_boxes))

    train_length = int(len(bounding_boxes) * (1 - val_pct - test_pct))
    val_length = int(len(bounding_boxes) * val_pct)

    train_indices = indices[:train_length].tolist()
    val_indices = indices[train_length : (train_length + val_length)].tolist()
    test_indices = indices[(train_length + val_length) :].tolist()

    train_roi = [bounding_boxes[idx] for idx in train_indices]
    val_roi = [bounding_boxes[idx] for idx in val_indices]
    test_roi = [bounding_boxes[idx] for idx in test_indices]

    return train_roi, val_roi, test_roi


def _to_tuple(value: Union[Tuple[int, int], int]) -> Tuple[int, int]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (int, int)):
        return (value, value)
    else:
        return value
