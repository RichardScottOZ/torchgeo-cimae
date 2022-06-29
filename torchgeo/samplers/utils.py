# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

from decimal import Decimal
from math import ceil, floor
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch

from ..datasets import BoundingBox


def _to_tuple(value: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value


def get_random_bounding_box(
    bounds: BoundingBox, size: Union[Tuple[float, float], float], res: float
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size = _to_tuple(size)

    width = (bounds.maxx - bounds.minx - t_size[1]) // res
    height = (bounds.maxy - bounds.miny - t_size[0]) // res

    minx = bounds.minx
    miny = bounds.miny

    # random.randrange crashes for inputs <= 0
    if width > 0:
        minx += torch.rand(1).item() * width * res
    if height > 0:
        miny += torch.rand(1).item() * height * res

    maxx = minx + t_size[1]
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query


def get_random_bounding_box_from_grid(
    block_bounds: BoundingBox,
    size: Union[Tuple[float, float], float],
    res: float,
    block_size: Union[Tuple[float, float], float],
    bounds: Optional[BoundingBox] = None,
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample

    Returns:
        randomly sampled bounding box from the extent of the input
    """

    # TODO: CURRENT -> Wtf is this fucking precision
    t_size = _to_tuple(size)
    t_block_size = _to_tuple(block_size)

    if t_size > t_block_size:
        raise ValueError("Size is larger than block size.")

    minx = block_bounds.minx
    maxy = block_bounds.maxy

    num_blocks_x = (block_bounds.maxx - block_bounds.minx) // res // t_block_size[0]
    num_blocks_y = (block_bounds.maxy - block_bounds.miny) // res // t_block_size[1]

    if num_blocks_x > 0:
        block_offset_x = floor(torch.rand(1).item() * num_blocks_x)
        minx += block_offset_x * t_block_size[0] * res

    if num_blocks_y > 0:
        block_offset_y = floor(torch.rand(1) * num_blocks_y)
        maxy -= block_offset_y * t_block_size[1] * res

    # TODO: Check again
    if bounds is not None:
        block = BoundingBox(
            minx,
            minx + t_block_size[0],
            maxy - t_block_size[1],
            maxy,
            block_bounds.mint,
            block_bounds.maxt,
        )
        block_bounds = block & bounds
        t_block_size = (
            block_bounds.maxx - block_bounds.minx,
            block_bounds.maxy - block_bounds.miny,
        )

        minx = block_bounds.minx
        maxy = block_bounds.maxy

    max_x_offset = t_block_size[0] - (t_size[0] // res)
    max_y_offset = t_block_size[1] - (t_size[1] // res)

    x_offset = torch.rand(1).item() * max_x_offset
    y_offset = torch.rand(1).item() * max_y_offset

    minx += x_offset * res
    maxy -= y_offset * res

    maxx = minx + t_size[0]
    miny = maxy - t_size[1]

    query = BoundingBox(minx, maxx, miny, maxy, block_bounds.mint, block_bounds.maxt)
    return query


def get_bounds_from_grid(
    bounds: BoundingBox,
    bounding_box: BoundingBox,
    size: Union[Tuple[float, float], float],
    block_size: Union[Tuple[float, float], float],
) -> BoundingBox:
    """get_bounds_from_grid."""
    t_size = _to_tuple(size)
    t_block_size = _to_tuple(block_size)

    minx = (
        bounds.minx
        + floor((bounding_box.minx - bounds.minx) / t_block_size[0]) * t_block_size[0]
    )

    maxy = (
        bounds.maxy
        - floor((bounds.maxy - bounding_box.maxy) / t_block_size[1]) * t_block_size[1]
    )

    maxx = (
        bounds.maxx
        - floor((bounds.maxx - bounding_box.maxx) / t_block_size[0]) * t_block_size[0]
    )
    miny = (
        bounds.miny
        + floor((bounding_box.miny - bounds.miny) / t_block_size[1]) * t_block_size[1]
    )

    if minx != bounds.minx and bounding_box.minx - minx < t_size[0]:
        minx += t_block_size[0]

    if maxx != bounds.maxx and maxx - bounding_box.maxx < t_size[0]:
        maxx -= t_block_size[0]

    if miny != bounds.miny and bounding_box.miny - miny < t_size[1]:
        miny += t_block_size[1]

    if maxy != bounds.maxy and maxy - bounding_box.maxy < t_size[1]:
        maxy -= t_block_size[1]

    return BoundingBox(minx, maxx, miny, maxy, bounding_box.mint, bounding_box.maxt)
