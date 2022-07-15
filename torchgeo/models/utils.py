"""Common model utilities."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def random_masking(
    x: Tensor,
    ids_keep: Tensor,
    random_mask_ratio: float,
    random_mask_probability: float,
    **kwargs: Any,
) -> Tensor:
    """Perform per-sample random masking by per-sample shuffling.

    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    if torch.rand(1) > random_mask_probability:
        return ids_keep

    _, P, _ = x.shape

    len_keep = int(P * (1 - random_mask_ratio))
    ids_keep = ids_keep[torch.randperm(len(ids_keep))[:len_keep]]

    return ids_keep


def focal_masking(
    x: Tensor,
    ids_keep: Tensor,
    focal_mask_ratio: float,
    focal_mask_probability: float,
    **kwargs: Any,
) -> Tensor:
    """Perform focal masking."""
    if torch.rand(1) > focal_mask_probability:
        return ids_keep

    _, P, _ = x.shape

    side = int(P**0.5)
    focal_ratio = 1 - focal_mask_ratio

    # Generate focal mask
    center = side / 2
    half_scaled = side // 2 * focal_ratio
    low, high = round(center - half_scaled), round(center + half_scaled)

    mask = torch.arange(P, device=x.device).reshape((side, side))
    ids = mask[low:high, low:high].flatten()

    ids_keep = ids_keep[(ids_keep.view(1, -1) == ids.view(-1, 1)).any(dim=0)]

    return ids_keep


# TODO: Attribution (Facebook)
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False
) -> npt.NDArray[np.float32]:
    """Get 2D sine-cosine position embedding.

    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid_array = np.stack(grid, axis=0)

    grid_array = grid_array.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_array)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Get 2D sine-cosine position embedding from grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1, dtype=np.float32)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Get 1D sine-cosine position embedding from grid.

    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
