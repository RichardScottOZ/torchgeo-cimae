"""Common model utilities."""

from typing import cast

import numpy as np
import numpy.typing as npt


# TODO: Attribution (Facebook)
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, channels: list[int] = [], cls_token: bool = False
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

    if len(channels) > 0:
        channel_embed_dim = embed_dim // 3
        embed_dim = embed_dim // 3 * 2

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_array)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    if len(channels) > 0:
        channel_embed = get_1d_sincos_pos_embed_from_grid(
            channel_embed_dim, np.array(channels)
        )
        channel_embed = np.expand_dims(channel_embed, axis=1)
        channel_embed = np.repeat(channel_embed, grid_size * grid_size, axis=1)

        pos_embed = np.expand_dims(pos_embed, axis=0)
        pos_embed = pos_embed.repeat(len(channels), 0)

        pos_embed = np.concatenate([pos_embed, channel_embed], axis=-1)

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

    return cast(npt.NDArray[np.float32], emb)
