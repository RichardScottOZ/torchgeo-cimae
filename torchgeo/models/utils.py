"""Common model utilities."""

from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, init
from torch.nn.parameter import Parameter


# TODO: Attribution (Facebook)
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: int,
    cls_token: bool = False,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Get 2D sine-cosine position embedding.
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_h = torch.arange(grid_size, dtype=torch.float, device=device)
    grid_w = torch.arange(grid_size, dtype=torch.float, device=device)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")  # here w goes first

    grid_array = torch.stack(grid, dim=0)
    grid_array = grid_array.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid_array, device)

    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([1, embed_dim], dtype=torch.float, device=device), pos_embed],
            dim=0,
        )

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Get 2D sine-cosine position embedding from grid."""
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0], device=device
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1], device=device
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    """Get 1D sine-cosine position embedding from grid.

    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)

    return emb


def _init_weights(m: Module) -> None:
    """Initialize the weights."""
    if isinstance(m, Linear):
        init.xavier_uniform_(m.weight)
        if isinstance(m, Linear) and m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)
    elif isinstance(m, Conv2d):
        w = m.weight.data
        init.xavier_uniform_(w.view([w.shape[0], -1]))


@lru_cache(128)
def get_positional_encodings(
    embed_dim: int,
    num_patches: int,
    channel_wise: bool,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Initialize the positional embeddings."""
    positional_embeddings = get_2d_sincos_pos_embed(
        embed_dim, int(num_patches**0.5), cls_token=False, device=device
    )

    if not channel_wise:
        return positional_embeddings.unsqueeze(0)

    return positional_embeddings


@lru_cache(128)
def get_channel_encodings(
    channels: tuple[int], num_patches: int, embed_size: int, device: str | torch.device
) -> Tensor:
    """Get the channel encodings for the given channels."""
    channel_encoding = get_1d_sincos_pos_embed_from_grid(
        embed_size,
        torch.tensor(channels, dtype=torch.float, device=device),
        device=device,
    )

    channel_encoding = channel_encoding.repeat_interleave(repeats=num_patches, dim=0)

    return channel_encoding


@lru_cache(128)
def get_mask_token(
    batch_size: int,
    num_patches: int,
    embed_dim: int,
    channel_wise: bool = False,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Get the mask token."""
    embed_token = -torch.ones(batch_size, num_patches, embed_dim, device=device)
    embed_token += get_positional_encodings(
        embed_dim, num_patches, channel_wise, device=device
    )
    return embed_token
