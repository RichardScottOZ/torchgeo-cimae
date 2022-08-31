"""Common model utilities."""
from functools import lru_cache

import torch
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, init


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


def init_weights(m: Module) -> None:
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


def get_channel_encodings(
    embed_dim: int, channels: tuple[int], num_patches: int, device: str | torch.device
) -> Tensor:
    """Get the channel encodings for the given channels."""
    channels_sorted, channel_order = torch.tensor(
        channels, dtype=torch.float, device=device
    ).sort()

    channel_encoding = get_channel_encoding(
        embed_dim, tuple(channels_sorted.tolist()), num_patches, device
    )

    channel_encoding = channel_encoding.view(len(channels), num_patches, embed_dim)[
        channel_order.argsort()
    ].flatten(0, 1)

    return channel_encoding


def get_channel_encoding(
    embed_dim: int, channels: tuple[int], num_patches: int, device: str | torch.device
) -> Tensor:
    """Get the channel encodings for the given channels."""
    channel_encoding = get_1d_sincos_pos_embed_from_grid(
        embed_dim,
        torch.tensor(channels, dtype=torch.float, device=device),
        device=device,
    )

    channel_encoding = channel_encoding.repeat_interleave(repeats=num_patches, dim=0)

    return channel_encoding


@lru_cache(128)
def get_encoding(
    embed_dim: int,
    num_patches: int,
    channel_wise: bool = False,
    embed_enc: bool = False,
    return_pos_enc: bool = False,
    channels: tuple[int] | None = None,
    is_embed: bool | None = None,
    device: str | torch.device = "cpu",
    channel_ratio_div: int = 4,
    mask_ratio_div: int = 16,
) -> Tensor:
    """Get the encoding for the given channels."""
    encoding = torch.zeros((num_patches, embed_dim), dtype=torch.float, device=device)

    pos_embed_dim = embed_dim
    channel_embed_dim = embed_dim
    embed_embed_dim = embed_dim

    if channel_wise:
        channel_embed_dim //= channel_ratio_div
        pos_embed_dim -= channel_embed_dim

    if embed_enc:
        embed_embed_dim = embed_dim // mask_ratio_div
        channel_embed_dim -= embed_embed_dim

    if return_pos_enc:
        encoding[..., :pos_embed_dim] = get_positional_encodings(
            pos_embed_dim, num_patches, channel_wise, device
        )

    if channel_wise and channels is not None:
        channel_encoding = get_channel_encodings(
            channel_embed_dim, channels, num_patches, device=device
        )
        encoding = encoding.repeat(len(channels), 1)

        encoding[
            ..., pos_embed_dim : pos_embed_dim + channel_embed_dim
        ] = channel_encoding

    if embed_enc and is_embed is not None:
        mask_encoding = get_channel_encoding(
            embed_embed_dim, (int(is_embed),), num_patches, device=device
        )

        if channel_wise and channels is not None:
            mask_encoding = mask_encoding.repeat(len(channels), 1)

        encoding[..., -embed_embed_dim:] = mask_encoding

    return encoding


@lru_cache(10)
def get_mask_token(
    embed_dim: int,
    num_patches: int,
    channel_enc: bool = False,
    embed_enc: bool = False,
    channel_wise: bool = False,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Get the mask token."""
    mask_token = -torch.ones(num_patches, embed_dim, device=device)
    mask_token += get_encoding(
        embed_dim=embed_dim,
        num_patches=num_patches,
        channel_wise=channel_wise,
        embed_enc=True,
        return_pos_enc=True,
        channels=(0,) if channel_enc else None,
        is_embed=True if embed_enc else None,
        device=device,
    )
    return mask_token


def reduce_mask_token(
    x: Tensor,
    mask: Tensor,
    mask_token: Tensor,
    num_patches: int,
    keep_unreduced: bool = False,
) -> Tensor:
    """Reduce the embed token by using the values not masked in place."""
    _, PS, H = x.shape
    mask = mask.view(-1, num_patches)  # (C, P)

    visible_pos_indices = (~mask).nonzero()[:, 1]
    sorted_visible, indices = visible_pos_indices.sort(stable=True)
    _, counts = sorted_visible.unique_consecutive(return_counts=True)  # type: ignore
    counts = counts.cumsum(dim=0)[:-1]
    counts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=x.device), counts])

    mask_token[:, visible_pos_indices[indices[counts]]] = x[:, indices[counts]]
    # Mark as mask tokens
    mask_token[:, visible_pos_indices[indices[counts]]] += get_encoding(
        embed_dim=H,
        num_patches=len(counts),
        embed_enc=True,
        is_embed=True,
        device=x.device,
    )

    if keep_unreduced:
        all_patches = torch.arange(PS, device=x.device)
        unreduced_indices = all_patches[
            (all_patches != indices[counts].view(-1, 1)).all(dim=0)
        ]
        mask_token = torch.cat([mask_token, x[:, unreduced_indices]], dim=1)

    return mask_token


def add_embed_encoding(
    x: Tensor, mask: Tensor, num_patches: int, channel_wise: bool
) -> Tensor:
    """Get the reduced and unreduced indices for the given mask."""
    *_, H = x.shape
    mask = mask.view(-1, num_patches)  # (C, P)

    visible_pos_indices = (~mask).nonzero()[:, 1]
    sorted_visible, indices = visible_pos_indices.sort(stable=True)
    _, counts = sorted_visible.unique_consecutive(return_counts=True)  # type: ignore
    counts = counts.cumsum(dim=0)[:-1]
    counts = torch.cat([torch.zeros(1, dtype=counts.dtype, device=x.device), counts])

    # Mark as embed tokens
    x[:, indices[counts]] += get_encoding(
        embed_dim=H,
        num_patches=len(counts),
        embed_enc=True,
        is_embed=True,
        device=x.device,
    )

    return x


def get_encoding_masked(
    mask: Tensor, num_patches: int, embed_dim: int, channel_wise: bool
) -> Tensor:
    """Get the encoding for the input given the corresponding mask."""
    return get_encoding(
        embed_dim,
        num_patches,
        channel_wise,
        embed_enc=True,
        return_pos_enc=True,
        device=mask.device,
    ).repeat(len(mask) // num_patches, 1)[~mask]
