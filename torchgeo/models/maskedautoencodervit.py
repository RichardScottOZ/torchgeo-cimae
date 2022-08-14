"""MaskedVit."""

from functools import lru_cache
from typing import cast

import torch
from kornia.contrib.vit import TransformerEncoderBlock
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, Sequential, init
from torch.nn.parameter import Parameter

from .utils import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed

IN_CHANNELS = {"sentinel2": {"all": 10}, "naip": {"all": 4}}
NUM_CLASSES = {"sentinel2": 17, "naip": 0}


def _init_weights(m: Module) -> None:
    """Initialize the weights."""
    if isinstance(m, Linear):
        init.xavier_uniform_(m.weight)
        if isinstance(m, Linear) and m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, LayerNorm):
        init.constant_(m.bias, 0)
        init.constant_(m.weight, 1.0)


class TransformerEncoder(Module):
    """TransformerEncoder."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a TransformerEncoder."""
        super().__init__()
        self.blocks = Sequential(
            *(
                TransformerEncoderBlock(
                    embed_dim, num_heads, dropout_rate, dropout_attn
                )
                for _ in range(depth)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return cast(Tensor, self.blocks(x))


class EncoderEmbedding(Module):
    """Compute the 2d image patch embedding ready to pass to transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        patch_size: int,
        image_size: int,
        channel_wise: bool = False,
    ) -> None:
        """Initialize the encoder embedding module."""
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.channel_wise = channel_wise
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.embedder = Conv2d(
            input_dim if not channel_wise else 1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.initialize_positional_encodings()

        self.initialize_weights()

    def initialize_positional_encodings(self) -> None:
        """Initialize the positional embeddings."""
        positional_embeddings = get_2d_sincos_pos_embed(
            self.embed_dim, int(self.num_patches**0.5), cls_token=False
        )
        if not self.channel_wise:
            positional_embeddings = positional_embeddings.unsqueeze(0)

        self.positional_embeddings = Parameter(
            positional_embeddings, requires_grad=False
        )

    def initialize_weights(self) -> None:
        """Initialize the weights."""
        w = self.embedder.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    @lru_cache(128)
    def get_channel_encodings(
        self, channels: tuple[int], embed_size: int, device: str | torch.device
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        channel_tokens = channel_tokens.repeat_interleave(
            repeats=self.num_patches, dim=0
        )

        return channel_tokens

    def forward(self, x: Tensor, channels: list[int] = []) -> Tensor:
        """TODO: Docstring."""
        x = self.embedder(x)

        B, H, PW, _ = x.shape
        x = x.view(B, H, PW**2).permute(0, 2, 1)  # BxCxPSxPS -> BxPxH

        x += self.positional_embeddings

        if len(channels):
            x = x.reshape(-1, len(channels) * PW**2, H)
            x += self.get_channel_encodings(tuple(channels), x.shape[-1], x.device)

        return x


class MaskedEncoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.embed_module = EncoderEmbedding(
            in_channels, embed_dim, patch_size, image_size, channel_wise
        )
        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(_init_weights)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> Tensor:
        """Forward pass of the model."""
        x = self.embed_module(x, channels)

        if mask is not None:
            x = x[:, ~mask]

        x = self.encoder(x)
        x = self.norm(x)

        return x


class DecoderEmbedding(Module):
    """TODO: Docstring."""

    def __init__(
        self,
        num_patches: int,
        input_dim: int = 3,
        embed_dim: int = 768,
        channel_wise: bool = False,
    ) -> None:
        """TODO: Docstring."""
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.channel_wise = channel_wise

        self.embedder = Linear(input_dim, embed_dim, bias=True)
        self.mask_token = Parameter(
            -torch.ones(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.initialize_positional_encodings()
        self.apply(_init_weights)

    def initialize_positional_encodings(self) -> None:
        """Initialize the positional embeddings."""
        positional_embeddings = get_2d_sincos_pos_embed(
            self.embed_dim, int(self.num_patches**0.5), cls_token=False
        )

        self.positional_embeddings = Parameter(
            positional_embeddings, requires_grad=False
        )

    @lru_cache(128)
    def get_channel_encodings(
        self, channels: tuple[int], embed_size: int, device: str | torch.device
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )
        channel_tokens = channel_tokens.repeat_interleave(
            repeats=self.num_patches, dim=0
        )

        return channel_tokens

    def select_from_mask(self, x: Tensor, mask: Tensor, channels: list[int]) -> Tensor:
        """Select the elements given a mask."""
        B, _, _ = x.shape

        x_data = x
        x = self.mask_token.repeat(B, len(mask) // self.num_patches, 1)
        x[:, ~mask] = x_data

        # Add mask_tokens if output dim > input dim
        if len(mask) < self.num_patches * len(channels):
            x_expand = self.mask_token.repeat(
                B, len(channels) - (len(mask) // self.num_patches), 1
            )
            x = torch.cat([x, x_expand], dim=1)

        return x

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> Tensor:
        """TODO: Docstring."""
        x = self.embedder(x)

        if mask is not None:
            x = self.select_from_mask(x, mask, channels)

        x += self.positional_embeddings.repeat(len(channels) or 1, 1)

        if len(channels):
            x += self.get_channel_encodings(tuple(channels), x.shape[-1], x.device)

        return x


class MaskedDecoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        num_patches: int,
        image_size: int,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        self.embed_module = DecoderEmbedding(
            num_patches, in_channels, embed_dim, channel_wise
        )

        self.decoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )

        self.norm = LayerNorm(embed_dim)

        out_features = patch_size**2
        if not channel_wise:
            out_features *= out_channels
        self.predictor = Linear(embed_dim, out_features, bias=True)  # decoder to patch

        self.apply(_init_weights)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> Tensor:
        """Forward pass of the model."""
        x = self.embed_module(x, mask, channels)

        x = self.decoder(x)
        x = self.norm(x)

        x = self.predictor(x)

        return x


class MaskedAutoencoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        sensor: str,
        bands: str,
        image_size: int,
        patch_size: int = 16,
        channel_wise: bool = False,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_dropout_rate: float = 0.0,
        decoder_dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        if channel_wise:
            embed_dim = (embed_dim // 2) * 2
            decoder_embed_dim = (decoder_embed_dim // 2) * 2

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            in_channels=IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dropout_attn=dropout_attn,
        )

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.embed_module.num_patches,
            image_size=image_size,
            in_channels=embed_dim,
            out_channels=4,  # IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            dropout_rate=decoder_dropout_rate,
            dropout_attn=decoder_dropout_attn,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        encoder_channels: list[int] = [],
        decoder_channels: list[int] = [],
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass of the model."""
        latent = self.encoder(x, mask, encoder_channels)
        pred = self.decoder(latent, mask, decoder_channels)

        return cast(Tensor, pred)


class MaskedViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        sensor: str,
        bands: str,
        image_size: int,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=IN_CHANNELS[sensor][bands],
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dropout_attn=dropout_attn,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass of the model."""
        embedding = self.encoder(x, mask)

        return cast(Tensor, embedding)
