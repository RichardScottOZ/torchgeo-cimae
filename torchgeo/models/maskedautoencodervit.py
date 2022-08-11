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

        self.embedder = Conv2d(
            input_dim if not channel_wise else 1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.channels = torch.arange(input_dim).tolist() if channel_wise else []
        self.initialize_positional_encodings()

        self.initialize_weights()

    def initialize_positional_encodings(self) -> None:
        """Initialize the positional embeddings."""
        self.num_patches = (self.image_size // self.patch_size) ** 2

        positional_embeddings = get_2d_sincos_pos_embed(
            self.embed_dim,
            int(self.num_patches**0.5),
            cls_token=False,
            channels=self.channels,
        )
        if not len(self.channels):
            positional_embeddings = positional_embeddings.unsqueeze(0)

        self.positional_embeddings = Parameter(
            positional_embeddings, requires_grad=False
        )

    def initialize_weights(self) -> None:
        """Initialize the weights."""
        w = self.embedder.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: Tensor) -> Tensor:
        """TODO: Docstring."""
        x = self.embedder(x)

        B, C, PW, _ = x.shape
        x = x.view(B, C, PW**2).permute(0, 2, 1)  # BxCxPSxPS -> BxPxH

        if self.channel_wise:
            x += self.positional_embeddings.repeat(B // self.input_dim, 1, 1)
            x = x.reshape(-1, self.input_dim * PW**2, C)
        else:
            x += self.positional_embeddings

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

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize the weights."""
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @lru_cache(128)
    def get_channel_tokens(
        self,
        channels: tuple[int],
        batch_size: int,
        embed_size: int,
        device: str | torch.device,
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        channel_tokens = channel_tokens.repeat(batch_size, 1, 1)

        return channel_tokens

    def add_channel_encoding(self, x: Tensor, channels: list[int]) -> Tensor:
        """TODO: Docstring."""
        B, PS, H = x.shape
        device = x.device
        channel_tokens = self.get_channel_tokens(tuple(channels), B, H, device)
        x += channel_tokens.repeat_interleave(PS // len(channels), dim=1)

        return x

    @lru_cache(128)
    def get_channel_tokens_with_position(
        self,
        channels: tuple[int],
        batch_size: int,
        embed_size: int,
        device: str | torch.device,
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size // 2,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        positions = torch.arange(len(channels), dtype=torch.float, device=device)
        position_encodings = get_1d_sincos_pos_embed_from_grid(
            embed_size // 2, positions, device=device
        )

        channel_tokens = torch.cat([channel_tokens, position_encodings], dim=1)
        channel_tokens = channel_tokens.repeat(batch_size, 1, 1)

        return channel_tokens

    def cat_channel_encoding(self, x: Tensor, channels: list[int]) -> Tensor:
        """TODO: Docstring."""
        B, _, H = x.shape
        device = x.device
        channel_tokens = self.get_channel_tokens_with_position(
            tuple(channels), B, H, device
        )
        x = torch.cat([channel_tokens, x], dim=1)

        return x

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> Tensor:
        """Forward pass of the model."""
        x = self.embed_module(x)

        # TODO: Remove
        channels = []

        if mask is not None:
            B, _, H = x.shape
            x = x[:, ~mask]

        if len(channels):
            # x = self.cat_channel_encoding(x, channels)
            x = self.add_channel_encoding(x, channels)

        x = self.encoder(x)

        # Remove channel tokens
        # if len(channels):
        #     x = x[:, len(channels) :]

        x = self.norm(x)

        return x


class DecoderEmbedding(Module):
    """TODO: Docstring."""

    def __init__(
        self,
        num_patches: int,
        input_dim: int = 3,
        embed_dim: int = 768,
        in_channels: int = 3,
        channel_wise: bool = False,
    ) -> None:
        """TODO: Docstring."""
        super().__init__()

        self.num_patches = num_patches
        if channel_wise:
            self.num_patches *= in_channels
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.channel_wise = channel_wise

        self.embedder = Linear(input_dim, embed_dim, bias=True)
        self.mask_token = Parameter(-torch.ones(1, 1, embed_dim), requires_grad=False)
        self.channels = torch.arange(in_channels).tolist() if channel_wise else []
        self.initialize_positional_encodings()
        self.apply(self._init_weights)

    def initialize_positional_encodings(self) -> None:
        """Initialize the positional embeddings."""
        positional_embeddings = get_2d_sincos_pos_embed(
            self.embed_dim,
            int((self.num_patches // self.in_channels) ** 0.5),
            cls_token=False,
            channels=self.channels,
        )
        positional_embeddings = positional_embeddings.flatten(0, 1).unsqueeze(0)

        self.positional_embeddings = Parameter(
            positional_embeddings, requires_grad=False
        )

    def _init_weights(self, m: Module) -> None:
        """Initialize the weights."""
        if isinstance(m, Linear):
            init.xavier_uniform_(m.weight)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> tuple[Tensor, Tensor]:
        """TODO: Docstring."""
        x = self.embedder(x)
        latent = x.clone()
        B, _, _ = x.shape

        if mask is not None:
            x_data = x
            x = self.mask_token.repeat(B, self.num_patches, 1)
            x[:, ~mask] = x_data

        x += self.positional_embeddings

        return x, latent


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

        self.embed_module = DecoderEmbedding(
            num_patches, in_channels, embed_dim, out_channels, channel_wise
        )

        self.decoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )

        out_features = patch_size**2
        if not channel_wise:
            out_features *= out_channels
        self.predictor = Linear(embed_dim, out_features, bias=True)  # decoder to patch

        self.norm = LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m: Module) -> None:
        """Initialize the weights."""
        if isinstance(m, Linear):
            init.xavier_uniform_(m.weight)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @lru_cache(128)
    def get_channel_tokens(
        self,
        channels: tuple[int],
        batch_size: int,
        embed_size: int,
        device: str | torch.device,
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        channel_tokens = channel_tokens.repeat(batch_size, 1, 1)

        return channel_tokens

    def add_channel_encoding(self, x: Tensor, channels: list[int]) -> Tensor:
        """TODO: Docstring."""
        B, PS, H = x.shape
        device = x.device
        channel_tokens = self.get_channel_tokens(tuple(channels), B, H, device)
        x += channel_tokens.repeat_interleave(PS // len(channels), dim=1)

        return x

    @lru_cache(128)
    def get_channel_tokens_with_position(
        self,
        channels: tuple[int],
        batch_size: int,
        embed_size: int,
        device: str | torch.device,
    ) -> Tensor:
        """TODO: Docstring."""
        channel_tokens = get_1d_sincos_pos_embed_from_grid(
            embed_size // 2,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        positions = torch.arange(len(channels), dtype=torch.float, device=device)
        position_encodings = get_1d_sincos_pos_embed_from_grid(
            embed_size // 2, positions, device=device
        )

        channel_tokens = torch.cat([channel_tokens, position_encodings], dim=1)
        channel_tokens = channel_tokens.repeat(batch_size, 1, 1)

        return channel_tokens

    def cat_channel_encoding(self, x: Tensor, channels: list[int]) -> Tensor:
        """TODO: Docstring."""
        B, _, H = x.shape
        device = x.device
        channel_tokens = self.get_channel_tokens_with_position(
            tuple(channels), B, H, device
        )
        x = torch.cat([channel_tokens, x], dim=1)

        return x

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the model."""
        x, latent = self.embed_module(x, mask, channels)

        if len(channels):
            # x = self.cat_channel_encoding(x, channels)
            x = self.add_channel_encoding(x, channels)

        x = self.decoder(x)

        # Remove channel tokens
        # if len(channels):
        #     x = x[:, len(channels) :]

        x = self.norm(x)
        x = self.predictor(x)

        return x, latent


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
        return_latent: bool = False,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.return_latent = return_latent
        self.image_size = image_size
        self.patch_size = patch_size

        if channel_wise:
            embed_dim = (embed_dim // 3 // 2 // 2) * 3 * 2 * 2
            decoder_embed_dim = (decoder_embed_dim // 3 // 2 // 2) * 3 * 2 * 2

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
            out_channels=IN_CHANNELS[sensor][bands],
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
        pred, latent = self.decoder(latent, mask, decoder_channels)

        if self.return_latent:
            return pred, latent

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
