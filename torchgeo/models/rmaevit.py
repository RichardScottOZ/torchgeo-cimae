"""MaskedVit."""

from typing import cast

import torch
from kornia.contrib.vit import FeedForward, TransformerEncoderBlock
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Module, Sequential
from torch.nn.parameter import Parameter

from .utils import _init_weights, get_channel_encodings, get_positional_encodings

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

    def forward(self, x: Tensor) -> Tensor:
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

        self.num_patches = (image_size // patch_size) ** 2
        self.embedder = Conv2d(
            input_dim if not channel_wise else 1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.positional_encodings = get_positional_encodings(
            embed_dim, self.num_patches, channel_wise
        )

        self.apply(_init_weights)

    def forward(self, x: Tensor, channels: list[int] = []) -> Tensor:
        """Forward pass of the encoder embedding module.

        First embed the image to patches.
        Secondly, add the positional embeddings for each patch.
        Finally, add the channel embeddings if channels are passed.
        """
        x = self.embedder(x)

        B, H, PW, _ = x.shape
        x = x.view(B, H, PW**2).permute(0, 2, 1)  # BxCxPSxPS -> BxPxH

        x += self.positional_encodings

        if len(channels):
            x = x.reshape(-1, len(channels) * PW**2, H)
            x += get_channel_encodings(
                tuple(channels), self.num_patches, x.shape[-1], x.device
            )

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

        self.embed_dim = embed_dim
        self.embed_module = EncoderEmbedding(
            in_channels, embed_dim, patch_size, image_size, channel_wise
        )
        self.num_patches = self.embed_module.num_patches

        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )
        self.norm = LayerNorm(embed_dim)

        self.positional_encodings = get_positional_encodings(
            self.embed_dim, self.num_patches, channel_wise
        )
        self.embed_token = Parameter(
            -torch.ones(1, self.num_patches, self.embed_dim), requires_grad=False
        )
        self.embed_token += get_positional_encodings(  # type: ignore
            self.embed_dim, self.num_patches, channel_wise
        )

        self.apply(_init_weights)

    # TODO: Rework this to be more efficient.
    def get_embed_token(self, x: Tensor, mask: Tensor, embed_token: Tensor) -> Tensor:
        """Reduce the embed token by using the values not masked in place."""
        PS = len(mask)
        B, *_ = x.shape

        x_full = torch.zeros((B, PS, self.embed_dim), device=x.device, dtype=x.dtype)
        x_full[:, ~mask] = x

        mask = mask.view(-1, self.num_patches)  # (C, P)
        visible_mask = torch.zeros(
            1, self.num_patches, device=x.device, dtype=torch.bool
        )
        mask = torch.cat([mask, visible_mask], dim=0)

        selection_mask = (~mask).T.float().argmax(dim=1)
        selection_mask *= self.num_patches
        selection_mask += torch.arange(self.num_patches, device=x.device)

        x_full = torch.cat((x_full, embed_token), dim=1)
        embed_token = x_full[:, selection_mask]

        return embed_token

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        x = cast(Tensor, item["input"])
        channels = cast(list[int], item.get("encoder_channels", []))
        mask = cast(Tensor | None, item.get("mask", None))

        x = self.embed_module(x, channels)

        if mask is not None:
            x = x[:, ~mask]

        x = self.encoder(x)
        x = self.norm(x)

        if mask is not None:
            B, *_ = x.shape
            embed_token = self.embed_token.repeat(B, 1, 1)
            x = self.get_embed_token(x, mask, embed_token)
        else:
            x = x[:, : self.num_patches]

        return x


class MaskedEmbeddingExpander(Module):
    """Masked embedding module."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        dropout_rate: float = False,
        dropout_attn: float = False,
    ) -> None:
        """Initialize the masked embedding module."""
        super().__init__()

        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )
        self.norm = LayerNorm(embed_dim)

        self.apply(_init_weights)

    def forward(self, item: dict[str, Tensor]) -> Tensor:
        """Forward pass of the embedding module."""
        x = item["latent"]

        x = self.encoder(x)
        x = self.norm(x)

        return x


class DecoderEmbedding(Module):
    """Decoder embedding module."""

    def __init__(self, num_patches: int) -> None:
        """Initialize a new DecoderEmbedding module."""
        super().__init__()

        self.num_patches = num_patches

    def forward(self, x: Tensor, channels: list[int] = []) -> Tensor:
        """Embed the decoder input with channel encodings."""
        if len(channels):
            *_, H = x.shape
            x += get_channel_encodings(tuple(channels), self.num_patches, H, x.device)

        return x


class MaskedDecoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        num_patches: int,
        in_channels: int,
        out_channels: int,
        patch_size: int = 16,
        channel_wise: bool = False,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        out_features = patch_size**2
        if not channel_wise:
            out_features *= out_channels

        self.embed_module = DecoderEmbedding(num_patches)
        self.norm = LayerNorm(in_channels)
        self.predictor = Sequential(
            self.norm, FeedForward(in_channels, in_channels, out_features)
        )

        self.apply(_init_weights)

    def forward(self, item: dict[str, Tensor | list[int]]) -> Tensor:
        """Forward pass of the model."""
        latent = cast(Tensor, item["latent"])
        channels = cast(list[int], item.get("decoder_channels", []))

        output = []
        for channel in channels:
            x = self.embed_module(latent.clone(), [channel])
            x = self.predictor(x)

            output.append(x)

        x = torch.stack(output, dim=1).flatten(1, 2)

        return cast(Tensor, x)


class ReducingMaskedAutoencoderViT(Module):
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
        expander_depth: int = 24,
        expander_num_heads: int = 1,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

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

        self.expander = MaskedEmbeddingExpander(
            embed_dim=embed_dim, depth=expander_depth, num_heads=expander_num_heads
        )

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.num_patches,
            in_channels=embed_dim,
            out_channels=IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
        )

    def forward(self, item: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass of the model."""
        item["latent"] = self.encoder(item)
        item["latent"] = self.expander(item)
        item["pred"] = self.decoder(item)

        return item


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
