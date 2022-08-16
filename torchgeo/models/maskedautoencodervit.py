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


def get_positional_encodings(
    embed_dim: int, num_patches: int, channel_wise: bool
) -> Parameter:
    """Initialize the positional embeddings."""
    positional_embeddings = get_2d_sincos_pos_embed(
        embed_dim, int(num_patches**0.5), cls_token=False
    )
    if not channel_wise:
        positional_embeddings = positional_embeddings.unsqueeze(0)

    return Parameter(positional_embeddings, requires_grad=False)


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

        self.embed_dim = embed_dim
        self.channel_wise = channel_wise
        self.num_patches = (image_size // patch_size) ** 2

        self.embedder = Conv2d(
            input_dim if not channel_wise else 1,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.positional_encodings = get_positional_encodings(
            self.embed_dim, self.num_patches, self.channel_wise
        )
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize the weights."""
        w = self.embedder.weight.data
        init.xavier_uniform_(w.view([w.shape[0], -1]))

    @lru_cache(128)
    def get_channel_encodings(
        self, channels: tuple[int], embed_size: int, device: str | torch.device
    ) -> Tensor:
        """Get the channel encodings for the given channels."""
        channel_encoding = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )

        channel_encoding = channel_encoding.repeat_interleave(
            repeats=self.num_patches, dim=0
        )

        return channel_encoding

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
        embed_token: bool = False,
        embed_token_reduction: bool = False,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.embed_token_reduction = embed_token_reduction

        self.embed_module = EncoderEmbedding(
            in_channels, embed_dim, patch_size, image_size, channel_wise
        )
        self.num_patches = self.embed_module.num_patches
        self.encoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )
        self.norm = LayerNorm(embed_dim)

        self.embed_token: Tensor | None = None
        if embed_token:
            self.embed_token = Parameter(
                -torch.ones(1, self.embed_module.num_patches, embed_dim),
                requires_grad=False,
            )
            self.embed_token += get_positional_encodings(  # type: ignore
                embed_dim, self.embed_module.num_patches, channel_wise
            )

        self.apply(_init_weights)

    def reduce_embed_token(
        self, x: Tensor, mask: Tensor, embed_token: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Reduce the embed token by using the values not masked by the mask in place."""
        x = torch.cat((x, embed_token), dim=1)

        mask = mask.view(-1, self.num_patches)
        visible_mask = torch.zeros(
            1, self.num_patches, device=x.device, dtype=torch.bool
        )
        mask = torch.cat([mask, visible_mask], dim=0)

        selection_mask = (~mask).T.float().argmax(dim=1)
        selection_mask *= self.num_patches
        selection_mask += torch.arange(self.num_patches, device=x.device)

        embed_token = x[:, selection_mask]

        mask = mask.flatten()
        mask[selection_mask] = True
        mask = mask[: -self.num_patches]

        x = x[:, : -self.num_patches]
        x = x[:, ~mask]

        return x, embed_token

    def forward(
        self, x: Tensor, mask: Tensor | None = None, channels: list[int] = []
    ) -> Tensor:
        """Forward pass of the model."""
        x = self.embed_module(x, channels)

        if mask is not None and not self.embed_token_reduction:
            x = x[:, ~mask]

        if self.embed_token is not None:
            B, *_ = x.shape
            embed_token = self.embed_token.repeat(B, 1, 1)

            # TODO: CURRENT -> Check at inference without mask
            if self.embed_token_reduction and mask is not None:
                x, embed_token = self.reduce_embed_token(x, mask, embed_token)

            x = torch.cat([embed_token, x], dim=1)

        x = self.encoder(x)
        x = self.norm(x)

        if self.embed_token is not None:
            x = x[:, : self.num_patches]

        return x


class DecoderEmbedding(Module):
    """Decoder embedding module."""

    def __init__(
        self, num_patches: int, input_dim: int = 1024, embed_dim: int = 512
    ) -> None:
        """Initialize a new DecoderEmbedding module."""
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.embedder = Linear(input_dim, embed_dim, bias=True)
        self.mask_token = Parameter(
            -torch.ones(1, self.num_patches, embed_dim), requires_grad=False
        )

        self.positional_encodings = get_positional_encodings(
            embed_dim, num_patches, True
        )

        self.apply(_init_weights)

    @lru_cache(128)
    def get_channel_encodings(
        self, channels: tuple[int], embed_size: int, device: str | torch.device
    ) -> Tensor:
        """Get the channel encodings for a given channel."""
        channel_encoding = get_1d_sincos_pos_embed_from_grid(
            embed_size,
            torch.tensor(channels, dtype=torch.float, device=device),
            device=device,
        )
        channel_encoding = channel_encoding.repeat_interleave(
            repeats=self.num_patches, dim=0
        )

        return channel_encoding

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

        # if mask is not None:
        #     x = self.select_from_mask(x, mask, channels)

        x += self.positional_encodings.repeat(len(channels) or 1, 1)

        if len(channels):
            x += self.get_channel_encodings(tuple(channels), x.shape[-1], x.device)

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
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        dropout_attn: float = 0.0,
    ) -> None:
        """Initialize a new VisionTransformer model."""
        super().__init__()

        self.embed_module = DecoderEmbedding(num_patches, in_channels, embed_dim)
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
        output = []
        xx = x.clone()

        for channel in channels:
            x = self.embed_module(xx, mask, [channel])

            x = self.decoder(x)
            x = self.norm(x)

            x = self.predictor(x)

            output.append(x)

        x = torch.stack(output, dim=1).flatten(1, 2)

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
        embed_token: bool = False,
        embed_token_reduction: bool = False,
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

        self.encoder = MaskedEncoderViT(
            image_size=image_size,
            in_channels=IN_CHANNELS[sensor][bands],
            patch_size=patch_size,
            channel_wise=channel_wise,
            embed_token=embed_token,
            embed_token_reduction=embed_token_reduction,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            dropout_attn=dropout_attn,
        )

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.embed_module.num_patches,
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
