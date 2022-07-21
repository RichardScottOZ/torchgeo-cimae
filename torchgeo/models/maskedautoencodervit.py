"""MaskedVit."""

from typing import cast

import torch
from kornia.contrib.vit import TransformerEncoderBlock
from torch import Tensor
from torch.nn import Conv2d, LayerNorm, Linear, Module, Sequential, init
from torch.nn.parameter import Parameter

from .utils import get_2d_sincos_pos_embed

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
        self, input_dim: int, embed_dim: int, patch_size: int, image_size: int
    ) -> None:
        """Initialize the encoder embedding module."""
        super().__init__()

        # logic needed in case a backbone is passed
        self.embedder = Conv2d(
            input_dim, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.initialize_positional_embeddings(image_size, patch_size, embed_dim)
        self.initialize_weights()

    def initialize_positional_embeddings(
        self, image_size: int, patch_size: int, embed_dim: int
    ) -> None:
        """Initialize the positional embeddings."""
        self.num_patches = (image_size // patch_size) ** 2
        self.positional_embeddings = Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )
        positional_embeddings = get_2d_sincos_pos_embed(
            self.positional_embeddings.shape[-1],
            int(self.num_patches**0.5),
            cls_token=False,
        )
        self.positional_embeddings.data.copy_(
            torch.from_numpy(positional_embeddings).float().unsqueeze(0)
        )

    def initialize_weights(self) -> None:
        """Initialize the weights."""
        w = self.embedder.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x: Tensor) -> Tensor:
        """TODO: Docstring."""
        x = self.embedder(x)
        B, C, _, _ = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # BxPxH

        x += self.positional_embeddings

        return x


class MaskedEncoderViT(Module):
    """Vision transformer (ViT) module."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int = 16,
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

        self.embed_module = EncoderEmbedding(
            in_channels, embed_dim, patch_size, image_size
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass of the model."""
        out = self.embed_module(x)

        if mask is not None:
            B, _, H = out.shape
            out = out[~mask].view(B, -1, H)

        out = self.encoder(out)
        out = self.norm(out)

        return cast(Tensor, out)


class DecoderEmbedding(Module):
    """TODO: Docstring."""

    def __init__(
        self, num_patches: int, input_dim: int = 3, embed_dim: int = 768
    ) -> None:
        """TODO: Docstring."""
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.embedder = Linear(input_dim, embed_dim, bias=True)

        self.mask_token = Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.initialize_positional_embeddings()
        self.apply(self._init_weights)

    def initialize_positional_embeddings(self) -> None:
        """Initialize the positional embeddings."""
        self.positional_embeddings = Parameter(
            torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        positional_embeddings = get_2d_sincos_pos_embed(
            self.positional_embeddings.shape[-1],
            int(self.num_patches**0.5),
            cls_token=False,
        )
        self.positional_embeddings.data.copy_(
            torch.from_numpy(positional_embeddings).float().unsqueeze(0)
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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """TODO: Docstring."""
        x = self.embedder(x)
        B, _, _ = x.shape

        if mask is not None:
            x_flat = x.flatten(0, 1)
            x = self.mask_token.repeat(B, self.num_patches, 1)
            x[~mask] = x_flat

        x += self.positional_embeddings

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

        self.embed_module = DecoderEmbedding(num_patches, in_channels, embed_dim)

        self.decoder = TransformerEncoder(
            embed_dim, depth, num_heads, dropout_rate, dropout_attn
        )

        self.predictor = Linear(
            embed_dim, patch_size**2 * out_channels, bias=True
        )  # decoder to patch

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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Forward pass of the model."""
        out = self.embed_module(x, mask)
        out = self.decoder(out)
        out = self.norm(out)

        out = self.predictor(out)

        return cast(Tensor, out)


class MaskedAutoencoderViT(Module):
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

        self.decoder = MaskedDecoderViT(
            num_patches=self.encoder.embed_module.num_patches,
            image_size=image_size,
            patch_size=patch_size,
            in_channels=embed_dim,
            out_channels=3,  # IN_CHANNELS[sensor][bands],
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            dropout_rate=decoder_dropout_rate,
            dropout_attn=decoder_dropout_attn,
        )

    def forward(
        self, x: Tensor, mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass of the model."""
        latent = self.encoder(x, mask)
        pred: Tensor = self.decoder(latent, mask)

        if self.return_latent:
            return pred, latent

        return pred


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
