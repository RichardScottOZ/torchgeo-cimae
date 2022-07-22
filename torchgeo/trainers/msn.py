# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""VICReg tasks."""

from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import Linear, Module, Sequential, init
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models import MaskedViT
from ..utils import _to_tuple
from .utils import focal_masking, random_masking

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

MASKING_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "focal_masking": focal_masking,
    "random_masking": random_masking,
}


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        crop_size: tuple[int, int] | None = None,
        scale: float | tuple[float, float] = (0.6, 1.0),
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
            scale: Float or tuple of floats defining the scale range
        """
        super().__init__()

        if crop_size is None:
            crop_size = image_size

        scale = _to_tuple(scale)

        self.augmentation = {
            "train": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size, scale=scale, align_corners=False, resample="BICUBIC"
                ),
                K.RandomHorizontalFlip(),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
            ),
        }

    def forward(self, x: Tensor, stage: str | None = None) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        return cast(Tensor, self.augmentation[stage](x))


def vic_loss(
    x: Tensor,
    y: Tensor,
    invar_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """TODO: Docstring."""
    invar_loss = invar_term(x, y)
    var_loss = var_term(x, y)
    cov_loss = cov_term(x, y)

    loss = invar_loss * invar_coeff + var_loss * var_coeff + cov_loss * cov_coeff

    return loss, invar_loss, var_loss, cov_loss


def invar_term(x: Tensor, y: Tensor) -> Tensor:
    """Invariance term for the VICReg loss."""
    return F.mse_loss(x, y)


def var_term(x: Tensor, y: Tensor) -> Tensor:
    """Variance term for the VICReg loss."""
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    return std_loss


def cov_term(x: Tensor, y: Tensor) -> Tensor:
    """Covariance term for the VICReg loss."""
    batch_size = x.shape[0]
    num_features = x.shape[1]

    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)

    cov_loss_x = off_diagonal(cov_x).pow_(2).sum() / num_features
    cov_loss_y = off_diagonal(cov_y).pow_(2).sum() / num_features

    return cov_loss_x + cov_loss_y


def off_diagonal(x: Tensor) -> Tensor:
    """Indices of off-diagonal elements in a 2D tensor."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VICReg(Module):
    """VICReg implementation."""

    def __init__(self, model: Module, projector: Module) -> None:
        """Setup the VICReg model.

        Args:
            model: The encoder model
            projector: The projector model
        """
        super().__init__()

        self.encoder = model
        self.projector = projector

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """Forward pass of the encoder model through the projector model."""
        embeddings = self.encoder(x, *args, **kwargs)

        projections = self.projector(embeddings)
        projections = projections.mean(1)

        return cast(Tensor, projections)


class MSNTask(LightningModule):
    """Class for pre-training any PyTorch model using Masked Siamese Network."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.image_size = self.hyperparams.get("image_size", 256)
        self.crop_size = self.hyperparams.get("crop_size", 224)
        self.patch_size = self.hyperparams.get("patch_size", 16)

        embed_dim = self.hyperparams.get("embed_dim", 512)
        projection_dim = self.hyperparams.get("projection_dim", 2048)

        encoder = MaskedViT(
            sensor=self.hyperparams["sensor"],
            bands=self.hyperparams.get("bands", "all"),
            image_size=self.crop_size,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            depth=self.hyperparams.get("depth", 24),
            num_heads=self.hyperparams.get("num_heads", 16),
            dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
            dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
        )

        self.mask_fn = self.hyperparams.get(
            "mask_fn", ["focal_masking", "random_masking"]
        )
        self.mask_kwargs = self.hyperparams.get(
            "mask_kwargs",
            {
                "focal_mask_ratio": 0.3,
                "focal_mask_probability": 0.3,
                "random_mask_ratio": 0.7,
                "random_mask_probability": 1.0,
            },
        )

        projector = Linear(embed_dim, projection_dim)
        projector.weight.data.normal_(mean=0.0, std=0.01)
        projector.bias.data.zero_()

        self.model = VICReg(encoder, projector)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with BYOL.

        Keyword Args:
            in_channels: number of channels on the input imagery
            encoder_name: either "resnet18" or "resnet50"
            imagenet_pretrained: bool indicating whether to use imagenet pretrained
                weights

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(dict[str, Any], self.hparams)

        self.config_task()

    def setup(self, stage: str | None = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        image_size = _to_tuple(self.image_size)
        crop_size = _to_tuple(self.crop_size)

        self.augment = Augmentations(image_size, crop_size)

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "AdamW"))
        lr = self.hyperparams.get("lr", 1e-3)
        actual_lr = lr * self.hyperparams.get("batch_size", 64) / 256
        weight_decay = self.hyperparams.get("weight_decay", 0.05)
        betas = self.hyperparams.get("betas", (0.9, 0.95))
        optimizer = optimizer_class(
            self.parameters(), lr=actual_lr, weight_decay=weight_decay, betas=betas
        )

        if self.trainer is None:
            return {}

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(
                    optimizer,
                    self.trainer.max_epochs,
                    self.hyperparams.get("min_lr", 1e-6),
                ),
                "monitor": "val_loss",
            },
        }

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def shared_step(
        self, stage: str | None = None, *args: Any, **kwargs: Any
    ) -> Tensor:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        B, *_ = x.shape

        with torch.no_grad():
            num_patches = (self.crop_size // self.patch_size) ** 2
            mask1 = torch.zeros((B, num_patches), device=x.device, dtype=torch.bool)
            mask2 = mask1.clone()
            for masking_name in self.mask_fn:
                mask1 = MASKING_FUNCTIONS[masking_name](x, mask1, **self.mask_kwargs)
                mask2 = MASKING_FUNCTIONS[masking_name](x, mask2, **self.mask_kwargs)

            aug1, aug2 = self.augment(x, stage), self.augment(x, stage)

        pred1 = self.forward(aug1, mask1)
        pred2 = self.forward(aug2, mask2)

        loss, invar_loss, var_loss, cov_loss = vic_loss(
            pred1,
            pred2,
            invar_coeff=self.hyperparams.get("invar_coeff", 0.5),
            var_coeff=self.hyperparams.get("var_coeff", 0.5),
            cov_coeff=self.hyperparams.get("cov_coeff", 0.5),
        )

        if stage is not None:
            self.log_dict(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_invar_loss": invar_loss,
                    f"{stage}_var_loss": var_loss,
                    f"{stage}_cov_loss": cov_loss,
                },
                on_step=stage != "val",
                on_epoch=True,
            )

        return loss

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        loss = self.shared_step("train", *args, **kwargs)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        self.shared_step("val", *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
