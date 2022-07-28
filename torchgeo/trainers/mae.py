# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# TODO: Attribution (Facebook)
"""MAE tasks."""

from typing import Any, Callable, cast

import torch
import wandb
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn.modules import Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid

from ..models import MaskedAutoencoderViT
from ..utils import _to_tuple
from .utils import focal_masking, patchify, random_masking, unpatchify

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"

MASKING_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "focal_masking": focal_masking,
    "random_masking": random_masking,
}


def masked_reconstruction_loss(
    target: Tensor, pred: Tensor, mask: Tensor, patch_size: int
) -> Tensor:
    """Compute masked reconstruction loss."""
    target = patchify(target, patch_size)
    target = target.view(pred.shape)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    num_hidden = mask.sum()
    if num_hidden == 0:
        return cast(Tensor, loss.mean())

    loss = (loss * mask).sum() / num_hidden  # mean loss on removed patches

    return cast(Tensor, loss)


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        crop_size: tuple[int, int] | None = None,
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
        """
        super().__init__()

        if crop_size is None:
            crop_size = image_size

        self.augmentation = {
            "train": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size,
                    scale=(0.6, 1.0),
                    align_corners=False,
                    resample="BICUBIC",
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


class MAETask(LightningModule):
    """Class for pre-training any PyTorch model using VICReg."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.image_size = self.hyperparams.get("image_size", 256)
        self.crop_size = self.hyperparams.get("crop_size", 224)
        self.patch_size = self.hyperparams.get("patch_size", 16)
        self.channel_wise = self.hyperparams.get("channel_wise", False)
        self.batch_size = self.hyperparams.get("batch_size", 64)

        self.model = MaskedAutoencoderViT(
            sensor=self.hyperparams["sensor"],
            bands=self.hyperparams.get("bands", "all"),
            image_size=self.crop_size,
            patch_size=self.patch_size,
            channel_wise=self.channel_wise,
            embed_dim=self.hyperparams.get("embed_dim", 1024),
            depth=self.hyperparams.get("depth", 24),
            num_heads=self.hyperparams.get("num_heads", 16),
            dropout_rate=self.hyperparams.get("dropout_rate", 0.0),
            dropout_attn=self.hyperparams.get("dropout_attn", 0.0),
            decoder_embed_dim=self.hyperparams.get("decoder_embed_dim", 512),
            decoder_depth=self.hyperparams.get("decoder_depth", 8),
            decoder_num_heads=self.hyperparams.get("decoder_num_heads", 16),
            decoder_dropout_rate=self.hyperparams.get("decoder_dropout_rate", 0.0),
            decoder_dropout_attn=self.hyperparams.get("decoder_dropout_attn", 0.0),
        )

        self.mask_fn = self.hyperparams.get("mask_fn", ["random_masking"])
        self.mask_kwargs = self.hyperparams.get(
            "mask_kwargs",
            {
                # "focal_mask_ratio": 0.3,
                # "focal_mask_probability": 0.3,
                "random_mask_ratio": 0.7,
                "random_mask_probability": 1.0,
            },
        )

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
        actual_lr = lr * self.batch_size / 256
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
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        B, C, *_ = x.shape

        with torch.no_grad():
            aug = self.augment(x, stage)

            num_patches = (self.crop_size // self.patch_size) ** 2
            if self.channel_wise:
                num_patches *= C
                aug = aug.flatten(0, 1).unsqueeze(1)

            mask = torch.zeros((B, num_patches), device=aug.device, dtype=torch.bool)
            for masking_name in self.mask_fn:
                mask = MASKING_FUNCTIONS[masking_name](aug, mask, **self.mask_kwargs)

            aug_shuffled = aug  # aug[:, torch.randperm(C)]

        pred = self.forward(aug_shuffled, mask)
        loss = masked_reconstruction_loss(aug, pred, mask, self.patch_size)

        self.log(f"{stage}_loss", loss, on_step=stage != "val", on_epoch=True)

        if self.channel_wise:
            pred = pred.view(B * C, num_patches // C, -1)
            mask = mask.view(B * C, -1)

        return loss, aug, aug_shuffled, pred, mask

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        loss, *_ = self.shared_step("train", *args, **kwargs)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        _, aug, aug_shuffled, pred, mask = self.shared_step("val", *args, **kwargs)

        aug_shuffled = aug_shuffled[:, :3]
        aug_patch = patchify(aug_shuffled, self.patch_size)

        mask_expanded = (~mask.bool()).unsqueeze(-1).expand(aug_patch.shape)
        masked_aug = torch.zeros_like(aug_patch, device=self.device)  # type: ignore
        masked_aug[mask_expanded] = aug_patch[mask_expanded]

        pred = unpatchify(pred, self.patch_size)
        masked_aug = unpatchify(masked_aug, self.patch_size)

        if not self.channel_wise:
            return {"images": torch.stack([aug[:, :3], masked_aug, pred[:, :3]])}

        aug = aug.view(self.batch_size, -1, self.crop_size, self.crop_size)
        pred = pred.view(self.batch_size, -1, self.crop_size, self.crop_size)
        return {"images": torch.stack([aug[:, :3], pred[:, :3]])}

    def validation_epoch_end(
        self,
        validation_step_outputs: list[Tensor | dict[str, Any]]
        | list[list[Tensor | dict[str, Any]]],
    ) -> None:
        """Log images."""
        images = validation_step_outputs[0]["images"]  # type: ignore

        grid = make_grid(images[:, :8].flatten(0, 1)[:, :3])
        images = wandb.Image(grid, caption="Images")

        wandb.log({"Images": images})

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
