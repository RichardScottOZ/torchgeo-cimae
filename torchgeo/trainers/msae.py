# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""VICReg tasks."""

from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
import wandb
from kornia import augmentation as K
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn import Module, Sequential
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


def embedding_loss(
    embedding1: Tensor, embedding2: Tensor, mean_patches: bool = True
) -> Tensor:
    """Compute embedding loss."""
    if not mean_patches:
        return F.mse_loss(embedding1.mean(dim=1), embedding2.mean(dim=1))

    return F.mse_loss(embedding1, embedding2)


def masked_reconstruction_loss(item: dict[str, Tensor], patch_size: int) -> Tensor:
    """Compute masked reconstruction loss."""
    target, pred, mask = item["target"], item["pred"], item["mask_target"]
    B, *_ = pred.shape

    target = patchify(target, patch_size)
    target = target.view(pred.shape)

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    num_hidden = mask.sum()
    if num_hidden == 0:
        return cast(Tensor, loss.mean())

    loss = (loss * mask).sum() / (B * num_hidden)  # mean loss on removed patches

    return cast(Tensor, loss)


class Augmentations(Module):
    """A module for applying augmentations."""

    def __init__(
        self,
        image_size: int | tuple[int, int] = (256, 256),
        crop_size: int | tuple[int, int] | None = None,
        patch_size: int = 16,
        scale: float | tuple[float, float] = (0.6, 1.0),
    ) -> None:
        """Initialize augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            crop_size: Tuple of integers defining the crop size
            patch_size: Integer defining the patch size
            scale: Float or tuple of floats defining the scale range
        """
        super().__init__()

        crop_size = crop_size or image_size
        self.patch_size = patch_size
        scale = _to_tuple(scale)

        self.rotation: dict[str, int] = {"transform": -1, "transform_prime": -1}
        self.flip: dict[str, bool] = {"transform": False, "transform_prime": False}

        self.augmentation = {
            "train": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.RandomResizedCrop(
                    size=crop_size, scale=scale, align_corners=False, resample="BICUBIC"
                ),
            ),
            "val": Sequential(
                K.Resize(size=image_size, align_corners=False),
                K.CenterCrop(size=crop_size, align_corners=False, resample="BICUBIC"),
            ),
        }

    def forward(
        self, x: Tensor, stage: str | None = None, mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Applys SimCLR augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        if stage is None:
            return cast(Tensor, self.augmentation["train"](x))

        if "transform" in stage and mask is not None:
            _, C, *_ = x.shape
            P = len(mask)
            side = int((P // C) ** 0.5)
            mask = mask.view(1, C, side, side)

            self.rotation[stage] = int(
                torch.randint(0, 4, (1,), device=x.device).item()
            )
            x = torch.rot90(x, self.rotation[stage], [-2, -1])
            mask = torch.rot90(mask, self.rotation[stage], [-2, -1])

            self.flip[stage] = bool((torch.rand(1) > 0.5).item())
            if self.flip[stage]:
                x = x.flip(-1)
                mask = mask.flip(-1)

            mask = mask[0].flatten()

            return x, mask

        return cast(Tensor, self.augmentation[stage](x))

    def inverse(self, item: dict[str, Tensor], stage: str) -> dict[str, Tensor]:
        """Applys augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        pred, latent = item["pred"], item["latent"]

        B, *_ = latent.shape
        *_, PS = pred.shape

        pred = unpatchify(pred, self.patch_size)
        latent = unpatchify(latent, self.patch_size)

        _, _, H, W = pred.shape
        pred = pred.view(B, -1, H, W)

        pred = torch.rot90(pred, -self.rotation[stage], [-2, -1])
        latent = torch.rot90(latent, -self.rotation[stage], [-2, -1])

        if self.flip[stage]:
            pred = torch.flip(pred, [-1])
            latent = torch.flip(latent, [-1])

        pred = pred.flatten(0, 1).unsqueeze(1)
        item["pred"] = patchify(pred, self.patch_size).view(B, -1, PS)
        item["latent"] = patchify(latent, self.patch_size)

        return item


class MSAETask(LightningModule):
    """Class for pre-training any PyTorch model using Masked Siamese Network."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.image_size = self.hyperparams.get("image_size", 256)
        self.crop_size = self.hyperparams.get("crop_size", 224)
        self.patch_size = self.hyperparams.get("patch_size", 16)
        self.channel_wise = self.hyperparams.get("channel_wise", False)
        self.channel_shuffle = self.hyperparams.get("channel_shuffle", False)
        self.embed_token = self.hyperparams.get("embed_token", False)
        self.embed_token_reduction = self.hyperparams.get(
            "embed_token_reduction", False
        )
        self.B = self.hyperparams.get("batch_size", 64)
        self.num_patches = (self.crop_size // self.patch_size) ** 2

        self.model = MaskedAutoencoderViT(
            sensor=self.hyperparams["sensor"],
            bands=self.hyperparams.get("bands", "all"),
            image_size=self.crop_size,
            patch_size=self.patch_size,
            channel_wise=self.channel_wise,
            embed_token=self.embed_token,
            embed_token_reduction=self.embed_token_reduction,
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

        self.mask_fn = self.hyperparams.get(
            "mask_fn", ["random_masking"]  # ["focal_masking", "random_masking"]
        )
        self.mask_kwargs = self.hyperparams.get(
            "mask_kwargs",
            {
                "focal_mask_ratio": 0.3,
                "focal_mask_probability": 0.0,
                "num_patches": (self.crop_size // self.patch_size) ** 2,
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
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """TODO: Docstring."""
        batch = args[0]
        x: Tensor = batch["image"]
        _, self.C, *_ = x.shape

        with torch.no_grad():
            target = self.augment(x, stage)

            item1 = self.get_item(target, "transform", self.C)
            item2 = self.get_item(target, "transform_prime", self.C)

        item1 = self.forward(item1)
        item2 = self.forward(item2)

        item1["pred"] = item1["pred"].view(self.B * self.C, self.num_patches, -1)
        item2["pred"] = item2["pred"].view(self.B * self.C, self.num_patches, -1)

        # Restore transform augmentation for pred & latent
        item1 = self.augment.inverse(item1, "transform")
        item2 = self.augment.inverse(item2, "transform_prime")

        # Calculate losses
        loss_rec1 = masked_reconstruction_loss(item1, self.patch_size)
        loss_rec2 = masked_reconstruction_loss(item2, self.patch_size)
        loss_embedding = embedding_loss(item1["latent"], item2["latent"])
        loss = loss_embedding + (loss_rec1 + loss_rec2) / 2

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_loss_rec1": loss_rec1,
                f"{stage}_loss_rec2": loss_rec2,
                f"{stage}_loss_embedding": loss_embedding,
            },
            on_step=stage != "val",
            on_epoch=True,
        )

        return {"loss": loss, "item1": item1, "item2": item2}

    def get_item(
        self, target: Tensor, transform_stage: str, C: int
    ) -> dict[str, Tensor]:
        """Get the item for the given transform stage."""
        input = target.clone()
        mask = self.generate_mask(self.C, self.num_patches, input.device)
        mask_target = mask

        encoder_channels = torch.randperm(C, device=target.device).tolist()
        decoder_channels = torch.randperm(C, device=target.device).tolist()

        input = input[:, encoder_channels]
        target = target[:, decoder_channels]

        mask_target = mask.view(C, -1)[decoder_channels].flatten()
        mask = mask.view(C, -1)[encoder_channels].flatten()

        input, mask = self.augment(input, transform_stage, mask)

        input = input.flatten(0, 1).unsqueeze(1)
        target = target.flatten(0, 1).unsqueeze(1)

        return {
            "input": input,
            "target": target,
            "mask": mask,
            "mask_target": mask_target,
            "encoder_channels": encoder_channels,  # type: ignore
            "decoder_channels": decoder_channels,  # type: ignore
        }

    def generate_mask(
        self, C: int, num_patches: int, device: torch.device | str
    ) -> Tensor:
        """Generate a mask of the image."""
        mask = torch.zeros(
            num_patches if not self.channel_wise else num_patches * C,
            device=device,
            dtype=torch.bool,
        )
        for masking_name in self.mask_fn:
            mask = MASKING_FUNCTIONS[masking_name](mask, **self.mask_kwargs)

        return mask

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        output = self.shared_step("train", *args, **kwargs)

        return cast(Tensor, output["loss"])

    def validation_step(self, *args: Any, **kwargs: Any) -> dict[str, Tensor]:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        output = self.shared_step("val", *args, **kwargs)

        if args[1] > 0:
            return {}

        images1 = self.prepare_output(output["item1"])  # type: ignore
        images2 = self.prepare_output(output["item2"])  # type: ignore

        return {"images1": images1, "images2": images2}

    def prepare_output(self, item: dict[str, Tensor]) -> Tensor:
        """Prepare the output for logging.

        Args:
            aug: tensor of data to run through the model
            aug_shuffled: tensor of data to run through the model
            pred: tensor of data to run through the model

        Returns:
            output from the model
        """
        input, target, pred, mask = (
            item["input"],
            item["target"],
            item["pred"],
            item["mask"],
        )

        pred = pred.view(self.B * self.C, self.num_patches, -1)
        mask = mask.view(self.C, -1).repeat(self.B, 1)

        target_patch = patchify(input, self.patch_size)
        mask_expanded = (~mask.bool()).unsqueeze(-1).expand(target_patch.shape)
        masked_target = torch.zeros_like(target_patch, device=input.device)
        masked_target[mask_expanded] = target_patch[mask_expanded]

        pred = unpatchify(pred, self.patch_size)
        masked_target = unpatchify(masked_target, self.patch_size)

        target = target.view(self.B, -1, self.crop_size, self.crop_size)
        input = input.view(self.B, -1, self.crop_size, self.crop_size)
        masked_target = masked_target.view(self.B, -1, self.crop_size, self.crop_size)
        pred = pred.view(self.B, -1, self.crop_size, self.crop_size)

        return torch.stack([input, masked_target, target, pred])

    def validation_epoch_end(
        self,
        validation_step_outputs: list[Tensor | dict[str, Any]]
        | list[list[Tensor | dict[str, Any]]],
    ) -> None:
        """Log images."""
        images1 = validation_step_outputs[0]["images1"]  # type: ignore
        images2 = validation_step_outputs[0]["images2"]  # type: ignore

        grid1 = make_grid(images1[:, :8].flatten(0, 1)[:, :3], nrow=8)
        images1 = wandb.Image(grid1, caption="Images")

        grid2 = make_grid(images2[:, :8].flatten(0, 1)[:, :3], nrow=8)
        images2 = wandb.Image(grid2, caption="Images")

        wandb.log({"Images1": images1})
        wandb.log({"Images2": images2})

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
