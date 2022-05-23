# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tile2Vec tasks."""

from typing import Any, Dict, Optional, Tuple, cast

import torch
from kornia import augmentation as K
from kornia.geometry.transform import Rotate
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn.modules import BatchNorm1d, Conv2d, Linear, Module, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchgeo.models import resnet18, resnet50

from ..utils import _to_tuple

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def triplet_loss(
    anchor: Tensor,
    neighbor: Tensor,
    distant: Tensor,
    margin: float = 0.1,
    regularize: bool = True,
) -> Tensor:
    """Computes the triplet_loss between anchor, neighbor, and distant.

    Args:
        anchor: tensor anchor
        neighbor: tensor neighbor
        disant: tensor distant

    Returns:
        the normalized MSE between x and y
    """
    positive = torch.sqrt(((anchor - neighbor) ** 2).sum(dim=1))
    negative = torch.sqrt(((anchor - distant) ** 2).sum(dim=1))

    distance = positive - negative + margin
    loss = torch.relu(distance).mean()

    if regularize:
        loss += (
            torch.linalg.norm(anchor)
            + torch.linalg.norm(neighbor)
            + torch.linalg.norm(distant)
        )

    return loss


# TODO: This isn't _really_ applying the augmentations from SimCLR as we have
# multispectral imagery and thus can't naively apply color jittering or grayscale
# conversions. We should think more about what makes sense here.
class Augmentations(Module):
    """A module for applying SimCLR augmentations.

    SimCLR was one of the first papers to show the effectiveness of random data
    augmentation in self-supervised-learning setups. See
    https://arxiv.org/pdf/2002.05709.pdf for more details.
    """

    def __init__(self, image_size: Tuple[int, int] = (256, 256)) -> None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
        """
        super().__init__()
        self.size = image_size

        self.augmentation = Sequential(
            K.Resize(size=image_size, align_corners=False),
            K.RandomHorizontalFlip(),
            # ImageSequential(
            #     Rotate(torch.tensor([0])),
            #     Rotate(torch.tensor([90])),
            #     Rotate(torch.tensor([180])),
            #     Rotate(torch.tensor([270])),
            #     random_apply=1
            # )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        return cast(Tensor, self.augmentation(x))


class Tile2Vec(Module):
    """Tile2Vec implementation.

    See https://aaai.org/ojs/index.php/AAAI/article/view/4288 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(
        self,
        model: Module,
        image_size: Tuple[int, int] = (256, 256),
        augment_fn: Optional[Module] = None,
        **kwargs: Any,
    ) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            augment_fn: an instance of a module that performs data augmentation
        """
        super().__init__()

        self.augment: Module
        if augment_fn is None:
            self.augment = Augmentations(image_size)
        else:
            self.augment = augment_fn

        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.model(x).squeeze())


class Tile2VecTask(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        pretrained = self.hyperparams["imagenet_pretraining"]
        encoder = None

        if self.hyperparams["encoder_name"] == "resnet18":
            encoder = resnet18(
                sensor=self.hyperparams["sensor"],
                bands=self.hyperparams["bands"],
                pretrained=pretrained,
            )
        elif self.hyperparams["encoder_name"] == "resnet50":
            encoder = resnet50(
                sensor=self.hyperparams["sensor"],
                bands=self.hyperparams["bands"],
                pretrained=pretrained,
            )
        else:
            raise ValueError(
                f"Encoder type '{self.hyperparams['encoder_name']}' is not valid."
            )

        encoder = Sequential(*(list(encoder.children())[:-1]))

        image_size = self.hyperparams.get("image_size", (256, 256))
        image_size = _to_tuple(image_size)

        self.model = Tile2Vec(encoder, image_size=image_size)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with Tile2Vec.

        Keyword Args:
            sensor: type of sensor
            bands: which bands of sensor
            encoder_name: either "resnet18" or "resnet50"
            imagenet_pretraining: bool indicating whether to use imagenet pretrained
                weights

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "Adam"))
        lr = self.hyperparams.get("lr", 1e-3)
        weight_decay = self.hyperparams.get("weight_decay", 0)
        betas = self.hyperparams.get("betas", (0.5, 0.999))
        optimizer = optimizer_class(
            self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams.get(
                        "learning_rate_schedule_patience", 10
                    ),
                ),
                "monitor": "train_loss",
            },
        }

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]

        with torch.no_grad():
            anchor = self.model.augment(x[:, 0])
            neighbor = self.model.augment(x[:, 1])
            distant = self.model.augment(x[:, 2])

        pred1, pred2, pred3 = (
            self.forward(anchor),
            self.forward(neighbor),
            self.forward(distant),
        )

        loss = triplet_loss(pred1, pred2, pred3)

        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]

        anchor = self.model.augment(x[:, 0])
        neighbor = self.model.augment(x[:, 1])
        distant = self.model.augment(x[:, 2])

        pred1, pred2, pred3 = (
            self.forward(anchor),
            self.forward(neighbor),
            self.forward(distant),
        )

        loss = triplet_loss(pred1, pred2, pred3)

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
