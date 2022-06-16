# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tile2Vec tasks."""

from typing import Any, Dict, Optional, Tuple, cast

import torch
from kornia import augmentation as K
from kornia.augmentation.container.image import ImageSequential
from kornia.geometry.transform import Rotate
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn.functional import relu
from torch.nn.modules import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.resnet import BasicBlock, Bottleneck, _resnet

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
    l2: float = 0,
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
    loss = relu(distance, inplace=True).mean()

    if l2 > 0:
        anchor_norm = torch.linalg.norm(anchor, dim=1)
        neighbor_norm = torch.linalg.norm(neighbor, dim=1)
        distant_norm = torch.linalg.norm(distant, dim=1)

        norm = l2 * (anchor_norm + neighbor_norm + distant_norm).mean()

        loss += norm

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

    def __init__(
        self,
        image_size: Tuple[int, int] = (256, 256),
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize a module for applying SimCLR augmentations.

        Args:
            image_size: Tuple of integers defining the image size
            device: Device used
        """
        super().__init__()
        self.size = image_size
        self.rotations = (torch.tensor([90.0]) * torch.tensor([0, 1, 2, 3])).to(device)

        self.augmentation = Sequential(
            K.Resize(size=image_size, align_corners=False),
            K.RandomHorizontalFlip(),
            ImageSequential(
                *[Rotate(rotation) for rotation in self.rotations],
                random_apply=1,
                same_on_batch=True,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys augmentations to the input tensor.

        Args:
            x: a batch of imagery

        Returns:
            an augmented batch of imagery
        """
        return cast(Tensor, self.augmentation(x))


class MLP(Module):
    """MLP used in the BYOL projection head."""

    def __init__(
        self, dim: int, projection_size: int = 256, hidden_size: int = 4096
    ) -> None:
        """Initializes the MLP projection head.

        Args:
            dim: size of layer to project
            projection_size: size of the output layer
            hidden_size: size of the hidden layer
        """
        super().__init__()
        self.mlp = Sequential(
            Linear(dim, hidden_size),
            BatchNorm1d(hidden_size),  # type: ignore[no-untyped-call]
            ReLU(inplace=True),
            Linear(hidden_size, projection_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MLP model.

        Args:
            x: batch of imagery

        Returns:
            embedded version of the input
        """
        return cast(Tensor, self.mlp(x))


class Tile2Vec(Module):
    """Tile2Vec implementation.

    See https://aaai.org/ojs/index.php/AAAI/article/view/4288 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(self, model: Module) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            augment_fn: an instance of a module that performs data augmentation
        """
        super().__init__()

        self.encoder = model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.encoder(x).squeeze())


class Tile2VecTask(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self, **kwargs: Any) -> None:
        """TODO: Docstring."""
        pretrained = self.hyperparams.get("pretrained", False)
        imagenet_pretrained = self.hyperparams.get("imagenet_pretrained", False)
        sensor = self.hyperparams["sensor"]
        bands = self.hyperparams.get("bands", "all")
        encoder = None

        if self.hyperparams["encoder_name"] == "resnet18":
            encoder = resnet18(
                sensor=sensor,
                bands=bands,
                block=BasicBlock,
                pretrained=pretrained,
                imagenet_pretrained=imagenet_pretrained,
            )
        elif self.hyperparams["encoder_name"] == "resnet50":
            encoder = resnet50(
                sensor=sensor,
                bands=bands,
                pretrained=pretrained,
                imagenet_pretrained=imagenet_pretrained,
            )
        else:
            raise ValueError(
                f"Encoder type '{self.hyperparams['encoder_name']}' is not valid."
            )

        self.projector: Optional[Module] = None
        if self.hyperparams.get("project", False):
            prev_dim = encoder.fc.weight.shape[1]

            self.projector = Sequential(
                Linear(prev_dim, prev_dim),
                ReLU(inplace=True),
                Linear(prev_dim, self.hyperparams.get("projection_dim", 512)),
            )

        encoder = Sequential(*(list(encoder.children())[:-1]))

        self.model = Tile2Vec(encoder)

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/13108
        # current workaround
        if self.trainer is not None:
            device = self.trainer.strategy.root_device

        patch_size = self.hyperparams.get("patch_size", (256, 256))
        patch_size = _to_tuple(patch_size)

        self.augment = self.hyperparams.get(
            "augment_fn", Augmentations(patch_size, device)
        )

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
                "monitor": f"train_{'embedding' if self.projector is None else 'projector'}_loss",
            },
        }

    def shared_step(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, Optional[Tensor]]:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]

        with torch.no_grad():
            anchor = self.augment(x[:, 0])
            neighbor = self.augment(x[:, 1])
            distant = self.augment(x[:, 2])

        anchor_embeddings = self.forward(anchor)
        neighbor_embeddings = self.forward(neighbor)
        distant_embeddings = self.forward(distant)

        embedding_loss = triplet_loss(
            anchor_embeddings,
            neighbor_embeddings,
            distant_embeddings,
            self.hyperparams.get("margin", 0.1),
            self.hyperparams.get("l2", 0),
        )

        projector_loss: Optional[Tensor] = None
        if self.projector is not None:
            projector_loss = triplet_loss(
                self.projector(anchor_embeddings),
                self.projector(neighbor_embeddings),
                self.projector(distant_embeddings),
                self.hyperparams.get("margin", 0.1),
                self.hyperparams.get("l2", 0),
            )

        return (embedding_loss, projector_loss)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        embedding_loss, projector_loss = self.shared_step(*args, **kwargs)

        self.log("train_embedding_loss", embedding_loss, on_step=True, on_epoch=True)

        if projector_loss is None:
            return embedding_loss

        self.log("train_projector_loss", projector_loss, on_step=True, on_epoch=True)

        return projector_loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        embedding_loss, projector_loss = self.shared_step(*args, **kwargs)

        self.log("val_loss", embedding_loss, on_step=True, on_epoch=True)
        if projector_loss is not None:
            self.log("val_projector_loss", projector_loss, on_step=True, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
