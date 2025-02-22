# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BYOL tasks."""

import random
from typing import Any, Callable, Dict, Optional, Tuple, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from kornia import augmentation as K
from kornia import filters
from kornia.geometry import transform as KorniaTransform
from packaging.version import parse
from torch import Tensor, optim
from torch.autograd import Variable
from torch.nn.modules import BatchNorm1d, Conv2d, Linear, Module, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet18
from torchvision.models.resnet import resnet50

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    """Computes the normalized mean squared error between x and y.

    Args:
        x: tensor x
        y: tensor y

    Returns:
        the normalized MSE between x and y
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    mse = torch.mean(2 - 2 * (x * y).sum(dim=-1))
    return mse


# TODO: Move this to transforms
class RandomApply(Module):
    """Applies augmentation function (augm) with probability p."""

    def __init__(self, augm: Callable[[Tensor], Tensor], p: float) -> None:
        """Initialize RandomApply.

        Args:
            augm: augmentation function to apply
            p: probability with which the augmentation function is applied
        """
        super().__init__()
        self.augm = augm
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        """Applies an augmentation to the input with some probability.

        Args:
            x: a batch of imagery

        Returns
            augmented version of ``x`` with probability ``self.p`` else an un-augmented
                version
        """
        return x if random.random() > self.p else self.augm(x)


# TODO: This isn't _really_ applying the augmentations from SimCLR as we have
# multispectral imagery and thus can't naively apply color jittering or grayscale
# conversions. We should think more about what makes sense here.
class SimCLRAugmentation(Module):
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
            KorniaTransform.Resize(size=image_size, align_corners=False),
            # Not suitable for multispectral adapt
            # RandomApply(K.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            # K.RandomGrayscale(p=0.2),
            K.RandomHorizontalFlip(),
            K.ImageSequential(
                filters.GaussianBlur2d((3, 3), (1.5, 1.5)), random_apply_weights=[0.1]
            ),
            K.RandomResizedCrop(size=image_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Applys SimCLR augmentations to the input tensor.

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
            BatchNorm1d(hidden_size),
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


class EncoderWrapper(Module):
    """Encoder wrapper for joining a model and a projection head.

    When we call .forward() on this module the following steps happen:

    * The input is passed through the base model
    * When the encoding layer is reached a hook is called
    * The output of the encoding layer is passed through the projection head
    * The forward call returns the output of the projection head
    """

    def __init__(
        self,
        model: Module,
        projector_dim: int,
        projection_size: int = 256,
        hidden_size: int = 4096,
    ) -> None:
        """Initializes EncoderWrapper.

        Args:
            model: model to encode
            projection_size: size of the ouput layer of the projector MLP
            hidden_size: size of hidden layer of the projector MLP
            layer: layer from model to project
        """
        super().__init__()

        self.model = model
        self.projector = MLP(projector_dim, projection_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Pass through the model, and collect the representation from our forward hook.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        output: Tensor = self.projector(self.model(x).squeeze())
        return output


class BYOL(Module):
    """BYOL implementation.

    BYOL contains two identical encoder networks. The first is trained as usual, and its
    weights are updated with each training batch. The second, "target" network, is
    updated using a running average of the first encoder's weights.

    See https://arxiv.org/abs/2006.07733 for more details (and please cite it if you
    use it in your own work).
    """

    def __init__(
        self,
        model: Module,
        projector_dim: int,
        image_size: Tuple[int, int] = (256, 256),
        in_channels: int = 4,
        projection_size: int = 256,
        hidden_size: int = 4096,
        beta: float = 0.99,
        **kwargs: Any,
    ) -> None:
        """Sets up a model for pre-training with BYOL using projection heads.

        Args:
            model: the model to pretrain using BYOL
            image_size: the size of the training images
            hidden_layer: the hidden layer in ``model`` to attach the projection
                head to, can be the name of the layer or index of the layer
            projection_size: size of first layer of the projection MLP
            hidden_size: size of the hidden layer of the projection MLP
            augment_fn: an instance of a module that performs data augmentation
            beta: the speed at which the target encoder is updated using the main
                encoder
        """
        super().__init__()

        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projector_dim, projection_size, hidden_size
        )
        self.predictor = MLP(projection_size, projection_size, hidden_size)
        self.target = EncoderWrapper(model, projector_dim, projection_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the encoder model through the MLP and prediction head.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return cast(Tensor, self.predictor(self.encoder(x)))

    def update_target(self) -> None:
        """Method to update the "target" model weights."""
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data


class BYOLTask(pl.LightningModule):
    """Class for pre-training any PyTorch model using BYOL."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        in_channels = self.hyperparams["in_channels"]
        pretrained = self.hyperparams["imagenet_pretrained"]
        encoder_name = self.hyperparams["encoder_name"]

        if parse(torchvision.__version__) >= parse("0.13"):
            if pretrained:
                kwargs = {
                    "weights": getattr(
                        torchvision.models, f"ResNet{encoder_name[6:]}_Weights"
                    ).DEFAULT
                }
            else:
                kwargs = {"weights": None}
        else:
            kwargs = {"pretrained": pretrained}

        encoder = getattr(torchvision.models, encoder_name)(**kwargs)

        layer = encoder.conv1
        # Creating new Conv2d layer
        new_layer = Conv2d(
            in_channels=in_channels,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        # initialize the weights from new channel with the red channel weights
        copy_weights = 0
        # Copying the weights from the old to the new layer
        new_layer.weight[:, : layer.in_channels, :, :].data[:] = Variable(
            layer.weight.clone(), requires_grad=True
        )
        # Copying the weights of the old layer to the extra channels
        for i in range(in_channels - layer.in_channels):
            channel = layer.in_channels + i
            new_layer.weight[:, channel : channel + 1, :, :].data[:] = Variable(
                layer.weight[:, copy_weights : copy_weights + 1, ::].clone(),
                requires_grad=True,
            )

        encoder.conv1 = new_layer
        projector_dim = encoder.fc.weight.shape[1]
        encoder = Sequential(*(list(encoder.children())[:-1]))

        self.model = BYOL(
            encoder,
            projector_dim=projector_dim,
            in_channels=in_channels,
            image_size=self.hyperparams.get("image_size", (256, 256)),
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
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

    def setup(self, stage: Optional[str] = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        patch_size = self.hyperparams.get("patch_size", (256, 256))
        patch_size = _to_tuple(patch_size)

        self.augment = self.hyperparams.get(
            "augment_fn", SimCLRAugmentation(patch_size)
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
        lr = self.hyperparams.get("lr", 1e-4)
        weight_decay = self.hyperparams.get("weight_decay", 1e-6)
        optimizer = optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
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
            x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.forward(x1), self.forward(x2)
        with torch.no_grad():
            targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.model.update_target()

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.forward(x1), self.forward(x2)
        targ1, targ2 = self.model.target(x1), self.model.target(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """No-op, does nothing."""
