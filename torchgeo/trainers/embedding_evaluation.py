"""Embedding classifciation task."""

from os.path import isfile
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import wandb
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, optim
from torch.nn.modules import CrossEntropyLoss, Linear, Module, Sequential
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MetricCollection

from torchgeo.models import ResNet18

from .byol import BYOLTask
from .tile2vec import Tile2VecTask

class EmbeddingEvaluator(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hyperparams["task_name"] == "tile2vec":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = Tile2VecTask.load_from_checkpoint(
                    checkpoint_path=self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = Tile2VecTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder

            self.projector: Optional[Module] = None
            if self.hyperparams.get("projector_embeddings", False):
                self.projector = task.projector
        elif self.hyperparams["task_name"] == "tile2vec-original":
            checkpoint = torch.load(self.hyperparams["checkpoint_path"])
            self.encoder = ResNet18()
            self.encoder.load_state_dict(checkpoint)
            self.encoder.eval()
        elif self.hyperparams["task_name"] == "byol":
            if "checkpoint_path" in self.hyperparams and isfile(
                self.hyperparams["checkpoint_path"]
            ):
                task = BYOLTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
                print(f"Loaded from checkpoint: {self.hyperparams['checkpoint_path']}")
            else:
                task = BYOLTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        else:
            raise ValueError(
                f"Task type '{self.hyperparams['task_name']}' is not valid."
            )

        in_channels = list(self.encoder.children())[0].in_channels
        output = self.encoder(torch.zeros((2, in_channels, 512, 512))).squeeze()
        if self.projector is not None:
            output = self.projector(output)
        out_dim = output.shape[1]

        self.classifier = Linear(out_dim, self.hyperparams["num_classes"])
        self.classifier_loss = CrossEntropyLoss()

        self.metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hyperparams["num_classes"], average="macro"
                ),
                "JaccardIndex": JaccardIndex(
                    num_classes=self.hyperparams["num_classes"]
                ),
                "F1Score": FBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                ),
            }
        )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a LightningModule for pre-training a model with Tile2Vec.

        Keyword Args:
            sensor: type of sensor
            bands: which bands of sensor
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

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_class = getattr(optim, self.hyperparams.get("optimizer", "SGD"))
        lr = self.hyperparams.get("lr", 2e-2)
        weight_decay = self.hyperparams.get("weight_decay", 1e-6)
        momentum = self.hyperparams.get("momentum", 0.9)
        optimizer = optimizer_class(
            self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss"},
        }

    def get_embeddings(self, x: Tensor) -> Tensor:
        """TODO: Docstring."""
        embeddings = self.encoder(x).squeeze()

        if self.projector is not None:
            embeddings = self.projector(embeddings)

        return embeddings.squeeze()

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        with torch.no_grad():
            embeddings = self.get_embeddings(x)
        y_hat = self.classifier(embeddings)

        loss = self.classifier_loss(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, batch_size=x.shape[0])

        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> Any:
        """."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        embeddings = self.get_embeddings(x)

        metrics_classification = self.evaluate_classification(embeddings, y, "val")
        self.log_dict(
            metrics_classification, on_step=True, on_epoch=True, batch_size=x.shape[0]
        )

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        y = batch["label"].squeeze()

        embeddings = self.get_embeddings(x)

        metrics_classification = self.evaluate_classification(embeddings, y, "test")
        metrics_dimensionality = self.evaluate_dimensionality(embeddings)

        self.log_dict(
            metrics_classification, on_step=True, on_epoch=True, batch_size=x.shape[0]
        )

        return metrics_classification | metrics_dimensionality

    def evaluate_classification(
        self, embeddings: Tensor, y: Tensor, stage: Optional[str] = None
    ) -> Dict[str, Tensor]:
        """TODO: Docstring."""
        y_hat = self.classifier(embeddings)

        metrics = self.metrics(y_hat, y)

        if stage:
            metrics = {f"{stage}_{k}": v for k, v in metrics.items()}
            metrics[f"{stage}_loss"] = metrics[f"{stage}_OverallAccuracy"]

        return cast(Dict[str, Tensor], metrics)

    def evaluate_dimensionality(self, embeddings: Tensor) -> Dict[str, Tensor]:
        """TODO: Docstring."""
        embeddings_normalized = torch.nn.functional.normalize(embeddings, dim=1)
        cov_embeddings = torch.cov(embeddings_normalized.T)
        svdvals_embeddings = torch.linalg.svdvals(cov_embeddings.float())
        svdvals_embeddings = svdvals_embeddings.log().sort(descending=True)[0]

        metrics = {"svdvals_embeddings": svdvals_embeddings}

        return metrics

    def test_epoch_end(
        self,
        outputs: Union[
            List[Union[Tensor, Dict[str, Any]]],
            List[List[Union[Tensor, Dict[str, Any]]]],
        ],
    ) -> None:
        """TODO: Docstring."""
        svdvals: List[Tensor] = []

        for output in cast(List[Dict[str, Tensor]], outputs):
            svdvals_embeddings = output["svdvals_embeddings"]
            svdvals.append(svdvals_embeddings)

        svdvals_mean = torch.stack(svdvals).mean(0)

        data = [[x, y] for (x, y) in zip(range(len(svdvals_mean)), svdvals_mean)]
        table: wandb.data_types.Table = wandb.Table(data=data, columns=["Singular Value Rank Index", "Log of singular values"])  # type: ignore
        wandb.log(
            {
                "singular_values_embeddings": wandb.plot.line(
                    table,
                    "Singular Value Rank Index",
                    "Log of singular values",
                    title="Singular Values of Embeddings",
                )
            }
        )
