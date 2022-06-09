"""Embedding classifciation task."""

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
from kornia import augmentation as K
from kornia.augmentation.container.image import ImageSequential
from kornia.geometry.transform import Rotate
from pytorch_lightning.core.lightning import LightningModule
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from torch import Tensor, optim
from torch.nn.modules import Module, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, FBetaScore, JaccardIndex, MetricCollection

import wandb
from torchgeo.models import resnet18, resnet50

from ..utils import _to_tuple
from .byol import BYOLTask
from .tile2vec import Tile2VecTask


class EmbeddingEvaluator(LightningModule):
    """Class for pre-training any PyTorch model using Tile2Vec."""

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        # See https://github.com/PyTorchLightning/pytorch-lightning/issues/13108
        # current workaround

        if self.hyperparams["task_name"] == "tile2vec":
            if "checkpoint_path" in self.hyperparams:
                task = Tile2VecTask.load_from_checkpoint(
                    checkpoint_path=self.hyperparams["checkpoint_path"]
                )
            else:
                task = Tile2VecTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        elif self.hyperparams["task_name"] == "byol":
            if "checkpoint_path" in self.hyperparams:
                task = BYOLTask.load_from_checkpoint(
                    self.hyperparams["checkpoint_path"]
                )
            else:
                task = BYOLTask(**self.hyperparams)
            task.freeze()
            self.encoder = task.model.encoder
        else:
            raise ValueError(
                f"Task type '{self.hyperparams['task_name']}' is not valid."
            )

        self.classifiers = {
            "logistic_regression": LogisticRegression(),
            "random_forest": RandomForestClassifier(),
            "mlp": MLPClassifier(),
        }

    def test_step(self, *args: Any, **kwargs: Any) -> Any:
        """TODO: Docstring."""
        batch = args[0]
        x = batch["image"]
        y = batch["mask"].squeeze()

        if self.hyperparams.get("imagenet_pretraining", False):
            x = x[:, :3]

        from torchvision.utils import save_image

        save_image(x[:, :3], "batch.png")

        embeddings = self.encoder(x).squeeze()

        metrics_classification = self.evaluate_classification(embeddings, y)
        metrics_dimensionality = self.evaluate_dimensionality(embeddings)

        self.log_dict(metrics_classification, on_step=True, on_epoch=True)

        return metrics_classification | metrics_dimensionality

    def evaluate_classification(
        self, embeddings: Tensor, y: Tensor
    ) -> Dict[str, Tensor]:
        """."""
        embeddings = embeddings.cpu().numpy()
        y = y.cpu().numpy()
        y = LabelEncoder().fit_transform(y)

        embeddings_train, embeddings_test, y_train, y_test = train_test_split(
            embeddings, y, test_size=0.2
        )

        metrics: Dict[str, Tensor] = {}
        for classifier_name, classifier in self.classifiers.items():
            classifier.fit(embeddings_train, y_train)
            y_hat = classifier.predict(embeddings_test)

            y_test_tensor = torch.from_numpy(y_test).to(self.device)
            y_hat_tensor = torch.from_numpy(y_hat).to(self.device)

            metrics_classifier = self.metrics(y_hat_tensor, y_test_tensor)
            metrics_classifier = {
                f"{classifier_name}_{k}": v for k, v in metrics_classifier.items()
            }
            metrics |= metrics_classifier

        return metrics

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
