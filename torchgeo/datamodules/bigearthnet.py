# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet datamodule."""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from ffcv.fields.decoders import NDArrayDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import BigEarthNet

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"

from torch.utils.data import Dataset


class CombineDataset(Dataset):
    def __init__(self, ds1: Dataset, ds2: Dataset) -> None:
        self.ds1 = ds1
        self.ds2 = ds2

        self.ds1_len = len(self.ds1)
        self.ds2_len = len(self.ds2)

    def __len__(self) -> int:
        return self.ds1_len + self.ds2_len

    def __getitem__(self, index: int) -> dict:
        if index < self.ds1_len:
            return self.ds1[index]
        else:
            return self.ds2[index - self.ds1_len]


class BigEarthNetDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the BigEarthNet dataset.

    Uses the train/val/test splits from the dataset.
    """

    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 100k random samples
    band_mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    band_maxs_raw = torch.tensor(
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    band_mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )

    def __init__(
        self,
        root_dir: str,
        bands: str = "all",
        num_classes: int = 19,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 10,
        persistent_workers: bool = False,
        load_target: bool = True,
        use_ffcv: bool = False,
        distributed: bool = False,
        batches_ahead: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for BigEarthNet based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the BigEarthNet Dataset classes
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()

        self.root_dir = root_dir
        self.bands = bands
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.load_target = load_target
        self.use_ffcv = use_ffcv
        self.distributed = distributed
        self.batches_ahead = batches_ahead

        if bands == "all":
            self.mins = self.band_mins[:, None, None]
            self.maxs = self.band_maxs[:, None, None]
        elif bands == "s1":
            self.mins = self.band_mins[:2, None, None]
            self.maxs = self.band_maxs[:2, None, None]
        else:
            self.mins = self.band_mins[2:, None, None]
            self.maxs = self.band_maxs[2:, None, None]

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset."""
        sample["image"] = sample["image"].float()
        sample["image"] = (sample["image"] - self.mins) / (self.maxs - self.mins)
        sample["image"] = torch.clip(sample["image"], min=0.0, max=1.0)
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.use_ffcv:
            self.train_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_trainval.beton"
            )
            self.val_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_test.beton"
            )
            self.test_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_test.beton"
            )

            if (
                not os.path.isfile(self.train_dataset_path)
                or not os.path.isfile(self.val_dataset_path)
                or not os.path.isfile(self.test_dataset_path)
            ):
                raise ValueError("FFCV files not found in root_dir.")
            return

        BigEarthNet(self.root_dir, split="train", bands=self.bands, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.
        """
        if self.use_ffcv:
            self.train_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_trainval.beton"
            )
            self.val_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_test.beton"
            )
            self.test_dataset_path = os.path.join(
                self.root_dir, "BigEarthNet_test.beton"
            )

            self.train_pipeline = {
                "image": [NDArrayDecoder(), ToTensor()],
                "label": [NDArrayDecoder(), ToTensor()],
            }
            self.val_pipeline = {
                "image": [NDArrayDecoder(), ToTensor()],
                "label": [NDArrayDecoder(), ToTensor()],
            }
            self.test_pipeline = {
                "image": [NDArrayDecoder(), ToTensor()],
                "label": [NDArrayDecoder(), ToTensor()],
            }
            return

        transforms = Compose([self.preprocess])
        # self.train_dataset = BigEarthNet(
        #     self.root_dir,
        #     split="train",
        #     bands=self.bands,
        #     num_classes=self.num_classes,
        #     transforms=transforms,
        #     load_target=self.load_target,
        # )
        # self.val_dataset = BigEarthNet(
        #     self.root_dir,
        #     split="val",
        #     bands=self.bands,
        #     num_classes=self.num_classes,
        #     transforms=transforms,
        #     load_target=self.load_target,
        # )
        train_dataset = BigEarthNet(
            self.root_dir,
            split="train",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )
        val_dataset = BigEarthNet(
            self.root_dir,
            split="val",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )

        self.train_dataset = CombineDataset(train_dataset, val_dataset)

        self.test_dataset = BigEarthNet(
            "/scratch/users/mike/data/BigEarthNet",
            # self.root_dir,
            split="test",
            bands=self.bands,
            num_classes=self.num_classes,
            transforms=transforms,
            load_target=self.load_target,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        if self.use_ffcv:
            return Loader(
                fname=self.train_dataset_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.QUASI_RANDOM
                if not self.distributed
                else OrderOption.RANDOM,
                distributed=self.distributed,
                batches_ahead=self.batches_ahead,
                pipelines=self.train_pipeline,
            )

        if self.num_workers > 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                drop_last=True,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        if self.use_ffcv:
            return Loader(
                fname=self.val_dataset_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.SEQUENTIAL,
                distributed=self.distributed,
                batches_ahead=self.batches_ahead,
                pipelines=self.val_pipeline,
            )
        # return DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     shuffle=False,
        #     pin_memory=self.pin_memory,
        #     prefetch_factor=self.prefetch_factor,
        #     drop_last=False,
        # )

        if self.num_workers > 0:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
                drop_last=True,
            )

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        if self.use_ffcv:
            return Loader(
                fname=self.test_dataset_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                order=OrderOption.SEQUENTIAL,
                distributed=self.distributed,
                pipelines=self.test_pipeline,
                batches_ahead=self.batches_ahead,
                drop_last=False,
            )

        if self.num_workers > 0:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=self.pin_memory,
                prefetch_factor=self.prefetch_factor,
            )

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def plot(self, *args: Any, **kwargs: Any) -> plt.Figure:
        """Run :meth:`torchgeo.datasets.BigEarthNet.plot`.

        .. versionadded:: 0.2
        """
        return self.val_dataset.plot(*args, **kwargs)
