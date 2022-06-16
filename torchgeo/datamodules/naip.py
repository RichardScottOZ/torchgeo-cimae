# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Type, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torchgeo.datamodules as datamodules
import torchgeo.datasets as datasets
import torchgeo.samplers as samplers

from ..datasets import (
    CDL,
    NAIP,
    BoundingBox,
    Chesapeake13,
    GeoDataset,
    create_bounding_box,
    stack_samples,
    stack_triplet_samples,
)
from ..samplers.batch import (
    BatchGeoSampler,
    RandomBatchGeoSampler,
    TripletBatchGeoSampler,
    TripletTileBatchGeoSampler,
)
from ..samplers.single import GeoSampler, GridGeoSampler
from .utils import roi_split_half

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class NAIPChesapeakeDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    # TODO: tune these hyperparams
    length = 1000
    stride = 128

    def __init__(
        self,
        naip_root_dir: str,
        chesapeake_root_dir: str,
        batch_size: int = 64,
        num_workers: int = 0,
        patch_size: int = 256,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIP and Chesapeake based DataLoaders.

        Args:
            naip_root_dir: directory containing NAIP data
            chesapeake_root_dir: directory containing Chesapeake data
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of patches to sample
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.naip_root_dir = naip_root_dir
        self.chesapeake_root_dir = chesapeake_root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_size = patch_size

    def naip_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the NAIP Dataset.

        Args:
            sample: NAIP image dictionary

        Returns:
            preprocessed NAIP data
        """
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()

        del sample["bbox"]

        return sample

    def chesapeake_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Chesapeake Dataset.

        Args:
            sample: Chesapeake mask dictionary

        Returns:
            preprocessed Chesapeake data
        """
        sample["mask"] = sample["mask"].long()[0]

        del sample["bbox"]

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        Chesapeake13(self.chesapeake_root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: state to set up
        """
        # TODO: these transforms will be applied independently, this won't work if we
        # add things like random horizontal flip
        chesapeake = Chesapeake13(
            self.chesapeake_root_dir, transforms=self.chesapeake_transform
        )
        naip = NAIP(
            self.naip_root_dir,
            chesapeake.crs,
            chesapeake.res,
            transforms=self.naip_transform,
        )
        self.dataset = chesapeake & naip

        # TODO: figure out better train/val/test split
        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2
        train_roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
        val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
        test_roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)

        self.train_sampler = RandomBatchGeoSampler(
            naip, self.patch_size, self.batch_size, self.length, train_roi
        )
        self.val_sampler = GridGeoSampler(naip, self.patch_size, self.stride, val_roi)
        self.test_sampler = GridGeoSampler(naip, self.patch_size, self.stride, test_roi)

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.dataset,
            batch_sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=stack_samples,
        )


class NAIPCDLDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NAIP and CDL datasets.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        naip_root_dir: str,
        cdl_root_dir: Optional[str] = None,
        batch_size: int = 64,
        length: int = 1000,
        train_length: Optional[int] = None,
        val_length: Optional[int] = None,
        test_length: Optional[int] = None,
        num_workers: int = 0,
        patch_size: int = 256,
        neighborhood: int = 100,
        dataset_split: str = "roi_split_half",
        area_of_interest: Optional[List[Any]] = None,
        train_sampler_class: str = "RandomBatchGeoSampler",
        val_sampler_class: str = "RandomBatchGeoSampler",
        test_sampler_class: str = "RandomBatchGeoSampler",
        train_collate_fn: str = "stack_samples",
        val_collate_fn: str = "stack_samples",
        test_collate_fn: str = "stack_samples",
        cache: bool = True,
        cache_size: int = 128,
        pin_memory: bool = False,
        block_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NAIP and Chesapeake based DataLoaders.

        Args:
            naip_root_dir: directory containing NAIP data
            cdl_root_dir: directory containing Chesapeake data
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            patch_size: size of patches to sample
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.naip_root_dir = naip_root_dir
        self.cdl_root_dir = cdl_root_dir
        self.batch_size = batch_size
        self.length = length
        self.train_length = train_length if train_length else length
        self.val_length = val_length if val_length else length
        self.test_length = test_length if test_length else length
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.neighborhood = neighborhood
        self.dataset_split = getattr(datamodules, dataset_split)
        self.pin_memory = pin_memory
        self.area_of_interest: Optional[BoundingBox] = None
        self.date_range: Optional[str] = None
        if area_of_interest is not None:
            self.area_of_interest = create_bounding_box(*area_of_interest)
            self.date_range = f"{area_of_interest[4]}/{area_of_interest[5]}"
        self.train_sampler_class = getattr(samplers, train_sampler_class)
        self.val_sampler_class = getattr(samplers, val_sampler_class)
        self.test_sampler_class = getattr(samplers, test_sampler_class)
        self.train_collate_fn = getattr(datasets, train_collate_fn)
        self.val_collate_fn = getattr(datasets, val_collate_fn)
        self.test_collate_fn = getattr(datasets, test_collate_fn)
        self.cache = cache
        self.cache_size = cache_size
        self.block_size = block_size
        self.kwargs = kwargs

    def naip_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the NAIP Dataset.

        Args:
            sample: NAIP image dictionary

        Returns:
            preprocessed NAIP data
        """
        sample["image"] = sample["image"] / 255.0
        sample["image"] = sample["image"].float()

        del sample["bbox"]

        return sample

    def cdl_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the CDL Dataset.

        Args:
            sample: CDL mask dictionary

        Returns:
            preprocessed CDL data
        """
        sample["label"] = sample["mask"].long()[0]
        sample["label"] = sample["label"].mode()[0].mode()[0].unsqueeze(0)

        del sample["bbox"]

        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        NAIP(
            self.naip_root_dir,
            area_of_interest=self.area_of_interest,
            date_range=self.date_range,
            download=False,
        )
        if self.cdl_root_dir is not None:
            CDL(self.cdl_root_dir, download=False, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: state to set up
        """
        naip = NAIP(
            self.naip_root_dir,
            transforms=self.naip_transform,
            cache=self.cache,
            cache_size=self.cache_size,
        )

        self.dataset: GeoDataset
        if self.cdl_root_dir is not None:
            cdl = CDL(
                self.cdl_root_dir, naip.crs, naip.res, transforms=self.cdl_transform
            )
            self.dataset = naip & cdl
        else:
            self.dataset = naip

        if self.area_of_interest is None:
            self.area_of_interest = BoundingBox(*self.dataset.bounds)

        train_roi, val_roi, test_roi = self.dataset_split(
            self.area_of_interest, **self.kwargs
        )

        self.train_sampler = self.train_sampler_class(
            dataset=self.dataset,
            size=self.patch_size,
            neighborhood=self.neighborhood,
            batch_size=self.batch_size,
            length=self.train_length,
            roi=train_roi,
        )

        self.val_sampler = self.val_sampler_class(
            dataset=self.dataset,
            size=self.patch_size,
            neighborhood=self.neighborhood,
            batch_size=self.batch_size,
            length=self.val_length,
            roi=val_roi,
        )

        self.test_sampler = self.test_sampler_class(
            dataset=self.dataset,
            size=self.patch_size,
            block_size=self.block_size,
            batch_size=self.batch_size,
            length=self.test_length,
            roi=test_roi,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        if isinstance(self.train_sampler, GeoSampler):
            return DataLoader(
                self.dataset,
                sampler=self.train_sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=self.pin_memory,
            )
        elif isinstance(self.train_sampler, BatchGeoSampler):
            return DataLoader(
                self.dataset,
                batch_sampler=self.train_sampler,
                num_workers=self.num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        if isinstance(self.val_sampler, GeoSampler):
            return DataLoader(
                self.dataset,
                sampler=self.val_sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.val_collate_fn,
                pin_memory=self.pin_memory,
            )
        elif isinstance(self.val_sampler, BatchGeoSampler):
            return DataLoader(
                self.dataset,
                batch_sampler=self.val_sampler,
                num_workers=self.num_workers,
                collate_fn=self.val_collate_fn,
                pin_memory=self.pin_memory,
            )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        if isinstance(self.test_sampler, GeoSampler):
            return DataLoader(
                self.dataset,
                sampler=self.test_sampler,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.test_collate_fn,
                pin_memory=self.pin_memory,
            )
        elif isinstance(self.test_sampler, BatchGeoSampler):
            return DataLoader(
                self.dataset,
                batch_sampler=self.test_sampler,
                num_workers=self.num_workers,
                collate_fn=self.test_collate_fn,
                pin_memory=self.pin_memory,
            )
