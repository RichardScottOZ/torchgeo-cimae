# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) dataset."""

import os
from typing import Any, Callable, Dict, Optional, Union

import matplotlib.pyplot as plt
from rasterio.crs import CRS
from shapely.geometry import Polygon

from .geo import RasterDataset
from .utils import BoundingBox, download_url, search_stac


class NAIP(RasterDataset):
    """National Agriculture Imagery Program (NAIP) dataset.

    The `National Agriculture Imagery Program (NAIP)
    <https://www.fsa.usda.gov/programs-and-services/aerial-photography/imagery-programs/naip-imagery/>`_
    acquires aerial imagery during the agricultural growing seasons in the continental
    U.S. A primary goal of the NAIP program is to make digital ortho photography
    available to governmental agencies and the public within a year of acquisition.

    NAIP is administered by the USDA's Farm Service Agency (FSA) through the Aerial
    Photography Field Office in Salt Lake City. This "leaf-on" imagery is used as a base
    layer for GIS programs in FSA's County Service Centers, and is used to maintain the
    Common Land Unit (CLU) boundaries.

    If you use this dataset in your research, please cite it using the following format:

    * https://www.fisheries.noaa.gov/inport/item/49508/citation
    """

    # https://www.nrcs.usda.gov/Internet/FSE_DOCUMENTS/nrcs141p2_015644.pdf
    # https://planetarycomputer.microsoft.com/dataset/naip#Storage-Documentation
    filename_glob = "*m_*.*"
    filename_regex = r"""
        ^m
        _(?P<quadrangle>\d+)
        _(?P<quarter_quad>[a-z]+)
        _(?P<utm_zone>\d+)
        _(?P<resolution>(\d+|[a-z]+))
        _(?P<date>\d+)
        (?:_(?P<processing_date>\d+))?
        \..*$
    """

    # Plotting
    all_bands = ["R", "G", "B", "NIR"]
    rgb_bands = ["R", "G", "B"]

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        cache_size: int = 128,
        area_of_interest: Optional[Union[BoundingBox, Polygon]] = None,
        date_range: Optional[str] = None,
        download: bool = True,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            area_of_interest: BoundingBox or Polygon of interest
            date_range: Range of time to search in as string in format ``year-month-day/year-month-day``
            download: if True, download dataset and store it in the root directory

        Raises:
            FileNotFoundError: if no files are found in ``root``
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        self.root = root
        self.area_of_interest = area_of_interest
        self.date_range = date_range
        self.download = download

        self._verify()

        super().__init__(root, crs, res, transforms, cache, cache_size)

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        # Check if the zip file has already been downloaded
        if self.area_of_interest is None:
            return

        if self.date_range is None:
            self.date_range = "1990-01-01/2100-01-01"

        items = search_stac("naip", self.area_of_interest, self.date_range)

        for item in items:
            url = item.get_assets()["image"].href
            filename = url.split("/")[-1]

            if os.path.exists(os.path.join(self.root, filename)):
                continue

            # Check if the user requested to download the dataset
            if not self.download:
                raise RuntimeError(
                    f"File not found in `root={self.root}` and `download=False`, "
                    "either specify a different `root` directory"
                    " or use `download=True` to automaticaly download the dataset."
                )

            download_url(url, self.root, filename)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
            Method now takes a sample dict, not a Tensor. Additionally, possible to
            show subplot titles and/or use a custom suptitle.
        """
        image = sample["image"][0:3, :, :].permute(1, 2, 0)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        ax.imshow(image)
        ax.axis("off")
        if show_titles:
            ax.set_title("Image")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
