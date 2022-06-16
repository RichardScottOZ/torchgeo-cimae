# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

from typing import Any, List, Type, Union

import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import (
    BasicBlock,
    Bottleneck,
    ResNet,
    _resnet as _resnet_tv,
)

MODEL_URLS = {
    "sentinel2": {
        "all": {
            "resnet50": "https://zenodo.org/record/5610000/files/resnet50-sentinel2.pt"
        }
    }
}


IN_CHANNELS = {"sentinel2": {"all": 10}, "naip": {"all": 4}}

NUM_CLASSES = {"sentinel2": 17, "naip": 0}


def _resnet(
    sensor: str,
    bands: str,
    arch: str,
    block: Union[BasicBlock, Bottleneck],
    layers: List[int],
    pretrained: bool,
    imagenet_pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    """Resnet model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    Args:
        sensor: imagery source which determines number of input channels
        bands: which spectral bands to consider: "all", "rgb", etc.
        arch: ResNet version specifying number of layers
        block: type of network block
        layers: number of layers per block
        pretrained: if True, returns a model pre-trained on ``sensor`` imagery
        progress: if True, displays a progress bar of the download to stderr

    Returns:
        A ResNet model
    """
    # Initialize a new model
    num_classes = NUM_CLASSES[sensor] if not imagenet_pretrained else 1000
    model = _resnet_tv(
        arch,
        block,
        layers,
        pretrained=imagenet_pretrained,
        num_classes=num_classes,
        progress=progress,
        **kwargs,
    )

    # Replace the first layer with the correct number of input channels
    first_layer = model.conv1
    new_first_layer = nn.Conv2d(
        IN_CHANNELS[sensor][bands],
        out_channels=first_layer.out_channels,
        kernel_size=first_layer.kernel_size,
        stride=first_layer.stride,
        padding=first_layer.padding,
        bias=first_layer.bias,
    )

    if imagenet_pretrained:
        # initialize the weights from new channel with the red channel weights
        copy_weights = 0
        # Copying the weights from the old to the new layer
        new_first_layer.weight[
            :, : first_layer.in_channels
        ].data = first_layer.weight.clone()
        # Copying the weights of the old layer to the extra channels
        for channel_index in range(first_layer.in_channels, IN_CHANNELS[sensor][bands]):
            new_first_layer.weight[
                :, channel_index : channel_index + 1
            ].data = first_layer.weight[:, copy_weights : copy_weights + 1].clone()

    model.conv1 = new_first_layer

    # Load pretrained weights
    # TODO: Add naip model_url
    if pretrained:
        state_dict = load_state_dict_from_url(  # type: ignore[no-untyped-call]
            MODEL_URLS[sensor][bands][arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def resnet18(
    sensor: str,
    bands: str,
    block: Union[BasicBlock, Bottleneck] = BasicBlock,
    layers: List[int] = [2, 2, 2, 2],
    pretrained: bool = False,
    imagenet_pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-18 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    Args:
        sensor: imagery source which determines number of input channels
        bands: which spectral bands to consider: "all", "rgb", etc.
        pretrained: if True, returns a model pre-trained on ``sensor`` imagery
        progress: if True, displays a progress bar of the download to stderr

    Returns:
        A ResNet-18 model
    """
    return _resnet(
        sensor,
        bands,
        "resnet18",
        block,
        layers,
        pretrained,
        imagenet_pretrained,
        progress,
        **kwargs,
    )


def resnet50(
    sensor: str,
    bands: str,
    pretrained: bool = False,
    imagenet_pretrained: bool = False,
    progress: bool = True,
    **kwargs: Any,
) -> ResNet:
    """ResNet-50 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/pdf/1512.03385.pdf

    Args:
        sensor: imagery source which determines number of input channels
        bands: which spectral bands to consider: "all", "rgb", etc.
        pretrained: if True, returns a model pre-trained on ``sensor`` imagery
        progress: if True, displays a progress bar of the download to stderr

    Returns:
        A ResNet-50 model
    """
    return _resnet(
        sensor,
        bands,
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        imagenet_pretrained,
        progress,
        **kwargs,
    )
