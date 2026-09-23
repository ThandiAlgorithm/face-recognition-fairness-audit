from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def _conv_bn(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    leaky: float = 0.0,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(
            negative_slope=leaky,
            inplace=True,
        ),
    )


def _depthwise_separable(
    in_channels: int,
    out_channels: int,
    stride: int,
    leaky: float = 0.1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        ),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(
            negative_slope=leaky,
            inplace=True,
        ),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(
            negative_slope=leaky,
            inplace=True,
        ),
    )


class MobileNetV1_025(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            _conv_bn(
                3,
                8,
                stride=2,
                leaky=0.1,
            ),
            _depthwise_separable(
                8,
                16,
                stride=1,
            ),
            _depthwise_separable(
                16,
                32,
                stride=2,
            ),
            _depthwise_separable(
                32,
                32,
                stride=1,
            ),
            _depthwise_separable(
                32,
                64,
                stride=2,
            ),
            _depthwise_separable(
                64,
                64,
                stride=1,
            ),
        )

        self.stage2 = nn.Sequential(
            _depthwise_separable(
                64,
                128,
                stride=2,
            ),
            _depthwise_separable(
                128,
                128,
                stride=1,
            ),
            _depthwise_separable(
                128,
                128,
                stride=1,
            ),
            _depthwise_separable(
                128,
                128,
                stride=1,
            ),
            _depthwise_separable(
                128,
                128,
                stride=1,
            ),
            _depthwise_separable(
                128,
                128,
                stride=1,
            ),
        )

        self.stage3 = nn.Sequential(
            _depthwise_separable(
                128,
                256,
                stride=2,
            ),
            _depthwise_separable(
                256,
                256,
                stride=1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        return stage1, stage2, stage3


class RetinaFaceBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        backbone = MobileNetV1_025()

        self.stage1 = backbone.stage1
        self.stage2 = backbone.stage2
        self.stage3 = backbone.stage3

    def forward(
        self,
        x: torch.Tensor,
    ) -> OrderedDict[str, torch.Tensor]:
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        return OrderedDict(
            [
                ("1", stage1),
                ("2", stage2),
                ("3", stage3),
            ]
        )
