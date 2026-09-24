from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .backbone import RetinaFaceBody


MNET_CONFIG = {
    "name": "mobilenet0.25",
    "min_sizes": [
        [16, 32],
        [64, 128],
        [256, 512],
    ],
    "steps": [
        8,
        16,
        32,
    ],
    "variance": [
        0.1,
        0.2,
    ],
    "clip": False,
    "in_channel": 32,
    "out_channel": 64,
}


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


def _conv_bn_no_relu(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
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
    )


def _conv_bn_1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    leaky: float = 0.0,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(
            negative_slope=leaky,
            inplace=True,
        ),
    )


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        leaky = (
            0.1
            if out_channels <= 64
            else 0.0
        )

        self.output1 = _conv_bn_1x1(
            in_channels[0],
            out_channels,
            leaky=leaky,
        )

        self.output2 = _conv_bn_1x1(
            in_channels[1],
            out_channels,
            leaky=leaky,
        )

        self.output3 = _conv_bn_1x1(
            in_channels[2],
            out_channels,
            leaky=leaky,
        )

        self.merge1 = _conv_bn(
            out_channels,
            out_channels,
            leaky=leaky,
        )

        self.merge2 = _conv_bn(
            out_channels,
            out_channels,
            leaky=leaky,
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        output1 = self.output1(
            inputs[0]
        )

        output2 = self.output2(
            inputs[1]
        )

        output3 = self.output3(
            inputs[2]
        )

        up3 = F.interpolate(
            output3,
            size=(
                output2.shape[2],
                output2.shape[3],
            ),
            mode="nearest",
        )

        output2 = self.merge2(
            output2 + up3
        )

        up2 = F.interpolate(
            output2,
            size=(
                output1.shape[2],
                output1.shape[3],
            ),
            mode="nearest",
        )

        output1 = self.merge1(
            output1 + up2
        )

        return [
            output1,
            output2,
            output3,
        ]


class SSH(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        if out_channels % 4 != 0:
            raise ValueError(
                "SSH out_channels must be divisible by four."
            )

        leaky = (
            0.1
            if out_channels <= 64
            else 0.0
        )

        self.conv3X3 = _conv_bn_no_relu(
            in_channels,
            out_channels // 2,
        )

        self.conv5X5_1 = _conv_bn(
            in_channels,
            out_channels // 4,
            leaky=leaky,
        )

        self.conv5X5_2 = _conv_bn_no_relu(
            out_channels // 4,
            out_channels // 4,
        )

        self.conv7X7_2 = _conv_bn(
            out_channels // 4,
            out_channels // 4,
            leaky=leaky,
        )

        self.conv7x7_3 = _conv_bn_no_relu(
            out_channels // 4,
            out_channels // 4,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        conv3 = self.conv3X3(x)

        conv5_1 = self.conv5X5_1(x)
        conv5 = self.conv5X5_2(
            conv5_1
        )

        conv7 = self.conv7x7_3(
            self.conv7X7_2(
                conv5_1
            )
        )

        return F.relu(
            torch.cat(
                [
                    conv3,
                    conv5,
                    conv7,
                ],
                dim=1,
            )
        )


class ClassHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        anchors: int = 3,
    ) -> None:
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels,
            anchors * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1x1(x)

        output = output.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        return output.view(
            output.shape[0],
            -1,
            2,
        )


class BboxHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        anchors: int = 3,
    ) -> None:
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels,
            anchors * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1x1(x)

        output = output.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        return output.view(
            output.shape[0],
            -1,
            4,
        )


class LandmarkHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 512,
        anchors: int = 3,
    ) -> None:
        super().__init__()

        self.conv1x1 = nn.Conv2d(
            in_channels,
            anchors * 10,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        output = self.conv1x1(x)

        output = output.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        return output.view(
            output.shape[0],
            -1,
            10,
        )


class RetinaFace(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.body = RetinaFaceBody()

        self.fpn = FPN(
            [
                64,
                128,
                256,
            ],
            64,
        )

        self.ssh1 = SSH(
            64,
            64,
        )

        self.ssh2 = SSH(
            64,
            64,
        )

        self.ssh3 = SSH(
            64,
            64,
        )

        self.ClassHead = nn.ModuleList(
            [
                ClassHead(
                    64,
                    anchors=2,
                )
                for _ in range(3)
            ]
        )

        self.BboxHead = nn.ModuleList(
            [
                BboxHead(
                    64,
                    anchors=2,
                )
                for _ in range(3)
            ]
        )

        self.LandmarkHead = nn.ModuleList(
            [
                LandmarkHead(
                    64,
                    anchors=2,
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        body_features = self.body(x)

        pyramid = self.fpn(
            list(
                body_features.values()
            )
        )

        features = [
            self.ssh1(
                pyramid[0]
            ),
            self.ssh2(
                pyramid[1]
            ),
            self.ssh3(
                pyramid[2]
            ),
        ]

        boxes = torch.cat(
            [
                self.BboxHead[index](
                    feature
                )
                for index, feature
                in enumerate(features)
            ],
            dim=1,
        )

        classes = torch.cat(
            [
                self.ClassHead[index](
                    feature
                )
                for index, feature
                in enumerate(features)
            ],
            dim=1,
        )

        landmarks = torch.cat(
            [
                self.LandmarkHead[index](
                    feature
                )
                for index, feature
                in enumerate(features)
            ],
            dim=1,
        )

        probabilities = torch.softmax(
            classes,
            dim=-1,
        )

        return (
            boxes,
            probabilities,
            landmarks,
        )
