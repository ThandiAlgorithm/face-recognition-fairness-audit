"""ResNet-34 built from scratch. No torchvision.models.resnet34, no checkpoint
loading of any kind -- the architecture and every weight are trained here.

Backbone only: returns the 512-d pooled feature vector. Task heads (linear
layers on top of that vector) live in heads.py, not here, so the backbone
stays identical across all three datasets and only the heads differ.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv3x3 -> BN, add shortcut, ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: identity unless stride != 1 or channels change, in which
        # case it's a 1x1 conv with the same stride, followed by BN.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet34FromScratch(nn.Module):
    BLOCK_COUNTS = (3, 4, 6, 3)
    CHANNELS = (64, 128, 256, 512)

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, self.BLOCK_COUNTS[0], stride=1)
        self.layer2 = self._make_layer(64, 128, self.BLOCK_COUNTS[1], stride=2)
        self.layer3 = self._make_layer(128, 256, self.BLOCK_COUNTS[2], stride=2)
        self.layer4 = self._make_layer(256, 512, self.BLOCK_COUNTS[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.pool(x), 1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sanity_check(model: ResNet34FromScratch | None = None, device: str = "cpu") -> dict:
    """Run the spec's start-up sanity checks. Returns a dict suitable for
    logging into environment.json / the safety summary. Raises AssertionError
    if anything is off."""
    model = model or ResNet34FromScratch()
    model = model.to(device)
    model.eval()

    n_params = count_parameters(model)
    print(f"Parameter count: {n_params:,}")
    assert 21_000_000 <= n_params <= 21_600_000, (
        f"Expected the backbone to have about 21.3M parameters, got {n_params:,}."
    )

    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print(f"Dummy forward pass output shape: {tuple(out.shape)}")
    assert tuple(out.shape) == (2, 512), f"Expected output shape (2, 512), got {tuple(out.shape)}."

    print("Pretrained weights: NO")

    return {"backbone_parameter_count": n_params, "pretrained": False}


if __name__ == "__main__":
    sanity_check()
