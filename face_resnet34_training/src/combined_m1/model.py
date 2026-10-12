"""Combined-M1 model: one shared from-scratch ResNet-34 backbone, five
linear heads on the 512-d pooled feature (same backbone/head-on-top
convention as heads.py's MultiTaskResNet34/AttributeResNet34 -- this is
their multi-dataset generalization, not a different architecture)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from resnet34 import ResNet34FromScratch

FEATURE_DIM = 512


class CombinedResNet34(nn.Module):
    def __init__(self, n_age: int, n_race: int, n_lfw_attributes: int):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.heads = nn.ModuleDict({
            "gender": nn.Linear(FEATURE_DIM, 2),
            "age_harmonized": nn.Linear(FEATURE_DIM, n_age),
            "race_harmonized": nn.Linear(FEATURE_DIM, n_race),
            "skin_tone": nn.Linear(FEATURE_DIM, 2),
            "lfw_attributes": nn.Linear(FEATURE_DIM, n_lfw_attributes),
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}
