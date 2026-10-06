"""Step 9: task heads on top of the from-scratch ResNet-34 backbone.

One shared backbone per model; linear heads only -- a baseline doesn't need
anything heavier stacked on top of the 512-d pooled feature vector.

FairFace and UTKFace each get three classification heads (race, gender,
age_group) over that dataset's own frozen label vocabulary
(configs/<dataset>_labels.json). LFW gets a single multi-label head over
the LFWA+ attributes (73-way, lfw_att_73.mat) -- this includes native
Male/Asian/White/Black/Indian flags, so gender and race ARE trained
targets for LFW too, just as two of the flat attribute set rather than as
separate heads. race_label/gender_label elsewhere in this codebase are a
second, independent source (derived demographics metadata) used only for
subgroup/fairness evaluation, never fed into training (see face_dataset.py
/ records.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from resnet34 import ResNet34FromScratch

FEATURE_DIM = 512


class MultiTaskResNet34(nn.Module):
    def __init__(self, task_num_classes: dict[str, int]):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.heads = nn.ModuleDict(
            {task: nn.Linear(FEATURE_DIM, n) for task, n in task_num_classes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {task: head(features) for task, head in self.heads.items()}


class AttributeResNet34(nn.Module):
    def __init__(self, n_attributes: int):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.head = nn.Linear(FEATURE_DIM, n_attributes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {"attributes": self.head(features)}


def build_model(dataset: str, labels: dict) -> nn.Module:
    if dataset == "lfw":
        return AttributeResNet34(n_attributes=len(labels["attributes"]))
    task_num_classes = {task: len(categories) for task, categories in labels.items()}
    return MultiTaskResNet34(task_num_classes)
