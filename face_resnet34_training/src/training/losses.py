"""Step 9b: per-task loss functions.

CrossEntropyLoss(ignore_index=-1) for FairFace/UTKFace's race/gender/age_group
heads: -1 is FaceDataset's sentinel for a sample whose label is unknown or
missing (see face_dataset.py / label_encoders.py), so it contributes zero
loss for that task instead of being dropped from the batch or trained
against a fabricated class. Ignore-index rate is tiny (4/24108 for UTKFace,
0 elsewhere) so an all-ignored batch -- which would make CrossEntropyLoss's
mean reduction divide by zero -- isn't expected in practice; not special-cased.

BCEWithLogitsLoss for LFW's multi-label attribute head (73-way, LFWA+).
"""
from __future__ import annotations

import torch.nn as nn

IGNORE_INDEX = -1
FAIRFACE_UTKFACE_TASKS = ["race", "gender", "age_group"]


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list[str] = FAIRFACE_UTKFACE_TASKS):
        super().__init__()
        self.tasks = tasks
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        losses = {task: self.ce(outputs[task], targets[task]) for task in self.tasks}
        losses["total"] = sum(losses[task] for task in self.tasks)
        return losses


class AttributeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        loss = self.bce(outputs["attributes"], targets["attributes"])
        return {"attributes": loss, "total": loss}


def build_loss(dataset: str) -> nn.Module:
    if dataset == "lfw":
        return AttributeLoss()
    return MultiTaskLoss()
