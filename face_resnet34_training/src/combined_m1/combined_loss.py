"""Combined-M1 loss: CrossEntropyLoss(ignore_index=-1) per classification
head (gender/age_harmonized/race_harmonized/skin_tone), same convention as
src/training/losses.py, plus a masked BCEWithLogitsLoss for lfw_attributes.

Unlike the single-dataset losses.py (whose docstring notes an all-ignored
batch is "not expected in practice"), a combined batch can plausibly have
zero valid labels for one task -- e.g. an unlucky batch drawing no
FairFace/UTKFace rows leaves age_harmonized entirely -1, since LFW has no
age label at all. CrossEntropyLoss's mean reduction divides by zero in
that case, so each task is skipped (contributes an honest zero, not NaN)
when its batch has no valid labels, rather than assuming that can't happen.
"""
from __future__ import annotations

import torch
import torch.nn as nn

IGNORE_INDEX = -1
CLASSIFICATION_TASKS = ["gender", "age_harmonized", "race_harmonized", "skin_tone"]


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        losses = {}
        for task in CLASSIFICATION_TASKS:
            valid = targets[task] != IGNORE_INDEX
            if valid.any():
                losses[task] = self.ce(outputs[task], targets[task])
            else:
                losses[task] = torch.zeros((), device=outputs[task].device)

        attr_logits = outputs["lfw_attributes"]
        attr_targets = targets["lfw_attributes"]
        attr_mask = targets["lfw_attributes_mask"]
        elementwise = self.bce(attr_logits, attr_targets)
        denom = attr_mask.sum().clamp(min=1.0)
        losses["lfw_attributes"] = (elementwise * attr_mask).sum() / denom

        losses["total"] = sum(losses[t] for t in [*CLASSIFICATION_TASKS, "lfw_attributes"])
        return losses
