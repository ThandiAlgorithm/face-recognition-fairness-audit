"""BAL-M1 model: one shared ResNet-34 backbone, four linear heads (race,
gender, age, skin_tone) -- exactly src/models/heads.py's existing
MultiTaskResNet34, which already does this. No new architecture needed;
this module just wires up BAL-M1's specific four task names against
vocab sizes determined from the actual manifest (skin_tone's band count
depends on what fitzpatrick_approx values are present)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from heads import MultiTaskResNet34  # noqa: E402


def build_bal_m1_model(n_race: int, n_gender: int, n_age: int, n_skin_tone: int) -> nn.Module:
    return MultiTaskResNet34({"race": n_race, "gender": n_gender, "age": n_age, "skin_tone": n_skin_tone})
