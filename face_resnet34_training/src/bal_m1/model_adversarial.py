"""BAL-M2: adversarial debiasing variant of BAL-M1. Race and skin tone
remain direct supervised targets straight off the shared backbone feature,
exactly as in BAL-M1 -- they are themselves what the model should predict
well, so there is nothing to adversarially remove from them. Gender and
age instead route through a small shared adapter, and a GRL-reversed
adversary tries to recover race/skin_tone from THAT adapter's
representation: the adapter is pushed to drop race/skin-tone-correlated
shortcuts gender/age might otherwise lean on (the cosmetics-as-gender-
proxy finding is exactly this kind of shortcut), without touching the
race/skin_tone heads' own training signal at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from resnet34 import ResNet34FromScratch  # noqa: E402

FEATURE_DIM = 512
ADAPTER_DIM = 256


class GradientReversalFunction(torch.autograd.Function):
    """Copied, not imported, from src/frs_m0/model.py's class of the same
    name -- that module and this one each have their own train.py, and
    cross-importing risks the sys.path collision documented there."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class BalM1AdversarialModel(nn.Module):
    def __init__(self, n_race: int, n_gender: int, n_age: int, n_skin_tone: int, adv_lambda: float = 1.0):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.race_head = nn.Linear(FEATURE_DIM, n_race)
        self.skin_tone_head = nn.Linear(FEATURE_DIM, n_skin_tone)

        self.adapter = nn.Sequential(nn.Linear(FEATURE_DIM, ADAPTER_DIM), nn.ReLU(inplace=True))
        self.gender_head = nn.Linear(ADAPTER_DIM, n_gender)
        self.age_head = nn.Linear(ADAPTER_DIM, n_age)

        self.grl = GradientReversalLayer(adv_lambda)
        self.adv_race_head = nn.Linear(ADAPTER_DIM, n_race)
        self.adv_skin_tone_head = nn.Linear(ADAPTER_DIM, n_skin_tone)

    def set_lambda(self, lambda_: float) -> None:
        self.grl.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        adapter_out = self.adapter(features)
        reversed_adapter = self.grl(adapter_out)
        return {
            "race": self.race_head(features),
            "skin_tone": self.skin_tone_head(features),
            "gender": self.gender_head(adapter_out),
            "age": self.age_head(adapter_out),
            "adv_race": self.adv_race_head(reversed_adapter),
            "adv_skin_tone": self.adv_skin_tone_head(reversed_adapter),
        }


def build_bal_m1_adversarial_model(n_race: int, n_gender: int, n_age: int, n_skin_tone: int, adv_lambda: float = 1.0) -> nn.Module:
    return BalM1AdversarialModel(n_race, n_gender, n_age, n_skin_tone, adv_lambda)
