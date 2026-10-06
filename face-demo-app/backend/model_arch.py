import torch.nn as nn
from torchvision.models import resnet34


class MultiHeadResNet(nn.Module):
    def __init__(self, n_age, n_race, with_race=True):
        super().__init__()
        base = resnet34(weights=None)
        feat = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.age_head = nn.Linear(feat, n_age)
        self.gender_head = nn.Linear(feat, 2)
        self.with_race = with_race
        self.race_head = nn.Linear(feat, n_race) if with_race else None

    def forward(self, x):
        f = self.backbone(x)
        out = {"age": self.age_head(f), "gender": self.gender_head(f)}
        if self.with_race:
            out["race"] = self.race_head(f)
        return out
