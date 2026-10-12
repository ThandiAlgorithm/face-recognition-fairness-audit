"""M0 model: ResNet-34 backbone -> 512-d L2-normalized embedding.
No pretrained weights (project convention).

Two training heads, selectable via train.py's --loss:
- M0EmbeddingModel (plain CrossEntropy identity head): M0 v1, deliberately
  the simplest version that produces a genuine identity embedding. Trains
  the network to guess "which training identity is this?" and discards
  the guessing head afterward -- an indirect way to learn a good
  embedding, since the training signal never directly rewards genuine
  pairs being close and impostor pairs being far apart.
- M0ArcFaceEmbeddingModel (ArcMarginProduct head): M0 v2. Directly
  optimizes the thing verification actually needs -- see ArcMarginProduct's
  docstring. Built after the v1 deep dive (see
  results/m0_identity_audit/deep_dive.md) found genuine- and impostor-pair
  similarity scores barely separate (mean 0.974 vs 0.967) -- a sign the
  plain-softmax embedding never learned to spread identities apart, which
  an angular margin loss is specifically designed to fix.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from resnet34 import ResNet34FromScratch

EMBEDDING_DIM = 512


class M0EmbeddingModel(nn.Module):
    """backbone -> 512-d embedding -> L2 normalize. This alone is what
    extract_embeddings.py uses at inference time; the identity_head below
    is training-only scaffolding, thrown away for verification."""

    def __init__(self, n_train_identities: int):
        super().__init__()
        self.backbone = ResNet34FromScratch()  # already outputs flattened 512-d pooled features
        self.identity_head = nn.Linear(EMBEDDING_DIM, n_train_identities)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.embed(x)
        logits = self.identity_head(embedding)
        return {"embedding": embedding, "identity_logits": logits}


class ArcMarginProduct(nn.Module):
    """ArcFace additive angular margin head (Deng et al. 2019).

    Plain softmax only asks "is the correct class's logit the largest?" --
    it stops pushing as soon as that's technically true, with no pressure
    to separate classes by any particular margin. ArcFace instead adds a
    fixed angular penalty (`margin`, in radians) to the TRUE class's angle
    before computing the softmax, so the network has to work harder (push
    the correct class's angle in further) to achieve the same loss -- the
    side effect is same-identity embeddings cluster tightly and
    different-identity embeddings end up well separated, which is exactly
    what cosine-similarity verification at test time needs.

    `scale` (temperature on the final cosine logits) and `margin` are the
    standard values from the ArcFace paper (s=64, m=0.5) -- not tuned here,
    since this is the first attempt at this loss for M0, not a sweep.

    Training-only, like M0EmbeddingModel's identity_head -- embed() never
    touches this; it's discarded for verification the same way.
    """

    def __init__(self, in_features: int, n_classes: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # Precomputed trig constants + the standard ArcFace numerical-stability
        # guard: beyond this angle, cos(theta + m) would start increasing again
        # (past pi), which would reward the wrong direction -- fall back to a
        # linear penalty (cosine - mm) instead of the true margin there.
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
        w = nn.functional.normalize(self.weight, p=2, dim=-1)
        cosine = nn.functional.linear(embedding, w)  # embedding is already L2-normalized
        if labels is None:
            # Eval-time / no-label forward (e.g. a sanity check outside training):
            # plain scaled cosine logits, margin only ever applies during training.
            return cosine * self.scale
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0, max=1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.scale


class M0ArcFaceEmbeddingModel(nn.Module):
    """Same backbone/embedding as M0EmbeddingModel; identity head replaced
    with ArcMarginProduct. See module docstring for why."""

    def __init__(self, n_train_identities: int, scale: float = 64.0, margin: float = 0.5):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.arcface_head = ArcMarginProduct(EMBEDDING_DIM, n_train_identities, scale, margin)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedding = self.embed(x)
        logits = self.arcface_head(embedding, labels)
        return {"embedding": embedding, "identity_logits": logits}


class GradientReversalFunction(torch.autograd.Function):
    """Identical to src/training/mitigation_m4.py's class of the same name
    (the already-debugged FairFace/UTKFace/LFW M4 implementation) -- copied
    rather than imported, since that module does `from train import ...`,
    which would collide with this directory's own train.py (see train.py's
    _WeightedSampler docstring for the same issue with mitigation_m2.py)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class M0AdversarialEmbeddingModel(nn.Module):
    """M0 v3: backbone + ArcFace identity head (same as M0ArcFaceEmbeddingModel)
    + gradient-reversal adversary heads for race, gender, and age. Deferred
    from the original MULTI-mitigation plan (M1-M4's classifiers stayed
    ITA-only, because training a model to both PREDICT a demographic
    attribute as its main task and simultaneously ERASE that same
    attribute from its features is self-contradictory) -- M0 never
    predicts race/gender/age as its main task (identity verification is),
    so no such contradiction applies here, and the full race+gender+age
    adversarial treatment originally planned for M0 can finally be built.

    Each adversary head (plain nn.Linear, matching mitigation_m4.py's
    single-linear-layer ita_head, not an MLP) tries to predict its
    attribute FROM the embedding; GradientReversalLayer flips and scales
    the gradient flowing back from each head into the backbone, so the
    backbone is pushed to make the embedding WORSE at revealing race/
    gender/age (while the identity loss simultaneously keeps it good at
    verification) -- the standard adversarial-debiasing minimax (Zhang
    et al. 2018; Ganin & Lempitsky 2015's original DANN formulation for
    the gradient-reversal mechanism itself).

    One shared `grl.lambda_` controls all three heads at once (set
    externally, per-epoch, by train.py's DANN sigmoid ramp -- see that
    file's docstring for why a RAMPED lambda, not a fixed one, is required
    for stability; this is the exact fix already found necessary for
    FairFace's M4 and reused here rather than rediscovered).

    Missing demographic labels (e.g. LFW has no age_group at all) use
    ignore_index=-1 in the adversary loss, same convention as every other
    ignore_index use in this project -- never fabricated, never imputed.
    """

    def __init__(self, n_train_identities: int, n_race: int, n_gender: int, n_age: int,
                 scale: float = 64.0, margin: float = 0.5, adv_lambda: float = 1.0):
        super().__init__()
        self.backbone = ResNet34FromScratch()
        self.arcface_head = ArcMarginProduct(EMBEDDING_DIM, n_train_identities, scale, margin)
        self.grl = GradientReversalLayer(adv_lambda)
        self.race_head = nn.Linear(EMBEDDING_DIM, n_race)
        self.gender_head = nn.Linear(EMBEDDING_DIM, n_gender)
        self.age_head = nn.Linear(EMBEDDING_DIM, n_age)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.nn.functional.normalize(features, p=2, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embedding = self.embed(x)
        id_logits = self.arcface_head(embedding, labels)
        reversed_emb = self.grl(embedding)
        return {
            "embedding": embedding,
            "identity_logits": id_logits,
            "race_logits": self.race_head(reversed_emb),
            "gender_logits": self.gender_head(reversed_emb),
            "age_logits": self.age_head(reversed_emb),
        }
