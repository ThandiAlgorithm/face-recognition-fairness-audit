"""M4 -- adversarial ITA removal (gradient reversal).

Architecture-level intervention, unlike M2 (data-level) and M3
(loss-level): a small discriminator head is attached to the shared
backbone's 512-d feature vector, trained to predict the sample's ITA
group (<28 / ≥28). Between the backbone and that head sits a Gradient
Reversal Layer (GRL, Ganin & Lempitsky 2015) -- identity in the forward
pass, but multiplies the upstream gradient by -lambda in the backward
pass. The net effect: the discriminator is trained normally to predict
ITA as well as it can, while that same gradient, negated, pushes the
*backbone* to produce features the discriminator predicts ITA from
*worse* over time. Everything else -- optimizer, scheduler, batches,
main task loss, validation/test data -- is identical to M1, so if
fairness changes, the gradient-reversal mechanism is the only thing that
can be credited, consistent with M2/M3's isolation design.

Missing ITA is never inferred: those rows use CrossEntropyLoss's
ignore_index for the adversarial term only (the main task loss is
untouched by this), so they stay in the training pool at full weight for
the task they were already being used for -- same philosophy as M2 and
M3's treatment of unknown ITA.

lambda (--adv-lambda) is RAMPED across training, not fixed, following the
original DANN schedule: lambda(p) = adv_lambda * (2/(1+exp(-10p)) - 1),
where p is the fraction of max_epochs completed -- near 0 at the start,
approaching adv_lambda by the end. This is not a stylistic choice: an
earlier version of this file used a fixed lambda from epoch 0, and on real
data (not the synthetic unit-test tensors) it diverged catastrophically on
both FairFace and UTKFace -- loss exploding from the millions into the
quintillions within ~10 epochs, a textbook instance of the well-known GAN/
DANN failure mode where full adversarial pressure before the backbone has
learned reasonable features sends the minimax game into runaway feedback.
Ramping gives the backbone time to settle into a sane feature space before
the adversarial pressure ramps up. --grad-clip-norm (default 5.0) is a
second, independent safety net on top of this.

Logs, mirroring M2/M3's per-run artifacts:
  adversarial_config.json   -- static: lambda, ITA class mapping, formula
  adversarial_history.csv   -- per epoch (training batches only): mean
                                 adversarial loss, discriminator accuracy
                                 (trending toward chance = 50% is the
                                 signal the backbone is becoming ITA-blind)

Checkpoints save the adversarial head's weights too (state_dict is a
strict superset of the plain model's keys: backbone.*, heads.*/head.*,
plus ita_head.* -- the GRL itself holds no parameters). evaluate.py loads
with strict=False specifically so these extra keys don't break loading a
plain model for test-set evaluation; the discriminator itself is never
evaluated at test time.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from label_encoders import load_labels
from heads import MultiTaskResNet34, AttributeResNet34, FEATURE_DIM

from train import train_dataset, EXPERIMENTS_DIR
from losses import IGNORE_INDEX, FAIRFACE_UTKFACE_TASKS, build_loss as build_base_loss

ITA_CLASS_INDEX = {"<28": 0, "≥28": 1}
N_ITA_CLASSES = 2


def _ita_class(value) -> int:
    if value in ("≥28", ">=28"):
        return ITA_CLASS_INDEX["≥28"]
    if value == "<28":
        return ITA_CLASS_INDEX["<28"]
    return IGNORE_INDEX


class GradientReversalFunction(torch.autograd.Function):
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


class AdversarialMultiTaskResNet34(MultiTaskResNet34):
    """FairFace/UTKFace: adds an ITA-group discriminator on top of the
    existing race/gender/age_group heads."""

    def __init__(self, task_num_classes: dict[str, int], grl_lambda: float):
        super().__init__(task_num_classes)
        self.grl = GradientReversalLayer(grl_lambda)
        self.ita_head = nn.Linear(FEATURE_DIM, N_ITA_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs = {task: head(features) for task, head in self.heads.items()}
        outputs["ita_adv"] = self.ita_head(self.grl(features))
        return outputs


class AdversarialAttributeResNet34(AttributeResNet34):
    """LFW: adds an ITA-group discriminator on top of the existing
    multi-label attribute head."""

    def __init__(self, n_attributes: int, grl_lambda: float):
        super().__init__(n_attributes)
        self.grl = GradientReversalLayer(grl_lambda)
        self.ita_head = nn.Linear(FEATURE_DIM, N_ITA_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        outputs = {"attributes": self.head(features)}
        outputs["ita_adv"] = self.ita_head(self.grl(features))
        return outputs


def build_adversarial_model(dataset: str, labels: dict, grl_lambda: float) -> nn.Module:
    if dataset == "lfw":
        return AdversarialAttributeResNet34(n_attributes=len(labels["attributes"]), grl_lambda=grl_lambda)
    task_num_classes = {task: len(categories) for task, categories in labels.items()}
    return AdversarialMultiTaskResNet34(task_num_classes, grl_lambda=grl_lambda)


class _EpochAdvTracker:
    """Accumulates the discriminator's training-pass loss/accuracy across
    an epoch. Train-only, same reasoning as M3's _EpochWeightTracker: the
    shared loss_fn also runs validation batches, and this mechanism's own
    diagnostic (is the discriminator being fooled?) is only meaningful for
    the pass that actually updates weights."""

    def __init__(self):
        self._loss_sum = 0.0
        self._correct = 0
        self._n = 0

    def record(self, logits: torch.Tensor, target: torch.Tensor, loss_value: float, n_valid: int) -> None:
        self._loss_sum += loss_value * n_valid
        valid_mask = target != IGNORE_INDEX
        if valid_mask.any():
            preds = logits[valid_mask].argmax(dim=1)
            self._correct += int((preds == target[valid_mask]).sum().item())
        self._n += n_valid

    def epoch_summary(self) -> dict:
        summary = {
            "n_valid": self._n,
            "mean_loss": (self._loss_sum / self._n) if self._n else None,
            "accuracy": (self._correct / self._n) if self._n else None,
        }
        self._loss_sum, self._correct, self._n = 0.0, 0, 0
        return summary


class AdversarialLossWrapper(nn.Module):
    """Wraps the dataset's normal M1 loss (unmodified) and adds the ITA
    discriminator's CE term on top. outputs["ita_adv"] is popped before
    being handed to the base loss and before run_epoch()'s generic
    per-task accuracy loop sees it -- that loop indexes `targets` by the
    same keys as `outputs`, and targets has no "ita_adv" entry (this
    mechanism's label comes from meta, not the dataset's normal targets),
    so leaving it in outputs would KeyError there."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss
        self.ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.tracker = _EpochAdvTracker()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_loss.train(mode)
        return self

    def forward(self, outputs: dict, targets: dict, meta: dict | None = None) -> dict:
        ita_logits = outputs.pop("ita_adv")
        losses = self.base_loss(outputs, targets, meta)

        ita_target = torch.tensor(
            [_ita_class(g) for g in meta["ita_group"]],
            dtype=torch.long,
            device=ita_logits.device,
        )
        adv_loss = self.ce(ita_logits, ita_target)
        n_valid = int((ita_target != IGNORE_INDEX).sum().item())
        if self.training:
            self.tracker.record(ita_logits.detach(), ita_target, float(adv_loss.item()), n_valid)

        losses["ita_adv"] = adv_loss
        losses["total"] = losses["total"] + adv_loss
        return losses


def _ramped_lambda(adv_lambda: float, epoch: int, max_epochs: int) -> float:
    """Standard DANN sigmoid ramp: ~0 at epoch 0, approaching adv_lambda by
    the final epoch. p = training progress in [0, 1); gamma=10 matches the
    original paper's constant."""
    p = epoch / max(max_epochs, 1)
    return float(adv_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def run_m4(
    dataset: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device_str: str,
    allow_dirty: bool,
    num_threads: int | None,
    num_workers: int,
    early_stopping_patience: int | None,
    adv_lambda: float = 1.0,
    grad_clip_norm: float | None = 5.0,
    max_steps_per_epoch: int | None = None,
) -> dict:
    out_dir = EXPERIMENTS_DIR / dataset / "M4_adversarial" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "adversarial_config.json").write_text(
        json.dumps(
            {
                "mechanism": "gradient_reversal",
                "adversarial_target": "ita_group",
                "ita_class_index": ITA_CLASS_INDEX,
                "grl_lambda_max": adv_lambda,
                "lambda_schedule": "DANN sigmoid ramp: adv_lambda * (2/(1+exp(-10p)) - 1), p = epoch/max_epochs",
                "grad_clip_norm": grad_clip_norm,
                "loss_formula": "total = base_task_loss + CrossEntropyLoss(ita_head(GRL(features)), ita_group)",
            },
            indent=2,
        )
    )

    history_path = out_dir / "adversarial_history.csv"
    history_rows: list[dict] = []
    loss_fn_holder: dict = {}
    model_holder: dict = {}

    def model_ready_callback(model: nn.Module, max_epochs_: int) -> None:
        model_holder["model"] = model
        model.grl.lambda_ = _ramped_lambda(adv_lambda, 0, max_epochs_)

    def epoch_callback(epoch: int, val_metrics: dict) -> None:
        stats = loss_fn_holder["loss_fn"].tracker.epoch_summary()
        history_rows.append({"epoch": epoch, "grl_lambda": model_holder["model"].grl.lambda_, **stats})
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        # Set for the epoch that's about to run next, so epoch 0's logged
        # lambda above reflects what epoch 0 actually trained with.
        model_holder["model"].grl.lambda_ = _ramped_lambda(adv_lambda, epoch + 1, max_epochs)

    import train as train_module

    original_build_model = train_module.build_model
    original_build_loss = train_module.build_loss

    def patched_build_model(dataset_: str, labels: dict) -> nn.Module:
        return build_adversarial_model(dataset_, labels, adv_lambda)

    def patched_build_loss(dataset_: str) -> nn.Module:
        loss_fn = AdversarialLossWrapper(build_base_loss(dataset_))
        loss_fn_holder["loss_fn"] = loss_fn
        return loss_fn

    train_module.build_model = patched_build_model
    train_module.build_loss = patched_build_loss
    try:
        result = train_dataset(
            dataset=dataset,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            device_str=device_str,
            allow_dirty=allow_dirty,
            num_threads=num_threads,
            num_workers=num_workers,
            early_stopping_patience=early_stopping_patience,
            max_steps_per_epoch=max_steps_per_epoch,
            out_dir=out_dir,
            epoch_callback=epoch_callback,
            model_ready_callback=model_ready_callback,
            grad_clip_norm=grad_clip_norm,
            method="M4",
        )
    finally:
        train_module.build_model = original_build_model
        train_module.build_loss = original_build_loss

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, required=True, help="Use M1's winning lr for a controlled comparison.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--adv-lambda", type=float, default=1.0, help="Target gradient-reversal strength; ramped up to this over training (DANN sigmoid schedule), not applied in full from epoch 0.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient-norm clip, a stability safety net for adversarial training; pass a negative value to disable.")
    parser.add_argument("--smoke-test-steps", type=int, default=None)
    args = parser.parse_args()

    run_m4(
        dataset=args.dataset,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_str=args.device,
        allow_dirty=args.allow_dirty,
        num_threads=args.num_threads,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        adv_lambda=args.adv_lambda,
        grad_clip_norm=(None if args.grad_clip_norm < 0 else args.grad_clip_norm),
        max_steps_per_epoch=args.smoke_test_steps,
    )


if __name__ == "__main__":
    main()
