"""BAL-M2: trains BAL-M1's adversarial-debiasing variant (model_adversarial.py).
Race and skin_tone keep the exact masked multi-task supervision BAL-M1
used; gender and age additionally get a GRL-reversed adversary (sigmoid-
ramped lambda, same DANN schedule as M0's and FairFace M4's adversarial
variants) trying to recover race/skin_tone from the gender/age adapter,
pushing that adapter to drop race/skin-tone-correlated shortcuts.

Gradient clipping at --grad-clip-norm is the same fix that stabilized
FairFace M4's adversarial training (see mitigation_m4.py) -- applied here
from the start rather than rediscovered after an unstable first run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from dataset import BalM1Dataset, build_skin_tone_vocab, RACE_VOCAB, GENDER_VOCAB, AGE_VOCAB, IGNORE_INDEX, _load_rows  # noqa: E402
from model_adversarial import build_bal_m1_adversarial_model  # noqa: E402
from transforms import train_transform, eval_transform  # noqa: E402
from seed import set_seed, worker_init_fn, make_generator  # noqa: E402

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1" / "harmoface_manifest_enriched.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M2-adversarial"
TASKS = ["race", "gender", "age", "skin_tone"]
ADV_PAIRS = {"adv_race": "race", "adv_skin_tone": "skin_tone"}

HEAD_WEIGHTS = {"race": 1.0, "gender": 1.0, "age": 1.0, "skin_tone": 1.0}


def collate_targets(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = {t: torch.stack([b[1][t] for b in batch]) for t in TASKS}
    rows = [b[2] for b in batch]
    return imgs, targets, rows


def _ramped_lambda(adv_lambda: float, epoch: int, max_epochs: int) -> float:
    """Identical formula to src/frs_m0/train.py's _ramped_lambda (the
    already-debugged DANN sigmoid ramp): ~0 at epoch 0, approaching
    adv_lambda by the final epoch."""
    p = epoch / max(max_epochs, 1)
    return float(adv_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def masked_multitask_loss(outputs: dict, targets: dict, criteria: dict[str, nn.Module]) -> tuple[torch.Tensor, dict[str, float]]:
    per_sample_terms = []
    valid_masks = []
    per_task_mean = {}
    for task in TASKS:
        per_sample = criteria[task](outputs[task], targets[task])
        valid = (targets[task] != IGNORE_INDEX).float()
        weighted = per_sample * HEAD_WEIGHTS[task]
        per_sample_terms.append(weighted)
        valid_masks.append(valid)
        n_valid = valid.sum().item()
        per_task_mean[task] = float((per_sample * valid).sum().item() / n_valid) if n_valid > 0 else None
    total = sum(per_sample_terms)
    active_heads = sum(valid_masks).clamp(min=1.0)
    main_loss = (total / active_heads).mean()

    adv_loss_terms = []
    for adv_key, true_task in ADV_PAIRS.items():
        per_sample = criteria[true_task](outputs[adv_key], targets[true_task])
        valid = (targets[true_task] != IGNORE_INDEX).float()
        n_valid = valid.sum().item()
        per_task_mean[adv_key] = float((per_sample * valid).sum().item() / n_valid) if n_valid > 0 else None
        if n_valid > 0:
            adv_loss_terms.append((per_sample * valid).sum() / valid.sum().clamp(min=1.0))
        else:
            adv_loss_terms.append(outputs[adv_key].sum() * 0.0)
    adv_loss = sum(adv_loss_terms) / len(adv_loss_terms)

    return main_loss + adv_loss, per_task_mean


@torch.no_grad()
def evaluate_metrics(model, loader, device, criteria: dict[str, nn.Module]) -> dict:
    from sklearn.metrics import f1_score, balanced_accuracy_score

    model.eval()
    preds = {t: [] for t in TASKS}
    trues = {t: [] for t in TASKS}
    loss_sum = {t: 0.0 for t in TASKS}
    loss_n = {t: 0 for t in TASKS}
    for imgs, targets, _rows in loader:
        imgs = imgs.to(device)
        targets_dev = {t: v.to(device) for t, v in targets.items()}
        out = model(imgs)
        for t in TASKS:
            per_sample = criteria[t](out[t], targets_dev[t]).cpu().numpy()
            p = out[t].argmax(dim=1).cpu().numpy()
            y = targets[t].numpy()
            mask = y != IGNORE_INDEX
            preds[t].extend(p[mask].tolist())
            trues[t].extend(y[mask].tolist())
            loss_sum[t] += float(per_sample[mask].sum())
            loss_n[t] += int(mask.sum())

    per_task_f1, per_task_bal_acc, per_task_loss = {}, {}, {}
    for t in TASKS:
        if len(set(trues[t])) >= 2:
            per_task_f1[t] = float(f1_score(trues[t], preds[t], average="macro"))
            per_task_bal_acc[t] = float(balanced_accuracy_score(trues[t], preds[t]))
        else:
            per_task_f1[t] = None
            per_task_bal_acc[t] = None
        per_task_loss[t] = (loss_sum[t] / loss_n[t]) if loss_n[t] > 0 else None
    valid_scores = [v for v in per_task_f1.values() if v is not None]
    mean_f1 = float(np.mean(valid_scores)) if valid_scores else 0.0
    return {"per_task_macro_f1": per_task_f1, "per_task_balanced_accuracy": per_task_bal_acc,
            "per_task_loss": per_task_loss, "mean_macro_f1": mean_f1}


def build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return optimizer, scheduler


def train(lr, batch_size, max_epochs, early_stopping_patience, num_threads, num_workers, seed, device_str,
          adv_lambda, grad_clip_norm):
    set_seed(seed)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    device = torch.device(device_str)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_rows = _load_rows(MANIFEST_PATH, "train")
    skin_vocab = build_skin_tone_vocab(train_rows)
    (OUT_DIR / "label_mappings.json").write_text(json.dumps({
        "race": RACE_VOCAB, "gender": GENDER_VOCAB, "age": AGE_VOCAB, "skin_tone": skin_vocab,
    }, indent=2))

    train_ds = BalM1Dataset(MANIFEST_PATH, "train", train_transform(), skin_tone_vocab=skin_vocab)
    val_ds = BalM1Dataset(MANIFEST_PATH, "val", eval_transform(), skin_tone_vocab=skin_vocab)
    print(f"[BAL-M2-adv] train n={len(train_ds)} (file_present only), val n={len(val_ds)}")

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator,
        worker_init_fn=worker_init_fn, num_workers=num_workers, collate_fn=collate_targets,
        persistent_workers=(num_workers > 0), **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_targets)

    n_race, n_gender, n_age, n_skin = len(RACE_VOCAB), len(GENDER_VOCAB), len(AGE_VOCAB), len(skin_vocab)
    model = build_bal_m1_adversarial_model(n_race, n_gender, n_age, n_skin, adv_lambda=0.0).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, 1e-4, max_epochs)
    criteria = {t: nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none") for t in TASKS}

    config = {
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience, "seed": seed,
        "optimizer": "sgd", "momentum": 0.9, "weight_decay": 1e-4, "scheduler": "cosine_annealing",
        "mitigation": "adversarial_grl", "model": "BAL-M2-adversarial",
        "adv_lambda": adv_lambda, "grad_clip_norm": grad_clip_norm,
        "n_classes": {"race": n_race, "gender": n_gender, "age": n_age, "skin_tone": n_skin},
        "head_weights": HEAD_WEIGHTS,
        "adv_design": "race/skin_tone supervised directly off the shared backbone (unchanged from BAL-M1); "
                      "gender/age route through a 256-d adapter; a GRL-reversed adversary on that adapter "
                      "tries to recover race/skin_tone, pushing gender/age to drop race/skin-tone shortcuts.",
    }
    (OUT_DIR / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in config.items()))

    history = []
    best_score = -float("inf")
    epochs_without_improvement = 0
    all_loss_keys = TASKS + list(ADV_PAIRS.keys())

    for epoch in range(max_epochs):
        t0 = time.time()
        lambda_now = _ramped_lambda(adv_lambda, epoch, max_epochs)
        model.set_lambda(lambda_now)
        model.train()
        total_loss, n_seen = 0.0, 0
        train_task_loss_sum = {t: 0.0 for t in all_loss_keys}
        train_task_loss_n = {t: 0 for t in all_loss_keys}
        for imgs, targets, _rows in train_loader:
            imgs = imgs.to(device)
            targets = {t: v.to(device) for t, v in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss, per_task = masked_multitask_loss(out, targets, criteria)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            for t in all_loss_keys:
                if per_task.get(t) is not None:
                    true_task = ADV_PAIRS.get(t, t)
                    n_valid_t = int((targets[true_task] != IGNORE_INDEX).sum().item())
                    train_task_loss_sum[t] += per_task[t] * n_valid_t
                    train_task_loss_n[t] += n_valid_t
        scheduler.step()
        train_loss = total_loss / n_seen
        train_task_loss = {t: (train_task_loss_sum[t] / train_task_loss_n[t]) if train_task_loss_n[t] > 0 else None
                            for t in all_loss_keys}

        val_metrics = evaluate_metrics(model, val_loader, device, criteria)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch, "train_loss": train_loss, "mean_macro_f1": val_metrics["mean_macro_f1"],
            "adv_lambda": lambda_now, "elapsed_s": elapsed,
            **{f"train_{t}_loss": train_task_loss[t] for t in all_loss_keys},
            **{f"val_{t}_loss": val_metrics["per_task_loss"][t] for t in TASKS},
            **{f"val_f1_{t}": val_metrics["per_task_macro_f1"][t] for t in TASKS},
            **{f"val_balanced_accuracy_{t}": val_metrics["per_task_balanced_accuracy"][t] for t in TASKS},
        }
        history.append(row)
        f1_str = " ".join(f"{t}={val_metrics['per_task_macro_f1'][t]}" for t in TASKS)
        print(f"[BAL-M2-adv] epoch {epoch}: train_loss={train_loss:.4f} mean_macro_f1={val_metrics['mean_macro_f1']:.4f} "
              f"adv_lambda={lambda_now:.3f} per-task F1: {f1_str} ({elapsed:.1f}s)")

        improved = val_metrics["mean_macro_f1"] > best_score
        if improved:
            best_score = val_metrics["mean_macro_f1"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "epoch": epoch, "mean_macro_f1": best_score,
                "config": config, "label_mappings": {"race": RACE_VOCAB, "gender": GENDER_VOCAB, "age": AGE_VOCAB, "skin_tone": skin_vocab},
            }, OUT_DIR / "best.pt")
        else:
            epochs_without_improvement += 1

        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "mean_macro_f1": val_metrics["mean_macro_f1"], "config": config},
                    OUT_DIR / "last.pt")

        with open(OUT_DIR / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"[BAL-M2-adv] early stopping at epoch {epoch} (no mean_macro_f1 improvement for {early_stopping_patience} epochs)")
            break

    return {"best_mean_macro_f1": best_score, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adv-lambda", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    args = parser.parse_args()

    train(lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
          early_stopping_patience=args.early_stopping_patience, num_threads=args.num_threads,
          num_workers=args.num_workers, seed=args.seed, device_str=args.device,
          adv_lambda=args.adv_lambda, grad_clip_norm=args.grad_clip_norm)


if __name__ == "__main__":
    main()
