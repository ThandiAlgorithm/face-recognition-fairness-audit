"""BAL-M1 training: shared ResNet-34, four heads (race/gender/age/
skin_tone), masked multi-task loss -- a sample only contributes a head's
loss term when that head's label is valid for it, and the per-sample
total is divided by how many heads were actually active for that sample
(not a fixed 4), exactly as specified:

    total_loss = sum(active head losses) / count(active heads)

BAL-M1 itself gets NO additional fairness intervention (no resampling, no
reweighting, no adversarial loss) -- deliberately, so it tests whether a
better-balanced dataset alone reduces disparity, before anything else is
layered on. Mitigated variants (BAL-M2/M3/M4) load this checkpoint and
fine-tune; they are separate scripts, not flags here.

Checkpoint selection: mean macro-F1 across whichever heads have enough
valid validation labels to score (never test). Early stopping and LR
schedule otherwise follow this project's frozen recipe (SGD+cosine).

--benchmark: same discipline as every other training script in this
project -- measure real images/sec before committing to a multi-hour run.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from dataset import BalM1Dataset, build_skin_tone_vocab, RACE_VOCAB, GENDER_VOCAB, AGE_VOCAB, IGNORE_INDEX, _load_rows
from model import build_bal_m1_model
from transforms import train_transform, eval_transform
from seed import set_seed, worker_init_fn, make_generator

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1" / "harmoface_manifest_enriched.csv"
OUT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "harmoface" / "BAL-M1"
TASKS = ["race", "gender", "age", "skin_tone"]


def collate_targets(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = {t: torch.stack([b[1][t] for b in batch]) for t in TASKS}
    rows = [b[2] for b in batch]
    return imgs, targets, rows


# Explicit, reproducible, and -- per explicit instruction -- fixed at 1.0
# for every head. BAL-M1 is the neutral balanced-data baseline: down-
# weighting any head here would be a task-priority intervention baked
# into training, confounding "did the balanced data help" with "we told
# the optimizer this attribute mattered less." Race/gender/age lead the
# written narrative; that is a reporting choice, not a training one.
HEAD_WEIGHTS = {"race": 1.0, "gender": 1.0, "age": 1.0, "skin_tone": 1.0}


def masked_multitask_loss(outputs: dict, targets: dict, criteria: dict[str, nn.Module]) -> tuple[torch.Tensor, dict[str, float]]:
    """Returns (batch total loss, {task: mean loss over that task's valid
    samples in this batch}) -- the per-task breakdown is for logging only,
    never fed back into the weighting itself."""
    per_sample_terms = []
    valid_masks = []
    per_task_mean = {}
    for task in TASKS:
        per_sample = criteria[task](outputs[task], targets[task])  # (B,), 0 where ignore_index
        valid = (targets[task] != IGNORE_INDEX).float()
        weighted = per_sample * HEAD_WEIGHTS[task]
        per_sample_terms.append(weighted)
        valid_masks.append(valid)
        n_valid = valid.sum().item()
        per_task_mean[task] = float((per_sample * valid).sum().item() / n_valid) if n_valid > 0 else None
    total = sum(per_sample_terms)
    active_heads = sum(valid_masks).clamp(min=1.0)
    return (total / active_heads).mean(), per_task_mean


@torch.no_grad()
def evaluate_metrics(model, loader, device, criteria: dict[str, nn.Module]) -> dict:
    """Returns per-task macro-F1, balanced accuracy, and mean validation
    loss, plus the mean macro-F1 across tasks with enough valid labels
    (>=2 classes present) to score meaningfully -- the checkpoint-
    selection criterion."""
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


def run_benchmark(train_ds, batch_size, num_threads, num_workers, n_race, n_gender, n_age, n_skin, device, n_batches=20):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    model = build_bal_m1_model(n_race, n_gender, n_age, n_skin).to(device)
    optimizer, _ = build_optimizer_and_scheduler(model, 0.01, 1e-4, 1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_targets)
    criteria = {t: nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none") for t in TASKS}
    model.train()
    times = []
    it = iter(loader)
    for i in range(n_batches):
        try:
            imgs, targets, _rows = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, targets, _rows = next(it)
        imgs = imgs.to(device)
        targets = {t: v.to(device) for t, v in targets.items()}
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(imgs)
        loss, _per_task = masked_multitask_loss(out, targets, criteria)
        loss.backward()
        optimizer.step()
        times.append(time.perf_counter() - t0)
    mean_s = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))
    batches_per_epoch = (len(train_ds) + batch_size - 1) // batch_size
    epoch_s = mean_s * batches_per_epoch
    result = {
        "n_train_images": len(train_ds), "batch_size": batch_size, "batches_per_epoch": batches_per_epoch,
        "mean_s_per_batch": mean_s, "estimated_epoch_s": epoch_s,
        "estimated_epoch_hm": f"{int(epoch_s//3600)}h{int((epoch_s%3600)//60)}m",
    }
    print(json.dumps(result, indent=2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def train(lr, batch_size, max_epochs, early_stopping_patience, num_threads, num_workers, seed, device_str,
          resume=False, resume_checkpoint=None, stop_after_epoch=None):
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
    print(f"[BAL-M1] train n={len(train_ds)} (file_present only), val n={len(val_ds)}")

    generator = make_generator(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=generator,
        worker_init_fn=worker_init_fn, num_workers=num_workers, collate_fn=collate_targets,
        persistent_workers=(num_workers > 0), **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_targets)

    n_race, n_gender, n_age, n_skin = len(RACE_VOCAB), len(GENDER_VOCAB), len(AGE_VOCAB), len(skin_vocab)
    model = build_bal_m1_model(n_race, n_gender, n_age, n_skin).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, 1e-4, max_epochs)
    criteria = {t: nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none") for t in TASKS}

    config = {
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience, "seed": seed,
        "optimizer": "sgd", "momentum": 0.9, "weight_decay": 1e-4, "scheduler": "cosine_annealing",
        "mitigation": "none", "model": "BAL-M1",
        "n_classes": {"race": n_race, "gender": n_gender, "age": n_age, "skin_tone": n_skin},
        # Explicit and fixed at 1.0 for every head -- BAL-M1 is the neutral
        # balanced-data baseline, never down-weighted. See HEAD_WEIGHTS'
        # docstring above for why this is saved explicitly, not implied.
        "head_weights": HEAD_WEIGHTS,
    }
    (OUT_DIR / "config.yaml").write_text("\n".join(f"{k}: {v}" for k, v in config.items()))

    history = []
    best_score = -float("inf")
    epochs_without_improvement = 0
    start_epoch = 0

    if resume:
        ckpt_path = resume_checkpoint or (OUT_DIR / "best.pt")
        print(f"[BAL-M1] resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt["mean_macro_f1"]
        # epochs_without_improvement resets to 0 on resume: the saved checkpoint
        # has no record of how many non-improving epochs preceded the interruption
        # (history.csv for this run was not retained), so this is a fresh patience
        # window from the resume point, not a continuation of an unknown count.
        epochs_without_improvement = 0
        print(f"[BAL-M1] resumed at epoch {start_epoch} (best_score={best_score:.4f} from epoch {ckpt['epoch']})")

    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        model.train()
        total_loss, n_seen = 0.0, 0
        train_task_loss_sum = {t: 0.0 for t in TASKS}
        train_task_loss_n = {t: 0 for t in TASKS}
        for imgs, targets, _rows in train_loader:
            imgs = imgs.to(device)
            targets = {t: v.to(device) for t, v in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss, per_task = masked_multitask_loss(out, targets, criteria)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
            for t in TASKS:
                if per_task[t] is not None:
                    n_valid_t = int((targets[t] != IGNORE_INDEX).sum().item())
                    train_task_loss_sum[t] += per_task[t] * n_valid_t
                    train_task_loss_n[t] += n_valid_t
        scheduler.step()
        train_loss = total_loss / n_seen
        train_task_loss = {t: (train_task_loss_sum[t] / train_task_loss_n[t]) if train_task_loss_n[t] > 0 else None
                            for t in TASKS}

        val_metrics = evaluate_metrics(model, val_loader, device, criteria)
        elapsed = time.time() - t0
        row = {
            "epoch": epoch, "train_loss": train_loss, "mean_macro_f1": val_metrics["mean_macro_f1"],
            "elapsed_s": elapsed,
            **{f"train_{t}_loss": train_task_loss[t] for t in TASKS},
            **{f"val_{t}_loss": val_metrics["per_task_loss"][t] for t in TASKS},
            **{f"val_f1_{t}": val_metrics["per_task_macro_f1"][t] for t in TASKS},
            **{f"val_balanced_accuracy_{t}": val_metrics["per_task_balanced_accuracy"][t] for t in TASKS},
        }
        history.append(row)
        f1_str = " ".join(f"{t}={val_metrics['per_task_macro_f1'][t]}" for t in TASKS)
        print(f"[BAL-M1] epoch {epoch}: train_loss={train_loss:.4f} mean_macro_f1={val_metrics['mean_macro_f1']:.4f} "
              f"per-task F1: {f1_str} ({elapsed:.1f}s)")

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
            print(f"[BAL-M1] early stopping at epoch {epoch} (no mean_macro_f1 improvement for {early_stopping_patience} epochs)")
            break

        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            print(f"[BAL-M1] stopping probe at epoch {epoch} (--stop-after-epoch={stop_after_epoch}); "
                  f"this is NOT early stopping, just a capped probe run")
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
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="resume from --resume-checkpoint (default OUT_DIR/best.pt): reloads "
                             "model/optimizer/scheduler state and continues from checkpoint['epoch']+1. "
                             "--max-epochs must match the ORIGINAL run's value so the cosine LR "
                             "schedule's shape is unchanged; use --stop-after-epoch to cap a short probe.")
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="break the loop after this absolute epoch index completes, regardless of "
                             "max-epochs/early-stopping-patience -- for a cheap partial probe.")
    args = parser.parse_args()

    if args.benchmark:
        set_seed(args.seed)
        train_rows = _load_rows(MANIFEST_PATH, "train")
        skin_vocab = build_skin_tone_vocab(train_rows)
        train_ds = BalM1Dataset(MANIFEST_PATH, "train", train_transform(), skin_tone_vocab=skin_vocab)
        run_benchmark(train_ds, args.batch_size, args.num_threads, args.num_workers,
                      len(RACE_VOCAB), len(GENDER_VOCAB), len(AGE_VOCAB), len(skin_vocab),
                      torch.device(args.device), args.benchmark_batches)
        return

    resume_ckpt = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    train(lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
          early_stopping_patience=args.early_stopping_patience, num_threads=args.num_threads,
          num_workers=args.num_workers, seed=args.seed, device_str=args.device,
          resume=args.resume, resume_checkpoint=resume_ckpt, stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
