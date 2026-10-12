"""M0 training: identity classification (plain CrossEntropy -- no ArcFace,
no fairness mitigation of any kind, see module docstring in model.py and
spec section 39). Checkpoint selection uses validation verification EER,
not training-identity-head accuracy, because validation identities are
never seen as classes during training (spec section 11) -- ordinary
classification accuracy over the train-identity head is meaningless for
validation images, which almost never belong to a known class.

--benchmark mirrors src/training/train.py's convention: measures real
images/sec for a handful of batches before committing to a multi-hour run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from dataset import load_m0_table, build_train_identity_vocab, M0TrainDataset, M0EvalDataset, IGNORE_INDEX
from model import M0EmbeddingModel, M0ArcFaceEmbeddingModel, M0AdversarialEmbeddingModel
from utils import m0_train_transform, m0_eval_transform, cosine_similarity_matrix_pairs, compute_roc, eer_from_roc
from make_pairs import generate_pairs
from seed import set_seed, worker_init_fn, make_generator, rng_state_dict

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"
SPLIT_HASH_PATH = Path(__file__).resolve().parents[2] / "experiments" / "lfw" / "split_manifest.sha256"
MITIGATIONS = ["none", "resampling", "reweighting", "adversarial"]


def _ramped_lambda(adv_lambda: float, epoch: int, max_epochs: int) -> float:
    """Identical formula to src/training/mitigation_m4.py's _ramped_lambda
    (the already-debugged fix for FairFace M4's adversarial training
    instability) -- standard DANN sigmoid ramp: ~0 at epoch 0, approaching
    adv_lambda by the final epoch. Reused here deliberately rather than
    rediscovered."""
    p = epoch / max(max_epochs, 1)
    return float(adv_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


class _WeightedSampler(Sampler):
    """M2-MULTI-equivalent for M0: draws indices with probability
    proportional to M0TrainDataset.weights (the combined race+gender+age+
    Fitzpatrick rarity weight), with replacement, num_samples per epoch ==
    len(dataset) so epoch size is unchanged. Deliberately a smaller copy of
    src/training/mitigation_m2.py's CountingWeightedSampler rather than an
    import of it -- that module does `from train import ...`, which would
    collide with this file's own name (both are called train.py)."""

    def __init__(self, weights: np.ndarray, generator: torch.Generator):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.generator = generator

    def __iter__(self):
        idx = torch.multinomial(self.weights, len(self.weights), replacement=True, generator=self.generator)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return len(self.weights)


def _build_model(loss: str, mitigation: str, n_classes: int, n_race: int, n_gender: int, n_age: int,
                  arcface_scale: float, arcface_margin: float, adv_lambda: float):
    if mitigation == "adversarial":
        # Always built on the ArcFace head internally -- adversarial
        # debiasing is a response to v2's deep dive (race/gender fairness
        # unresolved by the loss switch alone), so it only makes sense
        # layered on top of the better base recipe, not the old v1 softmax.
        return M0AdversarialEmbeddingModel(n_classes, n_race, n_gender, n_age,
                                            scale=arcface_scale, margin=arcface_margin, adv_lambda=adv_lambda)
    if loss == "arcface":
        return M0ArcFaceEmbeddingModel(n_classes, scale=arcface_scale, margin=arcface_margin)
    return M0EmbeddingModel(n_classes)


def _safe_adv_loss(criterion: nn.Module, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """nn.CrossEntropyLoss(ignore_index=...) with reduction='mean' returns
    NaN (0/0) when EVERY target in a batch is ignore_index -- guaranteed to
    happen every single batch for an attribute a dataset never has at all
    (LFW's age_group, 100% missing -- see dataset.py). A NaN loss term
    poisons the whole sum and silently breaks training from batch 1.
    Returns a real zero (no gradient, nothing to learn from an all-missing
    batch) instead."""
    if bool((targets != IGNORE_INDEX).any()):
        return criterion(logits, targets)
    return logits.sum() * 0.0


def _forward_step(model, loss: str, mitigation: str, imgs: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """ArcFace (and adversarial, which always has an ArcFace head
    internally) needs the true label at forward time (the margin only
    applies to the target class's logit); plain softmax doesn't."""
    if loss == "arcface" or mitigation == "adversarial":
        return model(imgs, labels)
    return model(imgs)


def build_optimizer_and_scheduler(model, lr, weight_decay, max_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    return optimizer, scheduler


@torch.no_grad()
def extract_val_embeddings(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    out = {}
    for imgs, image_ids in loader:
        imgs = imgs.to(device)
        emb = model.embed(imgs).cpu().numpy()
        for iid, e in zip(image_ids, emb):
            out[iid] = e
    return out


def compute_val_eer(model, val_loader, val_pairs, device) -> float:
    embeddings = extract_val_embeddings(model, val_loader, device)
    a = np.stack([embeddings[iid] for iid in val_pairs["image_id_a"]])
    b = np.stack([embeddings[iid] for iid in val_pairs["image_id_b"]])
    scores = cosine_similarity_matrix_pairs(a, b)
    labels = val_pairs["same_identity"].to_numpy()
    thresholds, fmr, fnmr = compute_roc(scores, labels)
    eer, _ = eer_from_roc(thresholds, fmr, fnmr)
    return eer


def run_benchmark(train_ds, batch_size, num_threads, num_workers, n_classes, device, n_batches=20,
                   loss="softmax", arcface_scale=64.0, arcface_margin=0.5, out_dir: Path = OUT_DIR,
                   mitigation="none", adv_lambda=1.0):
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    n_race = max(len(train_ds.race_vocab), 1)
    n_gender = max(len(train_ds.gender_vocab), 1)
    n_age = max(len(train_ds.age_vocab), 1)
    model = _build_model(loss, mitigation, n_classes, n_race, n_gender, n_age,
                          arcface_scale, arcface_margin, adv_lambda).to(device)
    optimizer, _ = build_optimizer_and_scheduler(model, 0.01, 1e-4, 1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    criterion = nn.CrossEntropyLoss()
    adv_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    model.train()
    times = []
    it = iter(loader)
    for i in range(n_batches):
        try:
            imgs, labels, _weights, race_l, gender_l, age_l = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, labels, _weights, race_l, gender_l, age_l = next(it)
        imgs, labels = imgs.to(device), labels.to(device)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = _forward_step(model, loss, mitigation, imgs, labels)
        batch_loss = criterion(out["identity_logits"], labels)
        if mitigation == "adversarial":
            batch_loss = batch_loss + _safe_adv_loss(adv_criterion, out["race_logits"], race_l.to(device))
            batch_loss = batch_loss + _safe_adv_loss(adv_criterion, out["gender_logits"], gender_l.to(device))
            batch_loss = batch_loss + _safe_adv_loss(adv_criterion, out["age_logits"], age_l.to(device))
        batch_loss.backward()
        optimizer.step()
        times.append(time.perf_counter() - t0)
    mean_s = float(np.mean(times[5:])) if len(times) > 5 else float(np.mean(times))
    batches_per_epoch = (len(train_ds) + batch_size - 1) // batch_size
    epoch_s = mean_s * batches_per_epoch
    result = {
        "loss": loss, "mitigation": mitigation, "n_train_images": len(train_ds), "n_classes": n_classes,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch, "mean_s_per_batch": mean_s,
        "estimated_epoch_s": epoch_s, "estimated_epoch_hm": f"{int(epoch_s//3600)}h{int((epoch_s%3600)//60)}m",
    }
    print(json.dumps(result, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    return result


def train(
    lr: float, batch_size: int, max_epochs: int, early_stopping_patience: int,
    num_threads: int | None, num_workers: int, seed: int, min_train_images: int,
    device_str: str = "cpu", loss: str = "softmax", arcface_scale: float = 64.0,
    arcface_margin: float = 0.5, out_dir: Path = OUT_DIR, mitigation: str = "none",
    adv_lambda: float = 1.0, grad_clip_norm: float | None = 5.0,
):
    set_seed(seed)
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    device = torch.device(device_str)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_m0_table()
    vocab, removed = build_train_identity_vocab(df, min_train_images)
    (out_dir / "removed_train_identities.json").write_text(json.dumps({
        "min_train_images_per_identity": min_train_images,
        "n_removed": len(removed), "removed_identity_ids": removed,
    }, indent=2))
    (out_dir / "identity_vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"[M0] train identity vocab: {len(vocab)} identities kept, {len(removed)} removed (< {min_train_images} images)")

    # mitigation in {resampling, reweighting} both need the same combined
    # race+gender+age+Fitzpatrick rarity weight (demographic_weights.py,
    # the M2-MULTI/M3-MULTI formula) -- they differ only in where it's
    # injected below: the sampler for resampling, the loss for reweighting.
    # adversarial doesn't use this weight at all (it uses the per-image
    # race/gender/age LABELS instead, always computed by M0TrainDataset
    # regardless of compute_weights).
    compute_weights = mitigation in ("resampling", "reweighting")
    train_ds = M0TrainDataset(df, vocab, m0_train_transform(), compute_weights=compute_weights)
    if train_ds.weight_diagnostics is not None:
        (out_dir / "weight_diagnostics.json").write_text(json.dumps(train_ds.weight_diagnostics, indent=2))
    print(f"[M0] demo vocabs: race={len(train_ds.race_vocab)} gender={len(train_ds.gender_vocab)} "
          f"age={len(train_ds.age_vocab)} (0 means that attribute has no real values in this dataset)")
    generator = make_generator(seed)
    if mitigation == "resampling":
        sampler = _WeightedSampler(train_ds.weights, generator)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            worker_init_fn=worker_init_fn, num_workers=num_workers,
            persistent_workers=(num_workers > 0), **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, generator=generator,
            worker_init_fn=worker_init_fn, num_workers=num_workers,
            persistent_workers=(num_workers > 0), **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )

    # Val pairs are generated once, here, and reused for every validation
    # check across all epochs -- never test pairs, never regenerated.
    val_pairs_path = out_dir / "pairs" / "val_pairs.csv"
    val_pairs_path.parent.mkdir(exist_ok=True)
    if val_pairs_path.exists():
        val_pairs = pd.read_csv(val_pairs_path)
    else:
        val_pairs = generate_pairs(df, "val", n_impostor=20000, pair_seed=42)
        val_pairs.to_csv(val_pairs_path, index=False)
    print(f"[M0] val pairs: {(val_pairs['same_identity']==1).sum()} genuine, {(val_pairs['same_identity']==0).sum()} impostor")

    val_eval_ds = M0EvalDataset(df, "val", m0_eval_transform())
    val_eval_loader = DataLoader(val_eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n_race = max(len(train_ds.race_vocab), 1)
    n_gender = max(len(train_ds.gender_vocab), 1)
    n_age = max(len(train_ds.age_vocab), 1)
    model = _build_model(loss, mitigation, len(vocab), n_race, n_gender, n_age,
                          arcface_scale, arcface_margin, adv_lambda).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, lr, 1e-4, max_epochs)
    # reduction="none": mitigation=="reweighting" needs the per-sample loss
    # to multiply by each image's weight before averaging; "none"/
    # "resampling"/"adversarial" just take a plain .mean() of the same
    # per-sample values, which is mathematically identical to the old
    # reduction="mean" default.
    criterion = nn.CrossEntropyLoss(reduction="none")
    adv_criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    if mitigation == "adversarial":
        (out_dir / "adversarial_config.json").write_text(json.dumps({
            "mechanism": "gradient_reversal", "adversarial_targets": ["race", "gender", "age"],
            "race_vocab": train_ds.race_vocab, "gender_vocab": train_ds.gender_vocab, "age_vocab": train_ds.age_vocab,
            "grl_lambda_max": adv_lambda,
            "lambda_schedule": "DANN sigmoid ramp: adv_lambda * (2/(1+exp(-10p)) - 1), p = epoch/max_epochs",
            "grad_clip_norm": grad_clip_norm,
            "loss_formula": "total = identity_loss + CE(race_head(GRL(emb))) + CE(gender_head(GRL(emb))) + CE(age_head(GRL(emb)))",
        }, indent=2))

    split_hash = SPLIT_HASH_PATH.read_text().strip() if SPLIT_HASH_PATH.exists() else None
    config = {
        "lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience, "seed": seed,
        "min_train_images_per_identity": min_train_images, "n_train_identities": len(vocab),
        "optimizer": "sgd", "momentum": 0.9, "weight_decay": 1e-4, "scheduler": "cosine_annealing",
        "criterion": "arcface" if (loss == "arcface" or mitigation == "adversarial") else "cross_entropy",
        "mitigation": mitigation,
        **({"arcface_scale": arcface_scale, "arcface_margin": arcface_margin}
           if (loss == "arcface" or mitigation == "adversarial") else {}),
        **({"adv_lambda": adv_lambda, "grad_clip_norm": grad_clip_norm} if mitigation == "adversarial" else {}),
    }

    history = []
    best_eer = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        if mitigation == "adversarial":
            # Set for the epoch about to run -- epoch 0 trains with
            # lambda~0 (identity task only, let the embedding stabilize
            # first), ramping to adv_lambda by the final epoch. Same fix
            # already required for FairFace's M4 adversarial training.
            model.grl.lambda_ = _ramped_lambda(adv_lambda, epoch, max_epochs)
        t0 = time.time()
        model.train()
        total_loss, n_seen = 0.0, 0
        adv_loss_sum = {"race": 0.0, "gender": 0.0, "age": 0.0}
        for imgs, labels, weights, race_l, gender_l, age_l in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = _forward_step(model, loss, mitigation, imgs, labels)
            per_sample_loss = criterion(out["identity_logits"], labels)
            if mitigation == "reweighting":
                batch_loss = (per_sample_loss * weights.to(device)).mean()
            else:
                batch_loss = per_sample_loss.mean()
            if mitigation == "adversarial":
                race_loss = _safe_adv_loss(adv_criterion, out["race_logits"], race_l.to(device))
                gender_loss = _safe_adv_loss(adv_criterion, out["gender_logits"], gender_l.to(device))
                age_loss = _safe_adv_loss(adv_criterion, out["age_logits"], age_l.to(device))
                batch_loss = batch_loss + race_loss + gender_loss + age_loss
                adv_loss_sum["race"] += float(race_loss.item()) * imgs.size(0)
                adv_loss_sum["gender"] += float(gender_loss.item()) * imgs.size(0)
                adv_loss_sum["age"] += float(age_loss.item()) * imgs.size(0)
            batch_loss.backward()
            if grad_clip_norm is not None and mitigation == "adversarial":
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            total_loss += batch_loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        scheduler.step()
        train_loss = total_loss / n_seen

        val_eer = compute_val_eer(model, val_eval_loader, val_pairs, device)
        elapsed = time.time() - t0
        row = {"epoch": epoch, "train_loss": train_loss, "val_eer": val_eer, "elapsed_s": elapsed}
        log_suffix = ""
        if mitigation == "adversarial":
            row["grl_lambda"] = model.grl.lambda_
            row["adv_race_loss"] = adv_loss_sum["race"] / n_seen
            row["adv_gender_loss"] = adv_loss_sum["gender"] / n_seen
            row["adv_age_loss"] = adv_loss_sum["age"] / n_seen
            log_suffix = (f" grl_lambda={row['grl_lambda']:.3f} adv_race={row['adv_race_loss']:.3f} "
                          f"adv_gender={row['adv_gender_loss']:.3f} adv_age={row['adv_age_loss']:.3f}")
        history.append(row)
        print(f"[M0/lfw] epoch {epoch}: train_loss={train_loss:.4f} val_eer={val_eer:.4f} ({elapsed:.1f}s){log_suffix}")

        improved = val_eer < best_eer
        if improved:
            best_eer = val_eer
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch, "val_eer": best_eer, "config": config,
                "identity_vocab": vocab, "split_manifest_sha256": split_hash,
                "seed": seed, "rng_states": rng_state_dict(),
            }, out_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        torch.save({
            "model_state_dict": model.state_dict(), "epoch": epoch, "val_eer": val_eer, "config": config,
        }, out_dir / "last.pt")

        import csv
        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"[M0/lfw] early stopping at epoch {epoch} (no val_eer improvement for {early_stopping_patience} epochs)")
            break

    (out_dir / "config.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in config.items())
    )
    return {"best_val_eer": best_eer, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train-images", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=20)
    parser.add_argument("--loss", choices=["softmax", "arcface"], default="softmax",
                         help="softmax = M0 v1 (plain CrossEntropy identity head, unchanged). "
                              "arcface = M0 v2 (ArcMarginProduct head, see model.py).")
    parser.add_argument("--arcface-scale", type=float, default=64.0)
    parser.add_argument("--arcface-margin", type=float, default=0.5)
    parser.add_argument("--mitigation", choices=MITIGATIONS, default="none",
                         help="none = M1-equivalent (no fairness intervention). resampling = M2-MULTI-"
                              "equivalent (WeightedRandomSampler on the combined race+gender+age+"
                              "Fitzpatrick rarity weight). reweighting = M3-MULTI-equivalent (same "
                              "weight, applied to the per-sample loss instead).")
    parser.add_argument("--adv-lambda", type=float, default=1.0,
                         help="Target gradient-reversal strength for --mitigation adversarial; ramped "
                              "up to this over training (DANN sigmoid schedule), not applied in full "
                              "from epoch 0. Ignored for other mitigations.")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0,
                         help="Gradient-norm clip, a stability safety net for adversarial training "
                              "(same default as FairFace/UTKFace/LFW's M4). Only applied when "
                              "--mitigation adversarial. Pass a negative value to disable.")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Defaults to outputs/M0_LFW (v1's location) -- pass a loss/mitigation-"
                              "specific dir (e.g. outputs/M0_LFW_arcface_resampling) so each variant "
                              "gets its own directory and never overwrites another's results.")
    args = parser.parse_args()
    out_dir = args.out_dir or OUT_DIR
    grad_clip_norm = None if args.grad_clip_norm < 0 else args.grad_clip_norm

    if args.benchmark:
        set_seed(args.seed)
        df = load_m0_table()
        vocab, _ = build_train_identity_vocab(df, args.min_train_images)
        train_ds = M0TrainDataset(df, vocab, m0_train_transform(),
                                   compute_weights=(args.mitigation in ("resampling", "reweighting")))
        run_benchmark(train_ds, args.batch_size, args.num_threads, args.num_workers, len(vocab), torch.device(args.device),
                       args.benchmark_batches, loss=args.loss, arcface_scale=args.arcface_scale,
                       arcface_margin=args.arcface_margin, out_dir=out_dir, mitigation=args.mitigation,
                       adv_lambda=args.adv_lambda)
        return

    train(
        lr=args.lr, batch_size=args.batch_size, max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        num_threads=args.num_threads, num_workers=args.num_workers, seed=args.seed,
        min_train_images=args.min_train_images, device_str=args.device,
        loss=args.loss, arcface_scale=args.arcface_scale, arcface_margin=args.arcface_margin,
        out_dir=out_dir, mitigation=args.mitigation, adv_lambda=args.adv_lambda,
        grad_clip_norm=grad_clip_norm,
    )


if __name__ == "__main__":
    main()
