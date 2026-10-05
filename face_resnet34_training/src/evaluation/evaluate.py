"""Step 12: locked test-set evaluation.

Runs a checkpoint exactly once against the test split -- which the split
freeze (splits.py) guarantees has never been touched by training, tuning,
or any model-selection decision -- and saves everything a later metrics
pass could need, without ever re-running the model:

  - true label(s) and predicted label(s)
  - raw logits AND probabilities (softmax for classification heads,
    sigmoid for LFW's multi-label head) -- needed for ROC/PR curves,
    calibration, log loss, and any future re-thresholding
  - the 512-d backbone embedding -- needed for any future face-verification
    analysis (FMR/FNMR/EER) or embedding-space exploration; these cannot be
    reconstructed from logits alone, so this is saved even though the
    current metrics pass doesn't use it
  - race/gender/age_group/ita_value/ita_group/identity_id metadata, for
    subgroup and fairness breakdowns

Output is parquet (not CSV): logits/probabilities/embeddings are
list-valued columns, which parquet stores natively and CSV would only
stringify.

Verifies the split manifest hash before loading the test set, same as
FaceDataset does for train/val: the test set must never be silently
regenerated either.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from face_dataset import FaceDataset
from transforms import eval_transform
from label_encoders import load_labels
from heads import build_model

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
CLASSIFICATION_TASKS = ["race", "gender", "age_group"]


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _forward_with_embedding(model, imgs: torch.Tensor):
    """Runs the backbone exactly once and reuses its output for both the
    task head(s) and the saved embedding, rather than calling model(imgs)
    (which would recompute the backbone a second time)."""
    embedding = model.backbone(imgs)
    if hasattr(model, "heads"):  # MultiTaskResNet34 (fairface/utkface)
        outputs = {task: head(embedding) for task, head in model.heads.items()}
    else:  # AttributeResNet34 (lfw)
        outputs = {"attributes": model.head(embedding)}
    return outputs, embedding


def evaluate(
    dataset: str,
    checkpoint_path: Path,
    batch_size: int = 32,
    device_str: str = "cpu",
    out_dir: Path | None = None,
    max_batches: int | None = None,
    method: str = "M1",
    seed: int = 0,
) -> dict:
    """max_batches truncates the test set -- wiring checks only, never a
    real locked-test-set evaluation; the real run must cover all of test."""
    device = torch.device(device_str)
    labels = load_labels(dataset)
    model = build_model(dataset, labels).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # strict=False: M4's checkpoint state_dict is a strict superset of the
    # plain model's keys (an extra ita_head.* discriminator, unused at test
    # time) -- missing_keys would still mean a real mismatch, so that case
    # is not silently swallowed.
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"Checkpoint is missing required keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[evaluate] ignoring {len(load_result.unexpected_keys)} unexpected checkpoint key(s) "
              f"(e.g. M4's adversarial head): {load_result.unexpected_keys[:5]}")
    model.eval()

    test_ds = FaceDataset(dataset, "test", transform=eval_transform())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    rows = []
    with torch.no_grad():
        for batch_idx, (imgs, targets, meta) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            imgs = imgs.to(device)
            outputs, embedding = _forward_with_embedding(model, imgs)
            n = imgs.size(0)
            for i in range(n):
                row = {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "split": "test",
                    **{k: _scalar(v[i]) for k, v in meta.items()},
                    "embedding": embedding[i].tolist(),
                }
                if dataset == "lfw":
                    logits = outputs["attributes"][i]
                    probs = torch.sigmoid(logits)
                    row["attributes_logits"] = logits.tolist()
                    row["attributes_probs"] = probs.tolist()
                    row["attributes_pred"] = (probs >= 0.5).long().tolist()
                    row["attributes_true"] = targets["attributes"][i].long().tolist()
                else:
                    for task in CLASSIFICATION_TASKS:
                        logits = outputs[task][i]
                        probs = F.softmax(logits, dim=0)
                        row[f"{task}_logits"] = logits.tolist()
                        row[f"{task}_probs"] = probs.tolist()
                        row[f"{task}_pred"] = int(logits.argmax().item())
                        row[f"{task}_true"] = targets[task][i].item()
                rows.append(row)

    predictions = pd.DataFrame(rows)
    out_dir = out_dir or (EXPERIMENTS_DIR / dataset / "test_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_dir / "test_predictions.parquet", index=False)

    metrics = compute_overall_metrics(dataset, predictions, labels)
    metrics["checkpoint_path"] = str(checkpoint_path)
    metrics["checkpoint_epoch"] = checkpoint.get("epoch")
    metrics["method"] = method
    metrics["seed"] = seed
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def compute_overall_metrics(dataset: str, predictions: pd.DataFrame, labels: dict) -> dict:
    """Quick headline numbers alongside the full predictions dump -- the
    comprehensive classification/fairness/bootstrap analysis lives in
    metrics.py, computed from the saved parquet, not here."""
    if dataset == "lfw":
        import numpy as np

        preds = np.array(predictions["attributes_pred"].tolist())
        trues = np.array(predictions["attributes_true"].tolist())
        per_attr_acc = (preds == trues).mean(axis=0)
        return {
            "n_test": len(predictions),
            "mean_attribute_accuracy": float(per_attr_acc.mean()),
            "per_attribute_accuracy": {
                name: float(acc) for name, acc in zip(labels["attributes"], per_attr_acc)
            },
        }

    metrics = {"n_test": len(predictions)}
    for task in CLASSIFICATION_TASKS:
        valid = predictions[predictions[f"{task}_true"] != -1]
        if valid.empty:
            continue
        metrics[f"{task}_accuracy"] = float((valid[f"{task}_pred"] == valid[f"{task}_true"]).mean())
        metrics[f"{task}_n_valid"] = int(len(valid))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["fairface", "utkface", "lfw"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="M1", help="M1/M2/M3/M4 -- recorded in the predictions file for later comparison.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Truncate the test set to this many batches -- wiring checks only, never a real evaluation.",
    )
    args = parser.parse_args()
    metrics = evaluate(
        args.dataset,
        args.checkpoint,
        args.batch_size,
        args.device,
        max_batches=args.max_batches,
        method=args.method,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
