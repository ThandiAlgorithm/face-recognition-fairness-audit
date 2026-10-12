"""M0 spec section 13: extract 512-d L2-normalized embeddings for val and
test images from the trained best.pt, once. Everything downstream
(pairs, thresholds, verification metrics) reads these saved embeddings
rather than re-running the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_m0_table, M0EvalDataset, APPEARANCE_ATTR_COLS, DEMOGRAPHIC_COLS
from model import M0EmbeddingModel
from utils import m0_eval_transform

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    ids, embs = [], []
    for imgs, image_ids in loader:
        imgs = imgs.to(device)
        emb = model.embed(imgs).cpu().numpy()
        ids.extend(image_ids)
        embs.append(emb)
    return ids, np.concatenate(embs, axis=0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Defaults to <out-dir>/best.pt.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                         help="Where embeddings/ gets written -- match this to the checkpoint's own "
                              "training out-dir (e.g. outputs/M0_LFW_arcface for the ArcFace variant).")
    args = parser.parse_args()
    out_dir = args.out_dir
    checkpoint_path = args.checkpoint or str(out_dir / "best.pt")

    device = torch.device(args.device)
    # weights_only=False: this checkpoint is our own (trained in this
    # project, never loaded from an external source) and its rng_states
    # entry contains numpy objects that torch>=2.6's safer default refuses
    # to unpickle automatically. Older torch (<2.0, e.g. the cluster's
    # 1.12.1) doesn't accept this kwarg at all -- fall back to the plain
    # call there, where unpickling everything is simply the only behavior.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint["identity_vocab"]
    # embed() only ever touches self.backbone, which every M0 model variant
    # (v1 plain, v2 ArcFace, v3 adversarial) builds identically -- always
    # load into the simplest class with strict=False rather than needing
    # to reconstruct the exact training-time architecture (v3's adversary
    # heads need n_race/n_gender/n_age, which aren't worth plumbing through
    # here just to immediately discard them). Unexpected/missing keys
    # (arcface_head, grl, race_head, ...) are exactly the non-backbone
    # parameters this script never uses.
    model = M0EmbeddingModel(len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    df = load_m0_table()
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)

    for split in ["val", "test"]:
        ds = M0EvalDataset(df, split, m0_eval_transform())
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        ids, embs = extract(model, loader, device)

        assert embs.shape[1] == 512, f"expected 512-d embeddings, got {embs.shape[1]}"
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), f"embeddings not unit-norm: min={norms.min()} max={norms.max()}"

        np.save(out_dir / "embeddings" / f"{split}_embeddings.npy", embs)
        meta = df[df["image_id"].isin(ids)].set_index("image_id").loc[ids].reset_index()
        meta.to_csv(out_dir / "embeddings" / f"{split}_metadata.csv", index=False)
        print(f"{split}: {embs.shape[0]} embeddings, shape={embs.shape}, norm range=[{norms.min():.4f},{norms.max():.4f}]")


if __name__ == "__main__":
    main()
