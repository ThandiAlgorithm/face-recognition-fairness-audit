"""M0 spec sections 20-21: select ONE global threshold from validation data
only, at the validation EER operating point. Test data is never touched
here -- this script only reads val_embeddings/val_pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import cosine_similarity_matrix_pairs, compute_roc, eer_from_roc

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "M0_LFW"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir

    embs = np.load(out_dir / "embeddings" / "val_embeddings.npy")
    meta = pd.read_csv(out_dir / "embeddings" / "val_metadata.csv")
    emb_by_id = {iid: e for iid, e in zip(meta["image_id"], embs)}

    pairs = pd.read_csv(out_dir / "pairs" / "val_pairs.csv")
    a = np.stack([emb_by_id[i] for i in pairs["image_id_a"]])
    b = np.stack([emb_by_id[i] for i in pairs["image_id_b"]])
    scores = cosine_similarity_matrix_pairs(a, b)
    labels = pairs["same_identity"].to_numpy()

    assert scores.min() >= -1.0001 and scores.max() <= 1.0001, "cosine scores out of [-1,1]"

    thresholds, fmr, fnmr = compute_roc(scores, labels)
    eer, threshold = eer_from_roc(thresholds, fmr, fnmr)

    result = {"selected_on": "validation", "criterion": "EER", "threshold": threshold, "val_eer": eer}
    (out_dir / "threshold.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
