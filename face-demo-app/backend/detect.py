import os

import cv2
import numpy as np
import yaml
from retinaface import RetinaFace

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(BACKEND_DIR, "..", ".."))
_PREPROCESSING_YAML = os.path.join(REPO_ROOT, "configs", "preprocessing.yaml")

with open(_PREPROCESSING_YAML) as f:
    _ALIGNMENT_CFG = yaml.safe_load(f)["alignment"]

_OUTPUT_SIZE = int(_ALIGNMENT_CFG["output_size_px"])
_CANONICAL = np.array(_ALIGNMENT_CFG["canonical_landmark_positions"], dtype=np.float32) * _OUTPUT_SIZE


def detect_primary(bgr):
    faces = RetinaFace.detect_faces(bgr)
    if not isinstance(faces, dict) or len(faces) == 0:
        return None
    best = None
    best_area = -1
    for key in faces:
        f = faces[key]
        x1, y1, x2, y2 = f["facial_area"]
        area = (x2 - x1) * (y2 - y1)
        score = f.get("score", 1.0)
        rank = area * (1.0 + score)
        if rank > best_area:
            best_area = rank
            best = f
    x1, y1, x2, y2 = best["facial_area"]
    return {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "score": float(best.get("score", 1.0)),
        "landmarks": best.get("landmarks", {}),
        "n_faces": len(faces),
    }


def crop(bgr, bbox, pad=0.15):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw))
    y1 = max(0, int(y1 - pad * bh))
    x2 = min(w, int(x2 + pad * bw))
    y2 = min(h, int(y2 + pad * bh))
    return bgr[y1:y2, x1:x2]


def align(bgr, landmarks):
    """Landmark-based similarity-transform alignment, matching
    configs/preprocessing.yaml's "alignment" block exactly -- the same
    protocol used to build the aligned crops our own models (M0/ThirdLens,
    BAL-M1, the LFW attribute model) were trained on. crop()'s plain
    bbox+padding is NOT that protocol: it has no rotation correction and no
    landmark-based scale/position normalisation, so feeding it to our own
    models is a real train/inference distribution mismatch (not just a
    cosmetic difference) -- most visible on attributes sensitive to exact
    framing, like Smiling.

    retinaface's landmark dict names eyes/mouth corners by the SUBJECT's
    own anatomical left/right, not by which side of the image they appear
    on -- the opposite of canonical_landmark_positions' convention (index 0
    is the point closest to x=0, i.e. whichever side appears on the left of
    the image). So "right_eye" (the subject's right eye, appearing on the
    image's LEFT side for a frontal face) maps to canonical index 0, not
    "left_eye". Verified against a real detection, not assumed.
    """
    required = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")
    if not landmarks or any(k not in landmarks for k in required):
        return None
    source = np.array([landmarks[k] for k in required], dtype=np.float32)
    transform, _ = cv2.estimateAffinePartial2D(source, _CANONICAL, method=cv2.LMEDS)
    if transform is None:
        return None
    return cv2.warpAffine(bgr, transform, (_OUTPUT_SIZE, _OUTPUT_SIZE), flags=cv2.INTER_LINEAR)
