import numpy as np
from retinaface import RetinaFace


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
