import cv2
from deepface import DeepFace
import config
import ours_models


def _deepface_compare(path1, path2):
    res = DeepFace.verify(
        img1_path=path1,
        img2_path=path2,
        model_name=config.RECOG_MODEL,
        detector_backend=config.DETECTOR,
        enforce_detection=False,
    )
    dist = float(res["distance"])
    score = max(0.0, 1.0 - dist)
    return {
        "same_person": bool(score >= config.MATCH_THRESHOLD),
        "score": round(score, 3),
        "threshold": config.MATCH_THRESHOLD,
        "model": config.RECOG_MODEL,
    }


def compare_all(path1, path2):
    """Runs every available recognition model and returns all results --
    no upfront choice, since the point of this demo is comparing a
    deployed system (DeepFace) against our own model, not picking one."""
    results = {"deepface": _deepface_compare(path1, path2)}
    if ours_models.m0_available():
        face1 = cv2.imread(path1)
        face2 = cv2.imread(path2)
        thirdlens = ours_models.m0_compare(face1, face2)
        if thirdlens is not None:
            results["thirdlens"] = thirdlens
    return results
