from deepface import DeepFace
import config


def compare(path1, path2):
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
