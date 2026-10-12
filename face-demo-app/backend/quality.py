import cv2
import numpy as np


def brightness(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))


def contrast(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray) / 255.0)


def sharpness(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_ratio(bbox, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return float(((x2 - x1) * (y2 - y1)) / (h * w))


def yaw_from_landmarks(landmarks):
    if not landmarks:
        return None
    try:
        le = np.array(landmarks["left_eye"], dtype=float)
        re = np.array(landmarks["right_eye"], dtype=float)
        nose = np.array(landmarks["nose"], dtype=float)
    except (KeyError, TypeError):
        return None
    eye_mid = (le + re) / 2.0
    eye_dist = np.linalg.norm(re - le) + 1e-6
    offset = (nose[0] - eye_mid[0]) / eye_dist
    return float(np.degrees(np.arctan(offset * 2.0)))


def ita(face_bgr):
    lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0] * (100.0 / 255.0)
    b = lab[:, :, 2] - 128.0
    Lm = np.median(L)
    bm = np.median(b) + 1e-6
    return float(np.degrees(np.arctan((Lm - 50.0) / bm)))


def measure(full_bgr, face_bgr, det):
    return {
        "brightness": round(brightness(face_bgr), 1),
        "contrast": round(contrast(face_bgr), 3),
        "sharpness": round(sharpness(face_bgr), 1),
        "head_turn_deg": (round(yaw_from_landmarks(det["landmarks"]), 1)
                          if yaw_from_landmarks(det["landmarks"]) is not None else None),
        "face_ratio": round(face_ratio(det["bbox"], full_bgr.shape) * 100.0, 1),
        "ita_deg": round(ita(face_bgr), 1),
    }
