import base64
import tempfile
import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from deepface import DeepFace

import config
import detect
import quality
import recognition
import classifiers

FRONTEND = os.path.join(os.path.dirname(config.BASE), "frontend")
app = Flask(__name__, static_folder=FRONTEND, static_url_path="")
CORS(app)


def decode(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def analyse_one(bgr):
    det = detect.detect_primary(bgr)
    if det is None:
        return {"face_found": False}
    face = detect.crop(bgr, det["bbox"])
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, face)
    df = DeepFace.analyze(
        img_path=tmp.name,
        actions=["age", "gender", "race"],
        detector_backend="skip",
        enforce_detection=False,
    )
    os.unlink(tmp.name)
    df = df[0] if isinstance(df, list) else df
    return {
        "face_found": True,
        "n_faces": det["n_faces"],
        "quality": quality.measure(bgr, face, det),
        "datasets": classifiers.classify_all(face, df),
        "_face": face,
    }


def build_diff(q1, q2):
    metrics = [
        ("brightness", "Brightness", 100.0),
        ("contrast", "Contrast", 0.3),
        ("sharpness", "Sharpness", 200.0),
        ("head_turn_deg", "Head turn", 15.0),
        ("face_ratio", "Face size", 30.0),
        ("ita_deg", "Skin tone reading (ITA)", 20.0),
    ]
    rows = []
    biggest = None
    biggest_val = -1
    for key, label, scale in metrics:
        a, b = q1.get(key), q2.get(key)
        if a is None or b is None:
            rows.append({"key": key, "label": label, "photo1": a, "photo2": b, "change": None})
            continue
        change = abs(b - a) / scale
        rows.append({"key": key, "label": label, "photo1": a, "photo2": b,
                     "change": round(change, 3)})
        if change > biggest_val:
            biggest_val = change
            biggest = (key, label, a, b)
    message = "The two photos are very similar."
    if biggest and biggest_val > 0.15:
        key, label, a, b = biggest
        direction = "higher" if b > a else "lower"
        message = (f"The biggest difference is {label.lower()}. "
                   f"Photo 2 is {direction} than Photo 1. "
                   f"Your face did not change, the capture conditions did.")
    return {"rows": rows, "biggest_key": biggest[0] if biggest else None, "message": message}


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    try:
        img1 = decode(body["photo1"])
        img2 = decode(body["photo2"])
    except (KeyError, TypeError):
        return jsonify({"error": "two photos are required"}), 400
    if img1 is None or img2 is None:
        return jsonify({"error": "could not read the photos"}), 400

    r1 = analyse_one(img1)
    r2 = analyse_one(img2)
    if not r1["face_found"] or not r2["face_found"]:
        return jsonify({
            "face_found": {"photo1": r1["face_found"], "photo2": r2["face_found"]},
            "error": "no face was found in one of the photos",
        }), 200

    t1 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    t2 = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(t1.name, r1["_face"])
    cv2.imwrite(t2.name, r2["_face"])
    match = recognition.compare(t1.name, t2.name)
    os.unlink(t1.name)
    os.unlink(t2.name)

    for r in (r1, r2):
        r.pop("_face", None)

    return jsonify({
        "face_found": {"photo1": True, "photo2": True},
        "photo1": r1,
        "photo2": r2,
        "recognition": match,
        "diff": build_diff(r1["quality"], r2["quality"]),
        "dataset_meta": {k: {"label": v["label"], "has_race": v["has_race"],
                             "race_labels": v["race"]}
                         for k, v in config.DATASETS.items()},
    })


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
