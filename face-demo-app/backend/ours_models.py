"""Wires this project's own trained models into the demo app, alongside
the existing DeepFace path -- M0 (ArcFace identity verification) as an
alternative to recognition.py's DeepFace.verify, and BAL-M1 (the
balanced-dataset demographic classifier) as an extra dataset entry
alongside classifiers.py's fairface/utkface/lfw models.

Both checkpoints are read directly from face_resnet34_training/'s own
output directories (not duplicated into backend/models/) so there is one
source of truth for which checkpoint is "the" model.
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_ROOT = os.path.normpath(os.path.join(BACKEND_DIR, "..", "..", "face_resnet34_training"))
sys.path.insert(0, os.path.join(TRAINING_ROOT, "src", "models"))


def _load_module(unique_name, file_path):
    """frs_m0/model.py and bal_m1/model.py share the filename "model.py" --
    a plain `sys.path.insert` + `from model import ...` for both would
    collide (the second import silently reuses the first one cached in
    sys.modules). Loading each from its exact file path under a unique
    name sidesteps that entirely."""
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    spec = importlib.util.spec_from_file_location(unique_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


_frs_m0_model = _load_module("_frs_m0_model", os.path.join(TRAINING_ROOT, "src", "frs_m0", "model.py"))
_bal_m1_model = _load_module("_bal_m1_model", os.path.join(TRAINING_ROOT, "src", "bal_m1", "model.py"))

M0_CHECKPOINT = os.path.join(TRAINING_ROOT, "outputs", "M0_LFW_arcface", "best.pt")
M0_THRESHOLD = 0.9509  # EER-selected on the LFW validation split, see outputs/M0_LFW_arcface/threshold.json
BAL_M1_CHECKPOINT = os.path.join(TRAINING_ROOT, "experiments", "harmoface", "BAL-M1", "best.pt")
BAL_M1_LABEL_MAPPINGS = os.path.join(TRAINING_ROOT, "experiments", "harmoface", "BAL-M1", "label_mappings.json")

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_cache = {}


def _to_tensor(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return _TF(rgb).unsqueeze(0)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_m0():
    if "m0" in _cache:
        return _cache["m0"]
    if not os.path.exists(M0_CHECKPOINT):
        _cache["m0"] = None
        return None
    # n_train_identities is irrelevant at inference: .embed() only touches
    # self.backbone, never the ArcFace head. But strict=False only
    # tolerates missing/unexpected KEYS, not a shape mismatch on a key
    # present in both -- the checkpoint's arcface_head was sized for the
    # real training identity count, so it's filtered out entirely rather
    # than attempted (same pattern as extract_embeddings.py's reasoning).
    model = _frs_m0_model.M0ArcFaceEmbeddingModel(n_train_identities=1)
    ckpt = _load_checkpoint(M0_CHECKPOINT)
    backbone_only = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    model.load_state_dict(backbone_only, strict=False)
    model.eval()
    _cache["m0"] = model
    return model


def _load_bal_m1():
    if "bal_m1" in _cache:
        return _cache["bal_m1"]
    if not os.path.exists(BAL_M1_CHECKPOINT):
        _cache["bal_m1"] = None
        return None
    import json

    with open(BAL_M1_LABEL_MAPPINGS) as f:
        vocabs = json.load(f)
    model = _bal_m1_model.build_bal_m1_model(len(vocabs["race"]), len(vocabs["gender"]), len(vocabs["age"]), len(vocabs["skin_tone"]))
    ckpt = _load_checkpoint(BAL_M1_CHECKPOINT)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _cache["bal_m1"] = (model, vocabs)
    return _cache["bal_m1"]


def m0_available():
    return _load_m0() is not None


def bal_m1_available():
    return _load_bal_m1() is not None


def m0_embed(face_bgr):
    model = _load_m0()
    if model is None:
        return None
    with torch.no_grad():
        emb = model.embed(_to_tensor(face_bgr))
    return emb.squeeze(0).numpy()


def m0_compare(face1_bgr, face2_bgr):
    """Same shape as recognition.compare()'s return value, so app.py can
    swap between DeepFace and M0 without changing the response contract."""
    e1 = m0_embed(face1_bgr)
    e2 = m0_embed(face2_bgr)
    if e1 is None or e2 is None:
        return None
    score = float(np.dot(e1, e2))  # both already L2-normalised by .embed()
    return {
        "same_person": bool(score >= M0_THRESHOLD),
        "score": round(score, 4),
        "threshold": M0_THRESHOLD,
        "model": "ThirdLens (ours, ArcFace, LFW-trained)",
    }


def bal_m1_classify(face_bgr):
    """Same output shape as classifiers.py's per-dataset result dict, plus
    a skin_tone field classifiers.py's existing datasets don't have."""
    loaded = _load_bal_m1()
    if loaded is None:
        return None
    model, vocabs = loaded
    inv = {task: {v: k for k, v in vocab.items()} for task, vocab in vocabs.items()}
    with torch.no_grad():
        out = model(_to_tensor(face_bgr))
    result = {"source": "bal_m1"}
    for task in ("race", "gender", "age", "skin_tone"):
        probs = F.softmax(out[task], dim=1).squeeze(0).numpy()
        labels = [inv[task][i] for i in range(len(probs))]
        top_i = int(np.argmax(probs))
        if task in ("gender",):
            result["gender"] = labels[top_i].capitalize()
            result["gender_conf"] = round(float(probs[top_i]) * 100, 1)
        elif task == "age":
            result["age"] = labels[top_i]
        else:
            result[task] = {labels[i]: round(float(probs[i]) * 100, 1) for i in range(len(probs))}
    return result
