import os
import numpy as np
import torch
import cv2
from torchvision import transforms

import config
from model_arch import MultiHeadResNet

_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_cache = {}


def _load(name):
    if name in _cache:
        return _cache[name]
    ds = config.DATASETS[name]
    path = os.path.join(config.MODEL_DIR, name + ".pth")
    if not os.path.exists(path):
        _cache[name] = None
        return None
    model = MultiHeadResNet(
        n_age=len(config.AGE_BUCKETS),
        n_race=len(ds["race"]) if ds["has_race"] else 0,
        with_race=ds["has_race"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    _cache[name] = model
    return model


def _softmax(t):
    return torch.softmax(t, dim=1).detach().numpy()[0]


def _custom(name, face_bgr):
    model = _load(name)
    if model is None:
        return None
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    x = _TF(rgb).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    ds = config.DATASETS[name]
    age_p = _softmax(out["age"])
    gen_p = _softmax(out["gender"])
    result = {
        "source": "custom",
        "age": config.AGE_BUCKETS[int(np.argmax(age_p))],
        "gender": config.GENDER[int(np.argmax(gen_p))],
        "gender_conf": round(float(np.max(gen_p)) * 100, 1),
    }
    if ds["has_race"]:
        race_p = _softmax(out["race"])
        result["race"] = {ds["race"][i]: round(float(race_p[i]) * 100, 1)
                          for i in range(len(ds["race"]))}
    else:
        result["race"] = None
    return result


_DF_RACE_MAP = {
    "fairface": {
        "East Asian": ["asian"], "Southeast Asian": [], "Indian": ["indian"],
        "Black": ["black"], "White": ["white"],
        "Middle Eastern": ["middle eastern"], "Latino Hispanic": ["latino hispanic"],
    },
    "utkface": {
        "White": ["white"], "Black": ["black"], "Asian": ["asian"],
        "Indian": ["indian"], "Others": ["middle eastern", "latino hispanic"],
    },
}


def _fallback(name, df_analysis):
    ds = config.DATASETS[name]
    age = int(df_analysis["age"])
    bucket = config.AGE_BUCKETS[-1]
    edges = [2, 9, 19, 29, 39, 49, 59, 69]
    for i, e in enumerate(edges):
        if age <= e:
            bucket = config.AGE_BUCKETS[i]
            break
    gender_scores = df_analysis["gender"]
    gender = "Female" if gender_scores["Woman"] >= gender_scores["Man"] else "Male"
    result = {
        "source": "general_fallback",
        "age": bucket,
        "gender": gender,
        "gender_conf": round(float(max(gender_scores.values())), 1),
    }
    if ds["has_race"]:
        race_raw = {k.lower(): float(v) for k, v in df_analysis["race"].items()}
        mapping = _DF_RACE_MAP[name]
        merged = {}
        for label, keys in mapping.items():
            merged[label] = round(sum(race_raw.get(k, 0.0) for k in keys), 1)
        total = sum(merged.values()) or 1.0
        result["race"] = {k: round(v / total * 100, 1) for k, v in merged.items()}
    else:
        result["race"] = None
    return result


def classify_all(face_bgr, df_analysis):
    out = {}
    for name in config.DATASETS:
        custom = _custom(name, face_bgr)
        out[name] = custom if custom is not None else _fallback(name, df_analysis)
    return out
