import os

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")

MATCH_THRESHOLD = 0.45
RECOG_MODEL = "ArcFace"
DETECTOR = "retinaface"

AGE_BUCKETS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
GENDER = ["Female", "Male"]

DATASETS = {
    "fairface": {
        "label": "FairFace",
        "race": ["East Asian", "Southeast Asian", "Indian", "Black",
                 "White", "Middle Eastern", "Latino Hispanic"],
        "has_race": True,
    },
    "utkface": {
        "label": "UTKFace",
        "race": ["White", "Black", "Asian", "Indian", "Others"],
        "has_race": True,
    },
    "lfw": {
        "label": "LFW",
        "race": [],
        "has_race": False,
    },
}
