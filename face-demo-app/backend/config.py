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
        # LFWA+'s 73 trained attributes include White/Black/Asian/Indian as
        # independent flags -- not a forced single choice the way
        # FairFace/UTKFace's race head is, see classifiers.py's
        # _lfw_from_attributes. LFW itself still has no OFFICIAL race
        # annotation; this is our own model's reading of those four flags.
        "race": ["White", "Black", "Asian", "Indian"],
        "has_race": True,
    },
}

# BAL-M1: this project's own classifier, trained on HarmoFace-95K (balanced
# across race/gender/age/skin tone). Not in DATASETS above since its label
# vocab differs (7-way race, plus a skin_tone head the other datasets don't
# have) -- classifiers.py adds it to the response under "harmoface_balanced"
# when ours_models.bal_m1_available(); this entry is just its display metadata.
HARMOFACE_BALANCED_META = {
    "label": "HarmoFace Balanced",
    "race": ["White", "Black", "Indian", "East Asian", "Southeast Asian", "Middle Eastern", "Latino"],
    "has_race": True,
    "has_skin_tone": True,
    "skin_tone": ["I-II (very light)", "II-III (light)", "III-IV (intermediate)",
                  "IV-V (tan)", "V (brown)", "VI (dark)"],
}
