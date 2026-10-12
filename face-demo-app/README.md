# Face Demo App

A two-photo face demo. A person enters their name, signs a consent form, takes
two photos, and the app reports:

- estimated age and gender
- how three datasets classify them (FairFace, UTKFace, LFW), switchable on the results
- whether the two photos are the same person, and what changed between them
  (brightness, contrast, sharpness, head turn, face size, skin tone reading)

The app has two parts: a Python backend that does the face work, and a browser
frontend that runs the camera and shows the results.

## What each file does

Backend
- app.py            the web server and the /api/analyze endpoint
- detect.py         finds the largest face (RetinaFace) and crops it
- quality.py        brightness, contrast, sharpness, head turn, face size, ITA skin tone
- recognition.py    compares two photos (ArcFace) and gives a match score
- classifiers.py    runs the three dataset models, or a fallback if a model is missing
- model_arch.py     the ResNet-34 classifier used by each dataset model
- train_classifier.py  trains one dataset model from a CSV of labelled faces
- config.py         thresholds, label lists, model paths
- models/           put fairface.pth, utkface.pth, lfw.pth here

Frontend
- index.html        the screens
- styles.css        the look
- app.js            camera, flow, calling the backend, drawing results

## Setup

You need Python 3.10 or 3.11.

    cd backend
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

The first run downloads the ArcFace and analysis model files. This needs
internet once, then it works offline.

## Run

    cd backend
    python app.py

Open http://localhost:5000 in Chrome.

The camera only works on http://localhost or on https. It will not work if you
open index.html straight from the file system.

## The three dataset models

Out of the box the app uses a general model and marks every dataset result as
"general model (fallback)". To show the real per-dataset behaviour, train your
own models.

1. Make a CSV per dataset with these columns:

       image_path,age,gender,race

   - age must be one of: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
   - gender must be: Female or Male
   - race must match the dataset labels in config.py
     FairFace: East Asian, Southeast Asian, Indian, Black, White, Middle Eastern, Latino Hispanic
     UTKFace:  White, Black, Asian, Indian, Others
     LFW has no race labels, so leave the race column empty for LFW.

2. Train:

       python train_classifier.py --dataset fairface --csv fairface_labels.csv
       python train_classifier.py --dataset utkface  --csv utkface_labels.csv
       python train_classifier.py --dataset lfw       --csv lfw_labels.csv

   This saves models/fairface.pth and so on. The app picks them up automatically
   and the "fallback" label disappears.

## Important notes

- The dataset percentages are the model's confidence for each label on one
  photo. They do not measure ancestry or identity. The results screen says this.
- LFW has no official race labels. The LFW model reports age and gender only.
  This is shown on purpose: different datasets label people differently.
- Nothing is stored. Photos stay in the browser and are cleared at the end.
- Showing race categories was a change from the earlier no-race demo plan.
  Confirm this with your supervisor and your ethics approval before any live use.
- Do not use anyone's photos without their consent.
