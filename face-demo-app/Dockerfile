FROM python:3.11-slim

# opencv-python needs these system libraries, which aren't in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Build context is the monorepo root (not face-demo-app/) -- the backend
# reaches outside its own folder at import time: detect.py reads
# ../../configs/preprocessing.yaml, and ours_models.py loads model-definition
# source (not the trained checkpoints, which are gitignored and optional --
# see ours_models.py's os.path.exists guards) from
# ../../face_resnet34_training/src/. Both need to exist in the image.
WORKDIR /app

COPY face-demo-app/backend/requirements.txt face-demo-app/backend/requirements.txt
RUN pip install --no-cache-dir -r face-demo-app/backend/requirements.txt

COPY face-demo-app/backend/ face-demo-app/backend/
COPY face-demo-app/frontend/ face-demo-app/frontend/
COPY configs/ configs/
COPY face_resnet34_training/src/ face_resnet34_training/src/

EXPOSE 8080

# --chdir so app:app resolves inside backend/. Shell form (not exec form) so
# $PORT gets expanded -- Render assigns this dynamically; ${PORT:-8080}
# falls back to 8080 for a plain local `docker run`.
CMD gunicorn --chdir face-demo-app/backend -w 1 -b 0.0.0.0:${PORT:-8080} --timeout 120 app:app
