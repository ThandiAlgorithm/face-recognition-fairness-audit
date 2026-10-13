FROM python:3.11-slim

# opencv-python needs these system libraries, which aren't in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/

EXPOSE 8080

# --chdir so app:app resolves inside backend/, while config.py's sibling
# path to ../frontend still lines up with the COPY layout above.
CMD ["gunicorn", "--chdir", "backend", "-w", "1", "-b", "0.0.0.0:8080", "--timeout", "120", "app:app"]
