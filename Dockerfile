# Gingigauge backend - Cloud Run image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# libgomp1 is needed by torch; opencv-python-headless brings its own deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache the layer.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the backend package; frontend/ is excluded via .dockerignore.
COPY backend/ ./backend/

EXPOSE 8080

# Cloud Run sets $PORT (default 8080). Single uvicorn worker is the right
# default for Cloud Run since the platform handles horizontal scaling.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
