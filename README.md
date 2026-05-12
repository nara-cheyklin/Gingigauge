# GINGIGAUGE

**Deep-learning Based Measurement of Keratinized Gingiva Width (KGW)**

A collaborative project between the Faculty of Engineering and the Faculty of Dentistry, Chulalongkorn University.

---

## Overview

Gingigauge is a web-based clinical tool that automatically measures the **Keratinized Gingiva Width (KGW)** — a key indicator of long-term oral health — from RealSense depth camera recordings (`.bag` files). The system processes RGB and depth data through an AI pipeline to produce per-tooth KGW measurements in millimeters, classified as **Healthy**, **At Risk**, or **Recession**.

---

## Pipeline

```
Upload .bag file
      │
      ▼
Direct PUT to GCS (signed URL — bypasses Cloud Run 32 MB limit)
      │
      ▼
Extract first RGB frame → Preview shown to user
      │
      ▼
Optional ROI crop (user draws region on preview in browser)
      │
      ▼
Extract RGB + aligned depth frame + camera intrinsics from .bag
      │
      ▼
CLAHE contrast enhancement (LAB colour space, L-channel)
      │
      ├──► Vertex AI gingiva segmentation → binary gingiva mask
      │
      └──► Roboflow tooth detection → per-tooth centre points
                    │
                    ▼
        Split mask into upper / lower gingiva bands
                    │
                    ▼
        Per-tooth vertical column scan on mask
        → inner boundary (CEJ side) + outer boundary (MGJ side)
                    │
                    ▼
        Back-project pixel pairs to 3D using depth map + intrinsics
        → Euclidean distance in mm = KGW per tooth
                    │
                    ▼
        FDI tooth ID mapping (front / right / left view)
                    │
                    ▼
        Annotated image + per-tooth JSON results
                    │
                    ▼
        Interpret: Healthy (≥ 3.5 mm) | At Risk (2.5–3.5 mm) | Recession (< 2.5 mm)
```

---

## Architecture

| Component | Technology |
|---|---|
| Frontend | Static HTML + Tailwind CSS + Vanilla JS |
| Frontend Hosting | Firebase Hosting |
| Backend API | FastAPI + Uvicorn (Python) |
| Backend Hosting | Google Cloud Run (`us-central1`) |
| Gingiva Segmentation | Vertex AI custom endpoint (SegFormer) |
| Tooth Detection | Roboflow serverless API (`toothcariesdetection/2`) |
| File Storage | Google Cloud Storage |
| Database | Google Cloud Firestore (Native mode) |
| Secrets | Google Cloud Secret Manager |

---

## Project Structure

```
gingigauge/
├── frontend/
│   ├── index.html          # Main SPA — upload, crop, results, cloud save
│   ├── about.html          # About page
│   ├── contact.html        # Contact form
│   ├── export.js           # Download, print, dental chart rendering
│   ├── config.js           # API base URL config (set before deploying)
│   ├── input.css           # Tailwind source
│   ├── style.css           # Compiled Tailwind output
│   └── images/
├── backend/
│   ├── main.py             # FastAPI app, CORS, router registration
│   ├── config/
│   │   └── settings.py     # Environment variable loading, clinical thresholds
│   ├── routes/
│   │   ├── predict.py      # /predict/rosbag/upload-url, /predict/rosbag/preview, /predict/rosbag
│   │   ├── upload.py       # /upload (save result to GCS + Firestore)
│   │   └── contact.py      # /contact (save contact form to Firestore)
│   └── services/
│       ├── full_pipeline.py      # Core KGW pipeline (CLAHE → seg → detect → measure → annotate)
│       ├── inference.py          # Vertex AI gingiva segmentation client
│       ├── rosbag_processing.py  # .bag parser, RGB/depth extraction, intrinsics
│       ├── rosbag_upload.py      # GCS signed URL generation, download, cleanup
│       ├── upload.py             # GCS image upload + Firestore record creation
│       └── contact.py            # Firestore contact message writer
├── firebase.json           # Firebase Hosting config
├── bucket-cors.json        # GCS CORS policy for direct browser uploads
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime version for Cloud Run
├── package.json            # Node dev dependencies (Tailwind CLI)
├── DEPLOY.md               # Step-by-step GCP deployment guide
└── .env                    # Local secrets (gitignored — see below)
```

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js (for Tailwind CSS compilation)
- A `.env` file in the repo root (see below)

### 1. Clone and install

```bash
git clone <your-repo-url>
cd gingigauge

# Python dependencies
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Node dependencies (Tailwind only)
npm install
```

### 2. Create your `.env` file

```env
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-gcs-bucket-name
GCP_ENDPOINT_ID=your-vertex-endpoint-id
GCP_ENDPOINT_IMAGE_MAX_SIZE=0
VERTEX_PROJECT_ID=your-vertex-project-id
VERTEX_REGION=us-central1
ROBOFLOW_API_KEY=your-roboflow-api-key
```

### 3. Run the backend

```bash
uvicorn backend.main:app --reload
```

FastAPI will also serve the `frontend/` folder statically at `http://localhost:8000`.

### 4. (Optional) Recompile Tailwind CSS

```bash
npx @tailwindcss/cli -i frontend/input.css -o frontend/style.css --watch
```

---

## Configuration Before Deploying

Before pushing code, update the two files below with your real values:

**`frontend/config.js`** — set to your Cloud Run service URL:
```js
window.API_BASE_URL = "https://YOUR_CLOUD_RUN_URL.us-central1.run.app";
```

**`bucket-cors.json`** — set to your Firebase Hosting domains:
```json
"origin": ["https://YOUR_FIREBASE_PROJECT.web.app", "https://YOUR_FIREBASE_PROJECT.firebaseapp.com"]
```

Then apply the CORS policy to your bucket:
```bash
gcloud storage buckets update gs://YOUR_BUCKET_NAME --cors-file=bucket-cors.json
```

---

## Deployment

See [DEPLOY.md](DEPLOY.md) for the full step-by-step guide. In brief:

```bash
# Backend → Cloud Run
gcloud run deploy gingigauge-api --source . --region us-central1

# Frontend → Firebase Hosting
firebase deploy --only hosting
```

---

## KGW Clinical Thresholds

| Result | KGW Value |
|---|---|
| Healthy | ≥ 3.5 mm |
| At Risk | 2.5 mm – 3.5 mm |
| Recession | < 2.5 mm |

---

## GCP Services Required

Enable the following APIs in your GCP project before deploying:

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  firestore.googleapis.com \
  storage.googleapis.com \
  firebasehosting.googleapis.com
```

---

## Security Notes

- The Roboflow API key is stored in **GCP Secret Manager** and injected into Cloud Run at deploy time — it is never in the container image or source code.
- Cloud Run uses a dedicated service account (`gingigauge-runtime`) with minimal IAM roles.
- GCS uploads from the browser go via **v4 signed URLs** (15-minute expiry) directly to GCS, bypassing Cloud Run's 32 MB request body limit.
- Patient result images stored in GCS should use **signed URLs** rather than public bucket access for production clinical use.
