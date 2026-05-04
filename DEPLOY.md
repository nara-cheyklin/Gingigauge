# Gingigauge — Deployment Guide

This guide walks through deploying:

- **Backend** (FastAPI) → **Cloud Run** in `us-central1`
- **Frontend** (static HTML/CSS/JS) → **Firebase Hosting** at `<your-project>.web.app`

You'll run all commands from the repo root (`Documents/Final_Project/gingigauge`).

---

## 0. One-time prerequisites

You already have `gcloud`. You also need the Firebase CLI:

```powershell
npm install -g firebase-tools
```

Authenticate both CLIs (each pops a browser window):

```powershell
gcloud auth login
gcloud auth application-default login
firebase login
```

Set the active project (replace `your-project-id` with your real GCP project ID — same one that appears in `.env` as `GCP_PROJECT_ID`):

```powershell
gcloud config set project your-project-id
gcloud config set run/region us-central1
```

Open `.firebaserc` and replace `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` with the same project ID.

---

## 1. Enable the GCP APIs you need

```powershell
gcloud services enable `
  run.googleapis.com `
  cloudbuild.googleapis.com `
  artifactregistry.googleapis.com `
  secretmanager.googleapis.com `
  aiplatform.googleapis.com `
  firestore.googleapis.com `
  storage.googleapis.com `
  firebasehosting.googleapis.com
```

(In bash/zsh use `\` instead of the backtick line continuation.)

---

## 2. Store the Roboflow API key in Secret Manager

```powershell
# Read your current value from .env, then:
gcloud secrets create ROBOFLOW_API_KEY --replication-policy=automatic
echo -n "YOUR_ROBOFLOW_KEY_VALUE" | gcloud secrets versions add ROBOFLOW_API_KEY --data-file=-
```

On Windows PowerShell, use this instead of `echo -n`:

```powershell
"YOUR_ROBOFLOW_KEY_VALUE" | Out-File -NoNewline -Encoding ascii roboflow.tmp
gcloud secrets versions add ROBOFLOW_API_KEY --data-file=roboflow.tmp
Remove-Item roboflow.tmp
```

---

## 3. Create a dedicated service account for Cloud Run

Using a dedicated SA (rather than the default Compute SA) keeps the blast radius small.

```powershell
$PROJECT_ID = (gcloud config get-value project)
$SA_NAME    = "gingigauge-runtime"
$SA_EMAIL   = "$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

gcloud iam service-accounts create $SA_NAME `
  --display-name="Gingigauge Cloud Run runtime"

# Vertex AI predict
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$SA_EMAIL" --role="roles/aiplatform.user"

# Firestore (contact_messages, patient_records)
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$SA_EMAIL" --role="roles/datastore.user"

# GCS (results bucket writes)
gcloud projects add-iam-policy-binding $PROJECT_ID `
  --member="serviceAccount:$SA_EMAIL" --role="roles/storage.objectAdmin"

# Read the Roboflow secret
gcloud secrets add-iam-policy-binding ROBOFLOW_API_KEY `
  --member="serviceAccount:$SA_EMAIL" --role="roles/secretmanager.secretAccessor"
```

(Bash users: replace `$VAR` with `${VAR}` and use `=` without the leading `$`.)

---

## 4. Deploy the backend to Cloud Run

From the repo root, with the `Dockerfile` in place:

```powershell
gcloud run deploy gingigauge-api `
  --source . `
  --region us-central1 `
  --service-account $SA_EMAIL `
  --allow-unauthenticated `
  --memory 4Gi `
  --cpu 2 `
  --timeout 600 `
  --concurrency 4 `
  --min-instances 0 `
  --max-instances 5 `
  --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=us-central1,GCS_BUCKET_NAME=YOUR_BUCKET_NAME,GCP_ENDPOINT_ID=YOUR_VERTEX_ENDPOINT_ID,GCP_ENDPOINT_IMAGE_MAX_SIZE=YOUR_MAX_SIZE,ALLOWED_ORIGINS=https://REPLACE_FIREBASE_PROJECT.web.app,https://REPLACE_FIREBASE_PROJECT.firebaseapp.com" `
  --set-secrets "ROBOFLOW_API_KEY=ROBOFLOW_API_KEY:latest"
```

Notes:

- `--source .` lets Cloud Build build the image from your `Dockerfile` — you don't need Docker Desktop.
- The PyTorch + OpenCV image is heavy; first build will take ~10 minutes. Subsequent deploys are faster.
- Memory `4Gi` is conservative for PyTorch CPU inference. If you hit OOM, bump to `8Gi`. If costs matter and your model is small, try `2Gi` first.
- `--concurrency 4` is sensible for CPU-bound inference; raise it later if you measure idle CPU.
- `--allow-unauthenticated` is required so the browser can call the API. The frontend calls it directly; per-user auth (Firebase Auth → ID token) can be layered later.

When it finishes, Cloud Run prints the **service URL**, e.g.:

```
Service URL: https://gingigauge-api-1234567890-uc.a.run.app
```

Keep that URL — you need it for the next step.

Quick smoke test:

```powershell
curl https://gingigauge-api-1234567890-uc.a.run.app/healthz
# {"status":"ok"}
```

---

## 5. Point the frontend at the Cloud Run URL

Edit `frontend/config.js` and set the URL you just got:

```js
window.API_BASE_URL = "https://gingigauge-api-1234567890-uc.a.run.app";
```

(Leave it as `""` if you also want to keep running locally with FastAPI serving the frontend.)

---

## 6. Set up Firebase Hosting (one-time)

If you've never used Firebase on this GCP project before:

1. Go to https://console.firebase.google.com/
2. Click **Add project** → choose your existing GCP project (Firebase will reuse it).
3. Skip Google Analytics if you don't need it.

Back in your terminal:

```powershell
firebase use --add
# pick your project, alias "default"
```

This refreshes `.firebaserc` with the right project ID.

---

## 7. Deploy the frontend

```powershell
firebase deploy --only hosting
```

Output ends with:

```
Hosting URL: https://your-project.web.app
```

Visit it and test the three flows: contact form, .bag prediction, GCS upload.

---

## 8. Lock CORS down (if you skipped it in step 4)

If your initial `gcloud run deploy` used `ALLOWED_ORIGINS=*`, tighten it now:

```powershell
gcloud run services update gingigauge-api `
  --region us-central1 `
  --update-env-vars "ALLOWED_ORIGINS=https://your-project.web.app,https://your-project.firebaseapp.com"
```

---

## 9. Operational notes

**Public read access on the GCS bucket.** `services/upload.py` returns
`https://storage.googleapis.com/<bucket>/<path>` as `image_url`. For that
URL to actually open in a browser, the bucket (or the object) needs to allow
public reads. Two options:

- **Public bucket** (simplest, fine for non-sensitive images):
  ```powershell
  gsutil iam ch allUsers:objectViewer gs://YOUR_BUCKET_NAME
  ```
- **Signed URLs** (recommended for patient-related data): change `upload.py`
  to call `blob.generate_signed_url(timedelta(hours=1))` instead of building
  the public URL string. Mention this and I can patch it.

**Firestore.** First time you write to Firestore on a fresh project you'll
be prompted to pick a database mode and location in the Firebase/GCP
console. Pick **Native mode** in the same region (`us-central1` or `nam5`).

**Logs.**
```powershell
gcloud run services logs tail gingigauge-api --region us-central1
```

**Rollback.**
```powershell
gcloud run services update-traffic gingigauge-api --region us-central1 --to-revisions=PREVIOUS_REVISION=100
```

**Cost guardrails.** With `--min-instances 0` you only pay when traffic
arrives. PyTorch cold starts will run 10–30 seconds — if that's a problem,
set `--min-instances 1`, but expect ~$30–$50/month per pinned instance at
2 vCPU / 4 GiB.

**Updating after code changes.**
- Backend: rerun the `gcloud run deploy` command from step 4.
- Frontend: rerun `firebase deploy --only hosting`.

---

## Quick deploy reference (after first-time setup)

```powershell
# Backend
gcloud run deploy gingigauge-api --source . --region us-central1

# Frontend
firebase deploy --only hosting
```
