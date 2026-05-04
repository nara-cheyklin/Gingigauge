import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.predict import router as predict_router
from backend.routes.contact import router as contact_router
from backend.routes.upload import router as upload_router

app = FastAPI(title="Gingigauge API")

# Allowed CORS origins are comma-separated in the ALLOWED_ORIGINS env var.
# Default "*" keeps local dev simple; in production set it to your Firebase
# Hosting domain(s), e.g.
#   ALLOWED_ORIGINS=https://gingigauge.web.app,https://gingigauge.firebaseapp.com
_allowed = os.getenv("ALLOWED_ORIGINS", "*").strip()
allowed_origins = ["*"] if _allowed == "*" else [o.strip() for o in _allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(contact_router)
app.include_router(upload_router)


@app.get("/healthz")
def healthz():
    """Lightweight liveness probe for Cloud Run."""
    return {"status": "ok"}


# Serve the frontend statically only when the folder is present alongside the
# backend (i.e. local development). The Cloud Run image excludes frontend/
# via .dockerignore, so this mount is automatically skipped in production.
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
