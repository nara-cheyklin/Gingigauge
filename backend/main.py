from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes.predict import router as predict_router
from backend.routes.contact import router as contact_router
from backend.routes.upload import router as upload_router

app = FastAPI(title="Gingigauge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(contact_router)
app.include_router(upload_router)

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
