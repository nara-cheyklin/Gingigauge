// Gingigauge frontend runtime config.
//
// API_BASE_URL is the origin of the backend API.
//   - Local development (FastAPI also serves the frontend): leave as ""
//     so fetches stay same-origin.
//   - Production (Firebase Hosting + Cloud Run): set to the Cloud Run
//     service URL printed by `gcloud run deploy`.
window.API_BASE_URL = "CLOUD_RUN_URL";
