// Gingigauge frontend runtime config.
//
// API_BASE_URL is the origin of the backend API.
//   - Local development (FastAPI also serves the frontend): leave as ""
//     so fetches stay same-origin.
//   - Production (Firebase Hosting + Cloud Run): set to the Cloud Run
//     service URL after `gcloud run deploy` prints it, e.g.
//     "https://gingigauge-api-XXXXXXXXXX-uc.a.run.app".
window.API_BASE_URL = "";
