// Optional frontend runtime config, for when the browser client is served
// separately from the backend (e.g. a static host in front of a hosted API).
//
// Copy this file to web/config.js and set your own backend URL.
// web/config.js is gitignored.
//
// Example:
//   window.API_BASE_URL = "https://your-backend-host.example.com";
//
// Leave blank when the FastAPI server serves the client itself, which is the
// default for local development (uvicorn server.app:app).
window.API_BASE_URL = "";

// Only needed when the backend sets BACKEND_API_KEY. Sent as the X-API-Key
// header. You can also supply it at runtime via ?api_key=... and the client
// will persist it, which avoids committing the key here.
window.API_AUTH_KEY = "";
