// In dev, the frontend (Vite, :5173) and backend (uvicorn, :8000) are separate
// servers, so requests need the full backend URL. In production the built frontend
// is served by the same FastAPI process, so relative paths hit the right place
// automatically regardless of host/port.
const API_BASE = import.meta.env.DEV ? "http://localhost:8000" : "";

// The backend explains refusals in `detail` — rate limits, unreadable files,
// oversized uploads. Surfacing the bare status code instead turns an actionable
// message ("Rate limit reached, try again later") into "Prediction failed (429)".
async function explain(res, fallback) {
  let detail = null;
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body — fall through to the generic message */
  }
  if (detail) return new Error(detail);
  if (res.status === 429) return new Error("Too many requests. Wait a moment and try again.");
  if (res.status >= 500) return new Error("The server hit an error handling that image.");
  return new Error(`${fallback} (${res.status})`);
}

// A failed fetch (backend down, wrong port) rejects with a bare TypeError whose
// message is browser-specific and meaningless to a user.
function unreachable() {
  return new Error(
    "Can't reach the server. Make sure the backend is running on port 8000."
  );
}

export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  let res;
  try {
    res = await fetch(`${API_BASE}/predict`, { method: "POST", body: formData });
  } catch {
    throw unreachable();
  }

  if (!res.ok) throw await explain(res, "Analysis failed");
  return res.json(); // { scores: {stable, unstable}, assessment, sky_cropped, ... }
}


// Returns the image exactly as the model receives it, after sky/water cropping.
// The number in the result says how much was removed; this shows what remained.
export async function croppedImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  let res;
  try {
    res = await fetch(`${API_BASE}/cropped`, { method: "POST", body: formData });
  } catch {
    throw unreachable();
  }

  if (!res.ok) throw await explain(res, "Crop preview failed");
  return URL.createObjectURL(await res.blob());
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw await explain(res, "Stats failed");
  return res.json();
}

// Reports whether the models have finished warming. A cold backend loads CLIP and
// SegFormer lazily, so the first analysis can stall for tens of seconds; the UI
// uses this to say so rather than showing an unexplained spinner.
export async function fetchHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw await explain(res, "Health check failed");
  return res.json(); // { status, models_ready }
}
