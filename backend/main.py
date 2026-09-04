"""
backend/main.py — FastAPI backend for GroundTruth, wrapping model_utils.py for the React frontend.
Run from the project root with: uvicorn backend.main:app --reload --port 8000
"""

import io
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils
import db

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

app = FastAPI(title="GroundTruth API")

# --- public-app limits -------------------------------------------------------
# This is a research demo on one box, not a service with a capacity plan. The
# caps exist so a single caller cannot monopolise it: /predict costs a CLIP
# forward pass. (The attribution heatmap was removed from the app — it cost
# ~2.5s of CPU per image and mostly showed the model attending away from the
# slope, which is a real property of the model but not a useful public feature.
# model_utils.gradcam_overlay and gradcam.py remain for offline diagnosis.)
RATE_LIMITS = {
    "predict": (40, 10),      # 40 analyses per 10 minutes per client
}

# /monitoring reports usage patterns, so it needs a shared secret. Unset means
# the endpoint is disabled rather than open — an admin route that silently
# defaults to public is how internal data leaks.
ADMIN_TOKEN = os.environ.get("GROUNDTRUTH_ADMIN_TOKEN")


def _client_fp(request: Request) -> str | None:
    return db.client_fingerprint(request.client.host if request.client else None)


def _enforce_rate_limit(request: Request, kind: str) -> str | None:
    limit, minutes = RATE_LIMITS[kind]
    fp = _client_fp(request)
    try:
        used = db.recent_count(fp, "predictions", minutes)
    except Exception:
        return fp        # a broken limiter must not take the app down
    if used >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit reached ({limit} per {minutes} minutes). Try again later.",
        )
    return fp


def _require_admin(token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Review API is disabled. Set GROUNDTRUTH_ADMIN_TOKEN to enable it.",
        )
    if not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_warm = {"ready": False}


@app.on_event("startup")
def _warmup():
    """
    Load CLIP, the probe and the sky segmenter in the background at startup.

    Both models are loaded lazily on first use, which made the first prediction
    after a cold start take tens of seconds while the user stared at a spinner.
    Warming in a daemon thread keeps startup itself non-blocking, so the server
    still answers /health immediately.
    """
    import threading

    def run():
        try:
            model_utils.load_model()
            model_utils._load_segmentation_model()
            _warm["ready"] = True
        except Exception as exc:            # never let warmup kill the server
            _warm["error"] = str(exc)

    threading.Thread(target=run, daemon=True).start()


def _load_upload_image(file: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(file.file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


@app.get("/health")
def health():
    # models_ready lets a caller tell "server up" from "server up and able to
    # answer a prediction without a long first-request stall".
    return {"status": "ok", "models_ready": _warm["ready"]}


@app.get("/stats")
def stats():
    """
    Dataset and model facts for the About page. Read from labels.csv at request
    time rather than hardcoded in the frontend, so the figures cannot drift out
    of date as the dataset changes.
    """
    import csv
    from collections import Counter

    root = Path(__file__).resolve().parent.parent
    counts, indicators = Counter(), Counter()
    try:
        with open(root / "labels.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                counts[row["label"]] += 1
                if row.get("primary_indicator"):
                    indicators[row["primary_indicator"]] += 1
    except FileNotFoundError:
        pass

    probe_info = {}
    try:
        model_utils.load_model()
        probe_info = {
            "backbone": model_utils._probe.get("clip_id"),
            "fitted_on": model_utils._probe.get("n_train"),
        }
    except Exception:
        pass

    # Measured accuracy, written by frozen_features.py. Served rather than
    # hardcoded in the frontend so the public figure cannot drift from the
    # measurement.
    metrics = {}
    try:
        import json
        metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    return {
        "metrics": metrics,
        "dataset": {
            "total": sum(counts.values()),
            "stable": counts.get("stable", 0),
            "unstable": counts.get("unstable", 0),
            "top_indicators": indicators.most_common(8),
        },
        "model": probe_info,
        "thresholds": {
            "flag": model_utils.UNSTABLE_THRESHOLD,
            "high_confidence": model_utils.UNSTABLE_HIGH_CONFIDENCE,
        },
    }


@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)):
    fp = _enforce_rate_limit(request, "predict")
    image = _load_upload_image(file)
    result = model_utils.predict(image)
    try:
        db.log_prediction(
            p_unstable=result["scores"].get("unstable", 0.0),
            assessment=result["assessment"],
            sky_cropped=result.get("sky_cropped", 0.0),
            width=image.width, height=image.height,
            client_fp=fp,
        )
    except Exception:
        pass        # logging must never break a prediction
    return result


@app.get("/monitoring")
def monitoring(x_admin_token: str | None = Header(default=None)):
    """
    Confidence distribution of served predictions.

    This cannot measure accuracy — nothing here knows the true label. It exists
    to catch the deployed probe drifting back toward the overconfidence that
    calibration corrected, which shows up without labels as most of the mass
    piling into the outermost bins.
    """
    _require_admin(x_admin_token)
    return {"served": db.counts(), "confidence_bins": db.calibration_summary()}


@app.post("/cropped")
def cropped(request: Request, file: UploadFile = File(...)):
    """
    The image exactly as the classifier receives it, after sky/water removal.

    /predict reports how much area the crop removed, which tells a user a number
    but not what it did. Cropping is the preprocessing step most likely to go
    wrong on an unusual photo — a hazy skyline, a water-filled cut — and it is
    invisible in the result. This makes it inspectable.
    """
    _enforce_rate_limit(request, "predict")
    image = _load_upload_image(file)
    out = model_utils.crop_sky(image.convert("RGB"))

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


# In production the React app is pre-built into frontend/dist and served from
# this same FastAPI process, so the whole app is one origin on one port. This
# mount must stay last so it doesn't shadow the API routes above; it only
# activates if the built frontend is actually present (i.e. not during local
# `uvicorn --reload` dev use).
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
