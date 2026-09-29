"""
model_utils.py — Shared model loading, preprocessing and prediction logic.
Used by backend/main.py (FastAPI) and the offline analysis scripts.

The deployed classifier is a calibrated linear probe on frozen CLIP image
features, not a fine-tuned CNN: grouped cross-validation put the probe well
ahead of a fine-tuned ResNet18 while training three orders of magnitude fewer
parameters. The measured figures live in metrics.json, written by
frozen_features.py — never repeated in prose, which goes stale silently.

Fit the probe with train_probe.py, which writes slope_probe.joblib.
"""

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

PROBE_PATH = "slope_probe.joblib"

# Decision threshold is intentionally lower than 0.5: missing a genuinely unstable
# slope (false negative) is a costlier error than a false alarm, so "Potentially
# Unstable" is flagged starting at P(unstable) >= 0.35 rather than waiting for it
# to be the argmax.
UNSTABLE_THRESHOLD = 0.35
UNSTABLE_HIGH_CONFIDENCE = 0.65

# ADE20K classes treated as background by crop_sky(). Sky is the main offender,
# but open water is equally uninformative for slope assessment and, on coastal
# photographs, occupies the side of the frame rather than the top — so a
# sky-only rule cannot trim it.
BACKGROUND_CLASS_INDICES = (2, 21, 26, 60, 128)   # sky, water, sea, river, lake
SKY_PIXEL_MIN_FRACTION = 0.03   # below this, treat as no real background band present
SKY_CROP_MAX_FRACTION = 0.55    # never crop away more than this much of either axis
CONTENT_ROW_MIN_FRACTION = 0.10  # a row/column counts as content above this share

# --- input gating ------------------------------------------------------------
# A linear probe on CLIP features answers every image it is handed. Given a
# carpet, a ceiling or a selfie it returns a calibrated-looking probability,
# because nothing in the pipeline represents "not applicable". These checks run
# BEFORE the probe and refuse the image instead.
#
# They read class proportions off the coarse segmentation map, which comes from the
# same forward pass crop_sky needs — so gating costs no extra network call and no
# full-resolution upsample.
#
# Every threshold is deliberately permissive. A false refusal on a real slope is
# worse than letting an odd photograph through, and the gate must pass all of
# slope_dataset/ — verify that with `python check_gate.py` after any change here.

# Positive evidence of an interior, rather than absence of evidence of terrain.
# That distinction matters: a close-up of a shotcrete face segments almost
# entirely as "wall", so a rule that required visible soil or rock would refuse
# exactly the engineered slopes this project cares most about.
INDOOR_CLASS_INDICES = (3, 5, 7, 10, 15, 18, 19, 23, 24, 28, 30, 33, 35, 36, 37,
                        39, 44, 45, 47, 50, 56, 57, 62, 64, 65, 70, 71, 73, 74,
                        75, 89, 107, 110, 118, 124, 129, 130, 131, 134, 145, 146)
# Natural ground and vegetation. Weak positive evidence only — see above.
TERRAIN_CLASS_INDICES = (4, 9, 13, 16, 17, 29, 34, 46, 52, 68, 72, 91, 94)
PERSON_CLASS_INDEX = 12

# Every number below is set from what the curated dataset actually measures, with
# margin, not from taste. The figures in brackets are the dataset's own extreme
# for that statistic, printed by check_gate.py — re-run it after any change here
# and after adding images, because the margins move as the dataset grows.
INDOOR_MAX_FRACTION = 0.35     # dataset max 0.171 — refuse well clear of it
INDOOR_HINT_FRACTION = 0.25    # weaker corroborating signal; still above 0.171
PERSON_MAX_FRACTION = 0.40     # dataset max 0.125
TERRAIN_MIN_FRACTION = 0.02    # dataset min 0.037 — bare engineered faces run low
VISIBLE_MIN_FRACTION = 0.10    # dataset min 0.654
MIN_EDGE_PIXELS = 64           # dataset min short edge 210
MAX_ASPECT_RATIO = 5.0         # dataset max 4.08 (stable_slope_003, a wide pan)
MIN_MEAN_LUMINANCE = 18.0      # dataset min 65.3
MIN_LUMINANCE_STD = 10.0       # dataset min 24.8


_device = None
_clip = None
_clip_proc = None
_probe = None
_seg_processor = None
_seg_model = None


def load_model():
    """Load CLIP and the fitted probe once and cache them. Safe to call repeatedly."""
    global _device, _clip, _clip_proc, _probe
    if _clip is not None:
        return _clip, _probe["class_names"], _device

    from transformers import AutoImageProcessor, CLIPModel

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _probe = joblib.load(PROBE_PATH)
    _clip_proc = AutoImageProcessor.from_pretrained(_probe["clip_id"])
    _clip = CLIPModel.from_pretrained(_probe["clip_id"]).eval().to(_device)
    return _clip, _probe["class_names"], _device


def _load_segmentation_model():
    """Load the pretrained sky-segmentation model once and cache it."""
    global _seg_processor, _seg_model
    if _seg_model is not None:
        return _seg_processor, _seg_model

    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    _seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    _seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    _seg_model.eval()
    return _seg_processor, _seg_model


def segmentation_logits(image: Image.Image) -> torch.Tensor:
    """
    One SegFormer forward pass, left at the model's own output resolution.

    Both label-map helpers below derive from this, so a caller needing a coarse
    view and a full-resolution view pays for the network once.
    """
    processor, seg_model = _load_segmentation_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        return seg_model(**inputs).logits


def segment_coarse(logits: torch.Tensor) -> np.ndarray:
    """
    Label map at SegFormer's own output resolution, for callers that only need
    class proportions.

    argmax first, no upsampling. The full-resolution path below interpolates
    150 channels of float32 up to the source image before reducing them: on a
    43 MP photograph that is a single ~26 GB tensor, measured at 28 seconds for
    one image. Proportions are scale-invariant, so the input gate reads them
    from here and costs effectively nothing.
    """
    return logits.argmax(dim=1)[0].numpy()


def segment(image: Image.Image, logits: torch.Tensor | None = None) -> np.ndarray:
    """
    ADE20K class index per pixel, upsampled to the image's own resolution.

    Interpolate-then-argmax, deliberately: that is the order crop_sky has always
    used and every image in slope_dataset/ was cropped with it. Reversing the
    two is cheaper but shifts boundaries by a pixel or so, which would put
    stored crops and inference crops on different rules — invariant 1. Change it
    only alongside `crop_dataset_sky.py --all` and a refit.
    """
    if logits is None:
        logits = segmentation_logits(image)
    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].numpy()


def crop_sky(image: Image.Image, pred: np.ndarray | None = None) -> Image.Image:
    """
    Crops out the sky using a pretrained ADE20K segmentation model, so the
    classifier focuses on the hillside itself rather than
    sky/clouds. Falls back to the original image if little/no sky is
    detected (e.g. close-up crops).

    Applied identically when fitting (crop_dataset_sky.py, train_probe.py) and
    at inference time below — cropping only at inference would show the model
    framing very different from what it was fitted on.

    `pred` lets a caller pass in a segmentation it has already computed; omitted,
    one is computed here, so existing callers are unaffected.
    """
    if pred is None:
        pred = segment(image)
    background = np.isin(pred, BACKGROUND_CLASS_INDICES)

    if background.mean() < SKY_PIXEL_MIN_FRACTION:
        return image

    content = ~background
    h, w = content.shape
    if content.mean() < 0.15:          # almost nothing but sky/water: leave alone
        return image

    # Trim inward from each edge while that row/column is overwhelmingly
    # background. An earlier version cropped a top band at the median sky depth
    # taken over sky-containing columns only — on a photo where a cliff fills
    # one side top-to-bottom, those columns are excluded entirely, so the median
    # reflected the open-horizon side and cut away most of the subject.
    rows = content.mean(axis=1)
    cols = content.mean(axis=0)
    row_idx = np.where(rows >= CONTENT_ROW_MIN_FRACTION)[0]
    col_idx = np.where(cols >= CONTENT_ROW_MIN_FRACTION)[0]
    if row_idx.size == 0 or col_idx.size == 0:
        return image

    top, bottom = int(row_idx[0]), int(row_idx[-1]) + 1
    left, right = int(col_idx[0]), int(col_idx[-1]) + 1

    # Caps, applied per edge, so a segmentation mistake can never remove most of
    # the photo. Bottom/right are trimmed only slightly: slope subjects sit low
    # in frame far more often than high.
    top = min(top, int(h * SKY_CROP_MAX_FRACTION))
    bottom = max(bottom, int(h * 0.85))
    left = min(left, int(w * SKY_CROP_MAX_FRACTION))
    right = max(right, int(w * 0.45))

    if bottom - top < 32 or right - left < 32:
        return image
    if top == 0 and left == 0 and bottom >= h and right >= w:
        return image

    return image.crop((left, top, right, bottom))


def assess(unstable_prob: float) -> str:
    """Turn a raw P(unstable) into the actual (threshold-adjusted) call."""
    if unstable_prob >= UNSTABLE_HIGH_CONFIDENCE:
        return "Potentially Unstable — visible surface indicators present (e.g. cracks, scarps, loose debris, disturbed soil)."
    elif unstable_prob >= UNSTABLE_THRESHOLD:
        return "Potentially Unstable (borderline) — some possible indicators present. Field inspection recommended."
    else:
        return "Stable — no significant visible surface indicators detected."


def check_assessable(image: Image.Image, pred: np.ndarray | None = None) -> dict:
    """
    Decide whether this photograph is one the classifier can speak to at all.

    Returns {"ok": True, "stats": {...}} or
            {"ok": False, "reason": str, "message": str, "stats": {...}}.

    The probe cannot abstain on its own. It projects any image into CLIP space
    and reports where it falls relative to one learned direction, so a photo of
    a kitchen floor gets a confident, well-calibrated-looking answer to a
    question that was never asked of it. Refusing is the only honest output
    there, and a tool that never refuses cannot be trusted in the field.

    Two families of check:

    * **Image quality** — size, aspect, brightness, contrast. Read from the
      photograph as taken, because that is what the user controls.
    * **Scene content** — computed over the *visible* region only, i.e. after
      sky and open water are excluded, since that is the part the classifier is
      shown. An all-sky frame has no visible region at all and is refused.

    `stats` is returned on success as well as failure. It costs nothing, and a
    caller that logs it can later ask how close real traffic runs to each
    threshold — which is the only way these numbers get tuned on evidence
    rather than taste.
    """
    stats: dict = {}

    w, h = image.size
    stats["width"], stats["height"] = w, h
    if min(w, h) < MIN_EDGE_PIXELS:
        return {"ok": False, "reason": "too_small", "stats": stats,
                "message": f"That image is only {w}x{h}. Slope indicators are not "
                           "legible below about 64 pixels on the short edge."}

    aspect = max(w, h) / max(min(w, h), 1)
    stats["aspect_ratio"] = round(aspect, 2)
    if aspect > MAX_ASPECT_RATIO:
        return {"ok": False, "reason": "extreme_aspect", "stats": stats,
                "message": "That looks like a panorama or a cropped strip. Use a "
                           "single frame showing the slope face."}

    grey = np.asarray(image.convert("L"), dtype=np.float32)
    stats["mean_luminance"] = round(float(grey.mean()), 1)
    stats["luminance_std"] = round(float(grey.std()), 1)
    if grey.mean() < MIN_MEAN_LUMINANCE:
        return {"ok": False, "reason": "too_dark", "stats": stats,
                "message": "That photo is too dark to read. Surface indicators need "
                           "daylight or direct illumination."}
    if grey.std() < MIN_LUMINANCE_STD:
        return {"ok": False, "reason": "no_detail", "stats": stats,
                "message": "That frame is almost uniform — fog, a blank surface, or a "
                           "blown-out exposure. Nothing in it can be assessed."}

    if pred is None:
        pred = segment_coarse(segmentation_logits(image))

    # Resolution-agnostic by construction: every test below is a proportion, so
    # a caller may pass the coarse map or the full-resolution one.
    visible = ~np.isin(pred, BACKGROUND_CLASS_INDICES)
    visible_frac = float(visible.mean())
    stats["visible_fraction"] = round(visible_frac, 3)
    if visible_frac < VISIBLE_MIN_FRACTION:
        return {"ok": False, "reason": "no_ground", "stats": stats,
                "message": "Almost the entire frame is sky or open water. Point the "
                           "camera at the slope face."}

    n_visible = max(int(visible.sum()), 1)
    seen = pred[visible]
    indoor_frac = float(np.isin(seen, INDOOR_CLASS_INDICES).sum()) / n_visible
    terrain_frac = float(np.isin(seen, TERRAIN_CLASS_INDICES).sum()) / n_visible
    person_frac = float((seen == PERSON_CLASS_INDEX).sum()) / n_visible
    stats["indoor_fraction"] = round(indoor_frac, 3)
    stats["terrain_fraction"] = round(terrain_frac, 3)
    stats["person_fraction"] = round(person_frac, 3)

    if indoor_frac > INDOOR_MAX_FRACTION:
        return {"ok": False, "reason": "indoor_scene", "stats": stats,
                "message": "That looks like an indoor scene. GroundTruth reads "
                           "outdoor slope faces."}
    if person_frac > PERSON_MAX_FRACTION:
        return {"ok": False, "reason": "subject_is_person", "stats": stats,
                "message": "A person fills most of that frame. Photograph the slope "
                           "itself, with people only incidentally in shot."}
    if terrain_frac < TERRAIN_MIN_FRACTION and indoor_frac > INDOOR_HINT_FRACTION:
        # Two weak signals agreeing: no natural surface, and enough indoor
        # evidence to corroborate. Both are required. Segmentation noise puts a
        # stray indoor pixel in almost any frame, so testing indoor_frac > 0
        # would make this fire on every close-up of a bare engineered face —
        # images that show no terrain and no furniture either, and that this
        # project exists to assess.
        return {"ok": False, "reason": "no_slope_detected", "stats": stats,
                "message": "No slope or ground surface is recognisable in that image."}

    return {"ok": True, "stats": stats}


def _clip_embedding(image: Image.Image) -> torch.Tensor:
    """CLIP image embedding for one PIL image, in the 512-d projected space."""
    load_model()
    inputs = _clip_proc(images=image, return_tensors="pt").to(_device)
    with torch.no_grad():
        vision_out = _clip.vision_model(pixel_values=inputs["pixel_values"])
        return _clip.visual_projection(vision_out.pooler_output)


def predict(image: Image.Image) -> dict:
    """
    Takes a PIL image, returns {"assessable": True, "scores": {class: prob, ...},
    "assessment": str, "sky_cropped": float, "unstable_threshold": float}.
    sky_cropped is the fraction of image *area* removed by crop_sky — surfaced so
    callers can tell the user the model saw a cropped view rather than the photo
    they uploaded.

    If the image is not one the classifier can speak to, returns
    {"assessable": False, "reason": str, "message": str, "input_checks": {...}}
    and no scores. Callers must branch on "assessable" before reading "scores";
    see check_assessable for why refusing is a first-class outcome rather than an
    error.
    """
    load_model()
    class_names = _probe["class_names"]

    image = image.convert("RGB")

    # One network pass, two views of it: the gate reads proportions off the
    # coarse map, crop_sky needs the full-resolution one for its bounding box.
    logits = segmentation_logits(image)

    gate = check_assessable(image, segment_coarse(logits))
    if not gate["ok"]:
        return {
            "assessable": False,
            "reason": gate["reason"],
            "message": gate["message"],
            "input_checks": gate["stats"],
        }

    cropped = crop_sky(image, segment(image, logits))
    # Area, not height: crop_sky trims sides as well as the top, so a
    # height-only figure understated how much was removed.
    kept = (cropped.width * cropped.height) / max(image.width * image.height, 1)
    sky_cropped = max(0.0, 1.0 - kept)

    emb = _clip_embedding(cropped)
    # Calibrated estimator, not the raw one: uncalibrated logistic regression on
    # separable CLIP features is badly overconfident — it reports near-certainty
    # far more often than it is right. See train_probe.py.
    probs = _probe["calibrated"].predict_proba(emb.cpu().numpy())[0]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    assessment = assess(scores.get("unstable", 0.0))

    return {
        "assessable": True,
        "scores": scores,
        "assessment": assessment,
        "sky_cropped": round(sky_cropped, 3),
        # What the gate measured on the way through. Logging this is how the
        # thresholds eventually get tuned on real traffic instead of taste.
        "input_checks": gate["stats"],
        "unstable_threshold": UNSTABLE_THRESHOLD,
        # Both cutoffs, so a caller can distinguish the three bands the
        # assessment actually has. Without the upper one the interface has to
        # infer "borderline" by matching on the prose, which silently breaks the
        # moment the wording is edited.
        "unstable_high_confidence": UNSTABLE_HIGH_CONFIDENCE,
    }
