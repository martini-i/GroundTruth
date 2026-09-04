"""
model_utils.py — Shared model loading, prediction, and explanation logic.
Used by backend/main.py (FastAPI) and the offline analysis scripts.

The deployed classifier is a linear probe on frozen CLIP image features, not a
fine-tuned CNN. Grouped 5-fold cross-validation over 10 seeds put the probe at 81.0% +/- 1.1
vs 66.5% +/- 9.6 for the fine-tuned ResNet18 (FINDINGS.md section 5) while
training 513 parameters instead of ~8.4M. The current figure lives in
metrics.json; this docstring is prose and can lag. Fit it with train_probe.py, which
writes slope_probe.joblib.
"""

import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

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

# Longest edge of a returned heatmap overlay, in pixels.
GRADCAM_MAX_EDGE = 900

# Occlusion attribution. The grid is sample points across the 224px model input;
# the window is the occluder size, larger than the spacing so patches overlap and
# the map is smooth rather than blocky. 14x14 costs 196 forward passes, ~5s
# batched on CPU — slower than a gradient pass, and worth it (see _attribution_map).
OCCLUSION_GRID = 14
OCCLUSION_WINDOW = 32
OCCLUSION_BATCH = 32
# Rendered range, as percentiles of the attribution map.
OCCLUSION_CLIP_LOW = 85
OCCLUSION_CLIP_HIGH = 99

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


def crop_sky(image: Image.Image) -> Image.Image:
    """
    Crops out the sky using a pretrained ADE20K segmentation model, so the
    classifier and the heatmap focus on the hillside itself rather than
    sky/clouds. Falls back to the original image if little/no sky is
    detected (e.g. close-up crops).

    Applied identically when fitting (crop_dataset_sky.py, train_probe.py) and
    at inference time below — cropping only at inference would show the model
    framing very different from what it was fitted on.
    """
    processor, seg_model = _load_segmentation_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    upsampled = nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy()
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


def _clip_embedding(image: Image.Image, grad: bool = False):
    """
    CLIP image embedding, optionally keeping the graph for attribution.

    When grad is set, the tensor returned alongside the embedding is the
    *second to last* hidden state, not the final one. The pooled output reads
    only the CLS token of the final layer, so final-layer patch tokens are a
    dead end — their gradient is exactly zero and any heatmap built from them
    is noise. One layer earlier, patch tokens still reach CLS through the last
    block attention and carry real gradient.
    """
    load_model()
    inputs = _clip_proc(images=image, return_tensors="pt").to(_device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        vision_out = _clip.vision_model(
            pixel_values=inputs["pixel_values"], output_hidden_states=grad
        )
        hidden = None
        if grad:
            hidden = vision_out.hidden_states[-2]
            hidden.retain_grad()
        emb = _clip.visual_projection(vision_out.pooler_output)
    return emb, hidden


def _unstable_logit(emb: torch.Tensor) -> torch.Tensor:
    """Reproduce the fitted probe decision function differentiably in torch."""
    scaler, clf = _probe["scaler"], _probe["clf"]
    mean = torch.tensor(scaler.mean_, dtype=emb.dtype, device=emb.device)
    scale = torch.tensor(scaler.scale_, dtype=emb.dtype, device=emb.device)
    coef = torch.tensor(clf.coef_[0], dtype=emb.dtype, device=emb.device)
    intercept = float(clf.intercept_[0])
    return ((emb - mean) / scale * coef).sum() + intercept


def predict(image: Image.Image) -> dict:
    """
    Takes a PIL image, returns {"scores": {class: prob, ...}, "assessment": str,
    "sky_cropped": float, "unstable_threshold": float}. sky_cropped is the
    fraction of image height removed by crop_sky — surfaced so callers can tell
    the user the model saw a cropped view rather than the photo they uploaded.
    """
    load_model()
    class_names = _probe["class_names"]

    image = image.convert("RGB")
    cropped = crop_sky(image)
    # Area, not height: crop_sky trims sides as well as the top, so a
    # height-only figure understated how much was removed.
    kept = (cropped.width * cropped.height) / max(image.width * image.height, 1)
    sky_cropped = max(0.0, 1.0 - kept)

    emb, _ = _clip_embedding(cropped)
    # Calibrated estimator, not the raw one: uncalibrated logistic regression on
    # separable CLIP features reports ~93% mean confidence at ~82% accuracy.
    probs = _probe["calibrated"].predict_proba(emb.cpu().numpy())[0]

    scores = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    assessment = assess(scores.get("unstable", 0.0))

    return {
        "scores": scores,
        "assessment": assessment,
        "sky_cropped": round(sky_cropped, 3),
        "unstable_threshold": UNSTABLE_THRESHOLD,
        # Both cutoffs, so a caller can distinguish the three bands the
        # assessment actually has. Without the upper one the interface has to
        # infer "borderline" by matching on the prose, which silently breaks the
        # moment the wording is edited.
        "unstable_high_confidence": UNSTABLE_HIGH_CONFIDENCE,
    }


def _probe_logits(pixel_values: torch.Tensor) -> torch.Tensor:
    """Unstable logit for a batch of already-preprocessed images."""
    with torch.no_grad():
        vision_out = _clip.vision_model(pixel_values=pixel_values)
        emb = _clip.visual_projection(vision_out.pooler_output)
        scaler, clf = _probe["scaler"], _probe["clf"]
        mean = torch.tensor(scaler.mean_, dtype=emb.dtype, device=emb.device)
        scale = torch.tensor(scaler.scale_, dtype=emb.dtype, device=emb.device)
        coef = torch.tensor(clf.coef_[0], dtype=emb.dtype, device=emb.device)
        return ((emb - mean) / scale * coef).sum(dim=1) + float(clf.intercept_[0])


def _attribution_map(image: Image.Image) -> np.ndarray:
    """
    Occlusion attribution at the size of `image`, in [0, 1].

    Hide a region, measure how far the unstable logit actually falls. That is
    the question a reader is asking of the heatmap — "what is it reacting to?" —
    answered by measurement rather than by a proxy, so it is faithful by
    construction.

    This replaces a gradient-times-token map that was blended across nine
    overlapping crops. Two things were wrong with it. Gradient-times-activation
    on transformer tokens is a noisy attribution to begin with, and worse, each
    crop was a *separate forward pass on a different image*: a tile answers
    "within this crop, what pushed toward unstable", which is not the same
    question as the whole photo, under different context and normalisation.
    Blending nine incommensurable answers produced disconnected blobs that did
    not correspond to anything in the terrain.

    Occlusion is done on the preprocessed tensor, not the source photo, so the
    image is resized and normalised once and the variants are a batched forward
    pass. Zero in normalised space is the dataset mean colour — the standard
    uninformative occluder.
    """
    load_model()

    # Reflect-pad before occluding, then discard the padded margin from the
    # result. Occluding a cell that sits on the true frame edge produces a large
    # spurious response — measured directly: on stable_cut_007 the border ring
    # averaged +1.34 against -0.07 for the interior, and padding the image so
    # that same content moved inward collapsed it to +0.01. The rim was tracking
    # the frame, not the photograph, and the percentile clip then amplified it
    # into a bright halo that read as "the model is looking at the edges".
    pad_frac = 1.0 / OCCLUSION_GRID
    pw, ph = int(image.width * pad_frac), int(image.height * pad_frac)
    padded = Image.new("RGB", (image.width + 2 * pw, image.height + 2 * ph))
    padded.paste(image, (pw, ph))
    # Mirror the photo into the margin so the padding has plausible local
    # statistics instead of a hard synthetic edge of its own.
    padded.paste(image.transpose(Image.FLIP_LEFT_RIGHT).crop((image.width - pw, 0, image.width, image.height)), (0, ph))
    padded.paste(image.transpose(Image.FLIP_LEFT_RIGHT).crop((0, 0, pw, image.height)), (image.width + pw, ph))
    full = padded.crop((0, ph, padded.width, ph + image.height))
    padded.paste(full.transpose(Image.FLIP_TOP_BOTTOM).crop((0, image.height - ph, padded.width, image.height)), (0, 0))
    padded.paste(full.transpose(Image.FLIP_TOP_BOTTOM).crop((0, 0, padded.width, ph)), (0, image.height + ph))

    inputs = _clip_proc(images=padded, return_tensors="pt").to(_device)
    pixel_values = inputs["pixel_values"]

    baseline = _probe_logits(pixel_values).item()
    side = pixel_values.shape[-1]
    step = side // OCCLUSION_GRID
    half = OCCLUSION_WINDOW // 2

    variants = []
    for gy in range(OCCLUSION_GRID):
        for gx in range(OCCLUSION_GRID):
            cy, cx = gy * step + step // 2, gx * step + step // 2
            v = pixel_values.clone()
            v[:, :, max(cy - half, 0):min(cy + half, side),
                    max(cx - half, 0):min(cx + half, side)] = 0.0
            variants.append(v)

    batch = torch.cat(variants, dim=0)
    outs = [_probe_logits(batch[i:i + OCCLUSION_BATCH])
            for i in range(0, batch.shape[0], OCCLUSION_BATCH)]
    occluded = torch.cat(outs).cpu().numpy()

    # Positive = removing that region cost the model confidence, i.e. it was
    # evidence for "unstable".
    grid = (baseline - occluded).reshape(OCCLUSION_GRID, OCCLUSION_GRID)

    # Drop the outermost ring: it covers the reflected margin, not the photo.
    grid = grid[1:-1, 1:-1]

    w, h = image.size
    cam = np.asarray(
        Image.fromarray(grid.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC)
    )

    # Percentile clip rather than min-max. On a confident prediction almost every
    # occlusion lowers the logit, so min-max renders the whole frame hot and hides
    # the structure. Showing the top band surfaces the regions actually carrying
    # the decision.
    lo = np.percentile(cam, OCCLUSION_CLIP_LOW)
    hi = np.percentile(cam, OCCLUSION_CLIP_HIGH)
    return np.clip((cam - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def gradcam_overlay(image: Image.Image) -> Image.Image:
    """
    Takes a PIL image, returns it with an attention heatmap overlaid showing
    which regions pushed the prediction toward "unstable".

    Not Grad-CAM, and not a gradient method at all. Grad-CAM needs a
    convolutional feature map to weight by channel gradient; the CLIP vision
    tower is a transformer and has none. The attribution here is occlusion —
    hide a region and measure how far the unstable logit actually falls. See
    _attribution_map for why the earlier gradient-based version was replaced.

    The name is kept because it is the public API the frontend and the Android
    client both call.
    """
    load_model()
    image = image.convert("RGB")
    cropped = crop_sky(image)

    # Cap the long edge first: attribution runs several passes, and there is no
    # gain from computing them at phone-camera resolution.
    display = cropped
    if max(display.size) > GRADCAM_MAX_EDGE:
        s = GRADCAM_MAX_EDGE / max(display.size)
        display = display.resize(
            (max(1, round(display.width * s)), max(1, round(display.height * s))),
            Image.LANCZOS,
        )

    cam = _attribution_map(display)
    rgb_img = np.asarray(display).astype(np.float32) / 255.0

    return Image.fromarray(show_cam_on_image(rgb_img, cam.astype(np.float32), use_rgb=True))
