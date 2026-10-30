# GroundTruth

[![checks](https://github.com/martini-i/GroundTruth/actions/workflows/checks.yml/badge.svg)](https://github.com/martini-i/GroundTruth/actions/workflows/checks.yml)

**Slope Surface Indicator Classifier** — a CSUF independent-study research project.

GroundTruth reads a ground-level photograph of a slope and reports whether visible
surface indicators suggest potential instability: cracks, scarps, loose debris,
disturbed soil, undercutting, rockfall.

> It does **not** predict landslides or assess subsurface conditions. Stability
> depends on water content, bedding and loading history, none of which show in a
> photo. This is a research prototype, not a safety assessment.

---

## What it does

A frozen CLIP backbone with a small calibrated probe trained on labelled slope
photos. Sky and water are cropped out first. Photos it can't assess (indoors, too
dark, no ground visible) are rejected instead of scored.

There is a web app and an Android client. Both use the same backend.

---

## Running it

```bash
python -m venv venv
venv/Scripts/pip install -r backend/requirements.txt
venv/Scripts/uvicorn backend.main:app --port 8000
```

Add `--host 0.0.0.0` to reach it from a phone on the same network.

Build the web app with `npm --prefix frontend install && npm --prefix frontend run
build`; FastAPI then serves it on the same port. For the Android client see
[`android/README.md`](android/README.md).

Both vision models download from Hugging Face on first run.

To reproduce the reported numbers, install the pinned environment instead:
`venv/Scripts/pip install -r requirements.lock.txt`.

---

## Where the numbers are

Performance, dataset size and thresholds change with the data, so they aren't
written here. Look them up instead:

| question | where the answer lives |
|---|---|
| how accurate is it | `metrics.json`, or `GET /stats` |
| how big is the dataset | `python verify_dataset.py` |
| what are the decision thresholds | `model_utils.py`, returned by every `POST /predict` |
| what does a script do, and what flags | `python <script>.py --help` |
| which images does it get wrong | `python audit_errors.py` |

`python frozen_features.py` regenerates `metrics.json`. `python train_probe.py`
refits the deployed model.

---

## The three invariants

Breaking any of these gives wrong numbers without an error.

**1. Crop symmetry.** Images in `slope_dataset/` are stored already cropped. Run
new images through `crop_dataset_sky.py` before training.

**2. Site grouping.** Photos of the same location must share a `site_id` in
`labels.csv`, so they never end up on both sides of a CV split.

**3. Feature cache keying.** The feature cache is keyed on the file list plus each
file's size and modification time, not the file count. Otherwise a stale cache can
silently line up old features with new labels.

`python verify_dataset.py` and `python -m pytest` check all three, and both run in
CI on every push.

---

## Known limitations

- **Surface only.** Two slopes that look the same can behave very differently
  underground.
- **Bare rock gets over-flagged.** Riprap, gabions, rock cuts and bedded cliffs are
  often marked unstable because rough rock texture looks like debris.
- **Scene shortcut.** The model tells engineered cuts from natural slopes more
  reliably than it tells failing ground from intact ground.
- **Indicator labels aren't blind.** Indicator categories were named after the
  stable/unstable label was set. New rows leave `primary_indicator` blank until it
  can be assigned without seeing the label.
- **Licensing gaps.** Some images have no recorded licence, so the dataset can't be
  released yet.

---

## Acknowledgements

Dataset images are drawn from field photography and Wikimedia Commons (CC BY-SA and
compatible licences), recorded per image in the `source` column of `labels.csv`.

Pretrained models: [CLIP](https://github.com/openai/CLIP) (OpenAI) and
[SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
(NVIDIA, ADE20K).
