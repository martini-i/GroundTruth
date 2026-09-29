# GroundTruth

**Slope Surface Indicator Classifier** — a CSUF independent-study research project.

GroundTruth reads a ground-level photograph of a slope and reports whether visible
surface indicators suggest potential instability: cracks, scarps, loose debris,
disturbed soil, undercutting, rockfall.

> It does **not** predict landslides and does **not** assess subsurface conditions.
> Slope stability depends on water content, bedding orientation and loading history,
> none of which appear in a photograph. This is a research prototype, not a safety
> determination.

That distinction is load-bearing throughout the codebase, not a disclaimer bolted on
the front. The wording is deliberate wherever it appears.

---

## What it does

A frozen CLIP backbone with a small calibrated probe fitted on labelled slope
photographs. Sky and open water are removed first, and images the model cannot speak
to — indoors, too dark, no ground in frame — are refused rather than scored.

Available as a web app and an Android client. Both call the same backend, so both
give the same answer for the same photograph.

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

Both vision models download from Hugging Face on first run, so budget for that on a
cold start.

---

## Where the numbers are

Nothing about performance, dataset size or thresholds is written down in this file,
because all of it goes stale the moment the data or the code changes. Ask the
project instead:

| question | where the answer lives |
|---|---|
| how accurate is it | `metrics.json`, or `GET /stats` |
| how big is the dataset | `python verify_dataset.py` |
| what are the decision thresholds | `model_utils.py`, returned by every `POST /predict` |
| what does a script do, and what flags | `python <script>.py --help` |
| which images does it get wrong | `python audit_errors.py` |

Regenerate the measurements with `python frozen_features.py`, which rewrites
`metrics.json`. Refit the deployed model with `python train_probe.py`.

---

## The three invariants

Each of these has been violated at least once, and each failure was silent — no
error, just quietly wrong numbers. They are the only rules in this project worth
memorising.

**1. Crop symmetry.** Images in `slope_dataset/` are stored already cropped. Any new
image must be run through `crop_dataset_sky.py` before training, or it is framed
differently from every image the probe was fitted on.

**2. Site grouping.** Photographs of the same location must share a `site_id` in
`labels.csv`. Without it, near-duplicate images land on both sides of an evaluation
split and the result measures memory rather than generalisation.

**3. Feature cache keying.** Cached features are keyed on the file list plus each
file's size and modification time — never on the number of files. A count-keyed
cache returns feature rows belonging to a previous set of images, which still line
up positionally against the new labels.

Run `python verify_dataset.py` before trusting any number. It catches what it can.

---

## Known limitations

**It cannot see underground.** The most important one. Two slopes can look identical
and behave completely differently.

**The model exploits a semantic shortcut.** It separates engineered cuttings from
natural cliffs more reliably than it separates failing ground from intact ground.
The counterexamples that would break it are unstable engineered cuts and stable
natural cliffs.

**The indicator taxonomy is not independent of the label.** Categories were named
after the stable/unstable call had been made, so none straddles the boundary. Fixing
it needs indicator labels assigned blind to the class.

**Some dataset images lack licence provenance.** Unresolved, and it blocks any
release of the dataset.

---

## Acknowledgements

Dataset images are drawn from field photography and Wikimedia Commons (CC BY-SA and
compatible licences), recorded per image in the `source` column of `labels.csv`.

Pretrained models: [CLIP](https://github.com/openai/CLIP) (OpenAI) and
[SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
(NVIDIA, ADE20K).
