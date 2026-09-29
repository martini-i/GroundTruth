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

## Contents

- [What it does](#what-it-does)
- [Results](#results)
- [Architecture](#architecture)
- [The three invariants](#the-three-invariants)
- [Requirements](#requirements)
- [Running it](#running-it)
- [Repository layout](#repository-layout)
- [Design decisions](#design-decisions)
- [Known limitations](#known-limitations)

---

## What it does

Upload or photograph a slope. If the image is not one the classifier can speak to —
indoors, too dark, almost entirely sky, no ground surface in frame — it says so and
stops rather than scoring it. Otherwise it returns:

1. **Class probabilities** — P(stable) and P(unstable), calibrated.
2. **An assessment** in one of three bands (see [thresholds](#asymmetric-thresholds)).
3. **What the model sees** — the image after sky and water removal, so you can check
   the preprocessing did not discard the slope.

Available as a web app and an Android client. Both call the same backend, so both
give the same answer for the same photograph.

---

## Results

**No performance figure is written down in this file, or anywhere else in prose.**
The measured numbers live in `metrics.json`, written by `frozen_features.py` and
served through `/stats`, so the figure shown in the app is always the figure that was
last measured. A number typed into a README goes stale the moment the dataset grows,
silently and invisibly.

```bash
python frozen_features.py
```

What the comparison is *for* — which does not change as the data grows:

| Configuration | Trained parameters |
|---|---:|
| **Frozen CLIP + calibrated probe** *(deployed)* | ~500 |
| Frozen DINOv2 + probe | ~400 |
| Frozen ResNet18 + probe | ~500 |
| Fine-tuned ResNet18 *(baseline)* | ~8.4M |

The finding is the shape of that table, not its numbers. A frozen ResNet18 probe
**matches** full fine-tuning while training thousands of times fewer parameters — so
overfitting was never the bottleneck. Swapping the frozen features to CLIP gains a
large margin on top of that. **The representation was the constraint**, not the
training procedure and not the volume of data.

That is also why the learning curve is close to flat: quadrupling the training set
moves accuracy by a few points, non-monotonically. Bulk collection does not help
here; targeted counterexamples might, and which ones is answered under
[Known limitations](#known-limitations).

### How the numbers are produced

Grouped k-fold cross-validation, split by site so photographs of one location never
appear on both sides of a fold, **averaged over ten fold seeds** — a single seed
varies by several points on fold assignment alone, so one seed is a lottery ticket.
The ± reported is the spread of per-seed means: uncertainty on the estimate.

Each fold fits the *deployed* pipeline, calibration included, and reports two
operating points:

- **at 0.5** — accuracy and per-class recall, comparable across backbones and against
  the fine-tuned baseline;
- **at the deployed flag threshold** — unstable recall, precision, and how often the
  borderline band fires. This is the point a user of the app actually experiences,
  and it is deliberately not the same as the accuracy figure.

Plus ROC AUC, which is threshold-free, and Brier score, which asks whether the
calibrated probabilities mean what they say.

---

## Architecture

Two consumers share one core. A React app and a FastAPI backend serve predictions; a
set of CLI scripts fits and measures the model. Both call `model_utils.py`.

```
  React app  ─┐                          ┌─ CLIP ViT-B/32   (frozen, pretrained)
  Android app ─┼─▶ FastAPI ──▶ model_utils.py ─┼─ SegFormer      (frozen, pretrained)
  CLI scripts ─┘                          └─ slope_probe.joblib (fitted here)
```

### The inference path

Five stages run on every image. Order matters and each stage's output shape is fixed.

| # | Stage | Output |
|---|---|---|
| 0 | **Input gate** — refuse images the probe cannot speak to: too small, too dark, near-uniform, indoors, all sky, no recognisable ground. Reads the stage-1 segmentation, so it costs no extra pass. | assessable / refusal |
| 1 | **Sky/water removal** — SegFormer segments against ADE20K; sky, water, sea, river and lake are background. Keeps the 2D bounding box of everything else. | cropped image |
| 2 | **Feature extraction** — frozen CLIP ViT-B/32, `vision_model` → `visual_projection`. | `float32[512]` |
| 3 | **Classification** — `StandardScaler` + balanced logistic regression, sigmoid-calibrated. | P(unstable) |
| 4 | **Assessment** — three bands at asymmetric thresholds. | stable / borderline / unstable |

Stage 1 has more guards than crop. If background covers too little of the frame
nothing is cropped; no edge may remove more than a capped share of its axis; the
result must stay above a minimum size; a close-up that is almost entirely background
is returned untouched. The constants are named at the top of `model_utils.py`.

An earlier version cropped at the median sky depth over sky-containing columns only —
on a photo where a cliff fills one side top to bottom, those columns are excluded
entirely, so the median came from the open-horizon side and removed most of the
subject.

### What is and is not learned from slope data

| Component | Source | Knows about slopes? |
|---|---|---|
| SegFormer *(frozen)* | ADE20K scene parsing | No — it finds sky and water |
| CLIP ViT-B/32 *(frozen)* | ~400M image–caption pairs | No — it describes rock, soil, vegetation |
| **The probe** *(trained here)* | the labelled slope photographs | **Yes — this is the entire classifier** |
| ResNet18 baseline *(trained here)* | the same photographs, fine-tuned end to end | Yes — the conventional approach, kept for comparison |

CLIP supplies the visual vocabulary; it has no concept of stability. The probe's
weights define a direction through that space which does not exist in CLIP and was
found only by fitting to labelled examples. In standard terminology: transfer
learning with a frozen backbone and a calibrated linear probe — the usual approach at
hundreds of images rather than hundreds of thousands.

### Asymmetric thresholds

The two cutoffs live in `model_utils.py` (`UNSTABLE_THRESHOLD` and
`UNSTABLE_HIGH_CONFIDENCE`) and are returned by `/predict` on every response:

```
P(unstable) ≥ high    →  Potentially Unstable
flag ≤ P < high       →  Borderline — field inspection recommended
P < flag              →  Stable
```

Flagging begins below 0.5, not at it, because missing an unstable slope costs more
than a false alarm. **The assessment can therefore disagree with the taller
confidence bar**, which the interface states outright rather than hiding.

Borderline is its own band with its own colour. It is the model reporting
uncertainty, which is a different claim from "unstable" and must not borrow its
alarm.

---

## The three invariants

Each has been violated at least once, and each failure was silent.

### 1. Crop symmetry

`crop_sky` runs at **both** dataset-prep and inference time. Images in
`slope_dataset/` are stored already cropped, so `train_probe.py` deliberately does
**not** crop again.

Any image added to the dataset must be run through `crop_dataset_sky.py` before
training, or it is framed differently from every image the probe was fitted on.
Nothing errors; accuracy just quietly drops.

```bash
python crop_dataset_sky.py slope_dataset/train/unstable/new_photo.jpg
```

### 2. Site grouping

Two photographs of the same cutting taken metres apart are not independent samples.
If one lands in train and the other in validation, the reported accuracy measures
memory rather than generalisation.

`labels.csv` carries a `site_id`. Every loader converts a blank one into
`__solo__{filename}` so unlinked images become their own group rather than
collapsing into one giant false group. `frozen_features.py` and `cross_validate.py`
assert no group spans a fold; `verify_dataset.py` checks the same property over the
CSV.

### 3. Feature cache keying

Extracted features are cached to disk, keyed by a **hash of the file list plus each
file's size and modification time** — never by its length. A count-keyed cache
returns feature rows belonging to a previous set of images whenever the dataset
changes without changing size. They still line up positionally against the new
labels, so nothing errors and every downstream number is quietly wrong. Size and
mtime additionally catch an *in-place* edit, which re-cropping an existing image is.

`dataset.py` owns the scheme, and every script goes through it.

---

## Requirements

| | Version | Notes |
|---|---|---|
| Python | 3.13 | 3.10+ should work; `venv/` is the assumed location |
| Node | 22 | for the web frontend |
| JDK | 17 | Android client only |
| Android SDK | platform 35, build-tools 35.0.0 | Android client only |

Two dependency files, deliberately:

- **`requirements.txt`** — local training and analysis. Includes `matplotlib`, used
  only by `train_model.py`.
- **`backend/requirements.txt`** — runtime. A strict subset plus FastAPI.
  `scikit-learn` and `joblib` are runtime dependencies, not dev-only: the probe
  unpickles sklearn objects at load.

Both vision models are downloaded from Hugging Face on first run — CLIP ViT-B/32
(~600MB) and SegFormer-b0. Budget for that on a cold start.

---

## Running it

### Backend

```bash
python -m venv venv
venv/Scripts/pip install -r backend/requirements.txt
venv/Scripts/uvicorn backend.main:app --port 8000
```

Add `--host 0.0.0.0` to reach it from a phone on the same network. Without it
uvicorn binds `127.0.0.1` and only the local machine can connect.

### Web frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Dev runs on `:5173` and calls the backend at `:8000`. For production,
`npm run build` writes `frontend/dist`, which FastAPI then serves itself — one
origin, one port.

### Android client

See [`android/README.md`](android/README.md). Open `android/` in Android Studio, or:

```bash
cd android && ./gradlew.bat assembleDebug
```

### Endpoints

| Endpoint | Returns | Notes |
|---|---|---|
| `GET /health` | status, `models_ready` | Distinguishes "up" from "warm" |
| `GET /stats` | dataset and model facts | Read from `labels.csv` per request |
| `POST /predict` | scores, assessment, both thresholds | Rate limited; upload size capped; **422** if the image is not assessable |
| `POST /cropped` | the image as the model receives it | |
| `GET /monitoring` | confidence distribution | Admin token; **fails closed** |

### Reproducing the results

```bash
python verify_dataset.py
python check_gate.py
python frozen_features.py
python cross_validate.py
python audit_errors.py
python learning_curve.py
python dupes.py
python train_probe.py
```

Run `verify_dataset.py` first — it gates everything else. `frozen_features.py`
rewrites `metrics.json`; `train_probe.py` rewrites `slope_probe.joblib`. Each script
takes flags; `--help` lists them.

---

## Repository layout

```
model_utils.py        shared core: crop_sky, predict, the thresholds
slope_probe.joblib    the deployed model
labels.csv            one row per image
slope_dataset/        {train,val}/{stable,unstable}/, stored pre-cropped

backend/main.py       FastAPI — warmup lifespan, rate limiting, upload caps
db.py                 prediction log (no images, no plaintext IPs)
frontend/src/         React + Vite
android/              Kotlin + Jetpack Compose client

dataset.py            the single dataset loader and feature cache
train_probe.py        fits the deployed model
train_model.py        retained ResNet18 baseline
crop_dataset_sky.py   crop new images before training

frozen_features.py    backbone comparison; writes metrics.json
cross_validate.py     grouped k-fold for the fine-tuned baseline
audit_errors.py       per-image failure rates across seeds
learning_curve.py     accuracy vs dataset size
dupes.py              near-duplicate audit
verify_dataset.py     invariant check
check_gate.py         input-gate check: refuses nothing in the dataset
```

### Dataset

Current counts are served by `/stats` and shown on the app's About page;
`python verify_dataset.py` prints the same breakdown on the command line. They are
not written here because they change every time an image is added.

`labels.csv` columns: `filename, split, label, primary_indicator, confidence,
source, site_id, notes`. The directory is the authority on split and label; the CSV
carries everything else.

`primary_indicator` uses a small, deliberately coarse set of categories, collapsed
from an earlier scheme in which most categories had three examples or fewer and
several were the same feature under different names. **No category straddles the
stable/unstable boundary** — `primary_indicator` is a deterministic function of
`label`, because the feature was named after the call was made. That is a real
limitation of the labelling, discussed under
[Known limitations](#known-limitations).

---

## Design decisions

**The classifier is tiny on purpose.** A few hundred trained parameters, measured
against a fine-tuned 8.4M-parameter baseline that performs worse. This is the
project's result, not a shortcut around doing the work.

**Thresholds live on the server.** `POST /predict` returns both cutoffs, and every
client derives its band from the numbers rather than string-matching the assessment
prose. Rewording the copy cannot silently change how results are presented, and the
web and Android apps cannot drift apart. `frozen_features.py` imports the same two
constants, so the measurement follows the app rather than describing an older
version of it.

**The Android app is a thin client.** Running the pipeline on-device means
converting two transformers to ONNX and reimplementing sigmoid calibration in Kotlin
— and it would fork the thresholds across two codebases. A phone that quietly
disagrees with the web app about the same photograph is worse than one that needs a
network.

**No public contribution path.** The app analyses an image and discards it. Two
things about a submitted photo cannot be determined automatically: its
licence/provenance — not knowable from pixels, and a web form's "I own it" is
unverifiable — and its true label. The label is the serious one. The only labels
available are the model's own prediction, which feeds its documented shortcuts back
into training while making the metrics look better, and an anonymous guess.

**The database stores no images and no plaintext IP addresses.** One row per
analysis: probability, assessment, frame size. It exists to catch the probe drifting
back toward the overconfidence that calibration corrected, which is visible without
labels as mass piling into the outermost confidence bins. Rate limiting needs to
recognise a repeat caller, not identify one, so only a salted hash is kept.

**Preprocessing is inspectable.** `crop_sky` is the step most likely to misfire on
an unusual photo — a hazy skyline, a water-filled cut — and it is invisible in the
result. `POST /cropped` makes it viewable, and the interface reports how much area
was removed.

---

## Known limitations

**The indicator taxonomy is not independent of the label.** No category appears under
both classes: engineered structures are always stable; erosion, scarps, debris and
cracks are always unstable. The feature was named because of the verdict, so the
operational question is closer to "is this an engineered structure or an eroded
surface?" than "is this slope failing?". This partly explains the semantic shortcut
below — the label scheme rewards exactly the separation CLIP is good at. Fixing it
needs indicator labels assigned blind to the class.

**It cannot see underground.** The most important one. Two slopes can look identical
and behave completely differently.

**The shortcut moved rather than vanished.** Under ResNet18 features the model
treated bare rock texture as evidence of instability, failing badly on stable natural
rock and cliffs. CLIP features largely fixed that class — and made engineered
cuttings worse.

CLIP separates "engineered cutting" from "natural cliff" more reliably than it
separates failing ground from intact ground. The same geometry is visible directly in
the embedding space: some stable-rock and unstable-cliff images sit closer together
than most same-label pairs. The counterexamples that would break it are unstable
engineered cuts and stable natural cliffs, and those are the images worth collecting.

Run `python audit_errors.py` for the current per-image picture and `python dupes.py`
for the embedding-space one.

**Accuracy is a range, not a number.** Earlier single-split figures in this project
were optimistic by several points; that is why `cross_validate.py` and
`frozen_features.py` exist, and why a single held-out split is not quoted anywhere.

**Training was unseeded until it was fixed.** Six seeds on identical data produced
recall gaps far exceeding every dataset-driven effect measured. All run-to-run
comparisons predating the fix are meaningless.

**Some dataset images lack licence provenance.** Several rows carry `source=unknown`
in a public repository. Unresolved.

**Indicator naming was attempted and rejected.** Telling the user *which* feature was
spotted was tested three ways — zero-shot CLIP, zero-shot with centering, and a
supervised head — and none beat the majority-class baseline by a usable margin. Not
shippable. The cause is data: per-indicator counts run as low as three, and roughly
30–50 per class would be needed.

---

## Acknowledgements

Dataset images are drawn from field photography and Wikimedia Commons (CC BY-SA and
compatible licences), recorded per image in the `source` column of `labels.csv`.

Pretrained models: [CLIP](https://github.com/openai/CLIP) (OpenAI) and
[SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512)
(NVIDIA, ADE20K).
