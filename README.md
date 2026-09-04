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

Upload or photograph a slope. The system returns:

1. **Class probabilities** — P(stable) and P(unstable), calibrated.
2. **An assessment** in one of three bands (see [thresholds](#asymmetric-thresholds)).
3. **What the model sees** — the image after sky and water removal, so you can check
   the preprocessing did not discard the slope.
4. **Where the model looked** — an occlusion heatmap of the regions the prediction
   actually depended on.

Available as a web app and an Android client. Both call the same backend, so both
give the same answer for the same photograph.

---

## Results

Grouped 5-fold cross-validation over 179 images, split by site so photographs of one
location never appear on both sides of a fold. 170 groups, **averaged over 10 fold
seeds** — a single seed varies by 4.5 points on fold assignment alone, so one seed is a
lottery ticket. The ± is the spread of per-seed means, i.e. uncertainty on the estimate.

The figure is written to `metrics.json` by `frozen_features.py` and served through
`/stats`, so the number shown in the app is the number that was measured.

| Configuration | Trained parameters | Accuracy |
|---|---:|---:|
| **Frozen CLIP + calibrated probe** *(deployed)* | **513** | **81.0% ± 1.1** |
| Frozen DINOv2 + probe | 385 | 76.3% |
| Frozen ResNet18 + probe | 513 | 66.9% |
| Fine-tuned ResNet18 *(baseline)* | ~8.4M | 66.5% ± 9.6 |

The third row is the finding. A frozen ResNet18 probe **matches** full fine-tuning
while training 8,000× fewer parameters — so overfitting was never the bottleneck.
Swapping the frozen features to CLIP gained 15 points. **The representation was the
constraint**, not the training procedure and not the volume of data.

That is also why the learning curve is flat: quadrupling the training set (35 → 136
images) gained 3.6 points, non-monotonically. Bulk collection does not help here.

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
| 1 | **Sky/water removal** — SegFormer segments against ADE20K; sky, water, sea, river and lake are background. Keeps the 2D bounding box of everything else. | cropped image |
| 2 | **Feature extraction** — frozen CLIP ViT-B/32, `vision_model` → `visual_projection`. | `float32[512]` |
| 3 | **Classification** — `StandardScaler` + balanced logistic regression, sigmoid-calibrated. | P(unstable) |
| 4 | **Assessment** — three bands at asymmetric thresholds. | stable / borderline / unstable |
| 5 | **Attribution** *(on request)* — occlusion. | PNG overlay |

Stage 1 has more guards than crop. If background covers under 3% of the frame nothing
is cropped; no edge may remove more than 55% of its axis; the result must stay at
least 32px per side; a close-up that is almost entirely background is returned
untouched. An earlier version cropped at the median sky depth over sky-containing
columns only — on a photo where a cliff fills one side top to bottom, those columns
are excluded entirely, so the median came from the open-horizon side and removed most
of the subject.

### What is and is not learned from slope data

| Component | Source | Knows about slopes? |
|---|---|---|
| SegFormer *(frozen)* | ADE20K scene parsing | No — it finds sky and water |
| CLIP ViT-B/32 *(frozen)* | ~400M image–caption pairs | No — it describes rock, soil, vegetation |
| **The probe** *(trained here)* | 179 labelled slope photos | **Yes — this is the entire classifier** |
| ResNet18 baseline *(trained here)* | 20 epochs, AdamW | Yes — the conventional approach, kept for comparison |

CLIP supplies the visual vocabulary; it has no concept of stability. The 512 weights
define a direction through that space which does not exist in CLIP and was found only
by fitting to labelled examples. In standard terminology: transfer learning with a
frozen backbone and a calibrated linear probe — the usual approach at hundreds of
images rather than hundreds of thousands.

### Asymmetric thresholds

```
P(unstable) ≥ 0.65   →  Potentially Unstable
0.35 ≤ P < 0.65      →  Borderline — field inspection recommended
P < 0.35             →  Stable
```

Flagging begins at 0.35, not 0.50, because missing an unstable slope costs more than a
false alarm. **The assessment can therefore disagree with the taller confidence bar**,
which the interface states outright rather than hiding.

Borderline is its own band with its own colour. It is the model reporting uncertainty,
which is a different claim from "unstable" and must not borrow its alarm.

### Attribution

Not Grad-CAM. Grad-CAM needs a convolutional feature map to weight by channel
gradient; CLIP's vision tower is a transformer and has none.

Instead: **mask a region, measure how far the unstable logit actually falls.** 14×14
sample points, 32px overlapping occluder, 196 batched forward passes, roughly 5s.
Faithful by construction — it measures the model rather than a proxy for it.

Two rendering details matter as much as the method:

- **Percentile clip (p85–p99), not min-max.** On a confident prediction almost every
  occlusion lowers the logit, so min-max paints the whole frame hot and hides the
  structure.
- **Reflect-pad by one cell and discard the outer ring.** Occluding a cell on the true
  frame edge produces a large spurious response. Measured: on one image the border
  ring averaged **+1.34** against **−0.07** interior, and padding so the same content
  moved inward collapsed it to **+0.01**. It was tracking the frame, not the
  photograph, and the percentile clip amplified it into a bright halo.

Validation: masking the top 10% of attributed cells destroys on average **44
percentage points more** of the unstable logit than masking the same number at random.

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

Two photographs of the same cutting taken metres apart are not independent samples. If
one lands in train and the other in validation, the reported accuracy measures memory
rather than generalisation.

`labels.csv` carries a `site_id`. Every loader converts a blank one into
`__solo__{filename}` so unlinked images become their own group rather than collapsing
into one giant false group. `cross_validate.py` asserts no group spans a fold;
`verify_dataset.py` checks the same property over the CSV.

### 3. Feature cache keying

Extracted features are cached to disk, keyed by a **hash of the file list** — never
its length. A count-keyed cache returns feature rows belonging to a previous set of
images whenever the dataset changes without changing size. They still line up
positionally against the new labels, so nothing errors and every downstream number is
quietly wrong.

`train_probe.py`, `frozen_features.py` and `dupes.py` all use the same scheme.

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

Add `--host 0.0.0.0` to reach it from a phone on the same network. Without it uvicorn
binds `127.0.0.1` and only the local machine can connect.

### Web frontend

```bash
npm --prefix frontend install
npm --prefix frontend run dev
```

Dev runs on `:5173` and calls the backend at `:8000`. For production, `npm run build`
writes `frontend/dist`, which FastAPI then serves itself — one origin, one port.

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
| `POST /predict` | scores, assessment, both thresholds | Rate limited: 40 / 10 min per client |
| `POST /cropped` | the image as the model receives it | |
| `POST /gradcam` | attribution overlay | ~5s |
| `GET /monitoring` | confidence distribution | Admin token; **fails closed** |

### Reproducing the results

```bash
python verify_dataset.py                              # gate: run before trusting anything
python frozen_features.py --backbones clip --folds 5  # the headline accuracy
python cross_validate.py --folds 5 --seed 42          # fine-tuned baseline
python audit_errors.py --seeds 1 2 3 --folds 5        # per-image failure rates
python learning_curve.py --fractions 0.25 0.5 0.75 1.0
python dupes.py --threshold 0.93                      # near-duplicate audit
python train_probe.py                                 # refit the deployed model
```

---

## Repository layout

```
model_utils.py        shared core: crop_sky, predict, attribution
slope_probe.joblib    the deployed model (513 parameters)
labels.csv            one row per image, 8 columns
slope_dataset/        179 images, {train,val}/{stable,unstable}/, pre-cropped

backend/main.py       FastAPI — 6 routes, warmup thread, rate limiting
db.py                 prediction log (no images, no plaintext IPs)
frontend/src/         React + Vite — App.jsx, api.js, 5 components
android/              Kotlin + Jetpack Compose client

train_probe.py        fits the deployed model
train_model.py        retained ResNet18 baseline
crop_dataset_sky.py   crop new images before training

cross_validate.py     grouped k-fold — the trustworthy number
frozen_features.py    backbone comparison
audit_errors.py       per-image failure rates across seeds
learning_curve.py     accuracy vs dataset size
dupes.py              near-duplicate audit
verify_dataset.py     invariant check
```

### Dataset

179 images — 82 stable, 97 unstable. Train 65/75, val 17/22.

Indicators, eight categories:

| category | stable | unstable |
|---|---:|---:|
| `engineered` | 37 | — |
| `bedrock` | 33 | — |
| `vegetation` | 12 | — |
| `erosion` | — | 37 |
| `debris` | — | 20 |
| `scarp` | — | 19 |
| `displacement` | — | 13 |
| `crack` | — | 8 |

Collapsed from an earlier 29-category scheme in which 15 categories had three examples
or fewer and several were the same feature under different names. **Note that no
category straddles the stable/unstable boundary** — `primary_indicator` is a
deterministic function of `label`, because the feature was named after the call was
made. That is a real limitation of the labelling and is discussed under
[Known limitations](#known-limitations).

`labels.csv` columns: `filename, split, label, primary_indicator, confidence, source,
site_id, notes`. The directory is the authority on split and label; the CSV carries
everything else.

---

## Design decisions

**The classifier is tiny on purpose.** 513 trained parameters, measured against a
fine-tuned 8.4M-parameter baseline that performs worse. This is the project's result,
not a shortcut around doing the work.

**Thresholds live on the server.** `POST /predict` returns both cutoffs, and every
client derives its band from the numbers rather than string-matching the assessment
prose. Rewording the copy cannot silently change how results are presented, and the
web and Android apps cannot drift apart.

**The Android app is a thin client.** Running the pipeline on-device means converting
two transformers to ONNX and reimplementing sigmoid calibration in Kotlin — and it
would fork the thresholds across two codebases. A phone that quietly disagrees with
the web app about the same photograph is worse than one that needs a network.

**No public contribution path.** The app analyses an image and discards it. Two things
about a submitted photo cannot be determined automatically: its licence/provenance —
not knowable from pixels, and a web form's "I own it" is unverifiable — and its true
label. The label is the serious one. The only labels available are the model's own
prediction, which feeds its documented shortcuts back into training while making the
metrics look better, and an anonymous guess.

**The database stores no images and no plaintext IP addresses.** One row per analysis:
probability, assessment, frame size. It exists to catch the probe drifting back toward
the overconfidence that calibration corrected, which is visible without labels as mass
piling into the outermost confidence bins. Rate limiting needs to recognise a repeat
caller, not identify one, so only a salted hash is kept.

**Preprocessing is inspectable.** `crop_sky` is the step most likely to misfire on an
unusual photo — a hazy skyline, a water-filled cut — and it is invisible in the
result. `POST /cropped` makes it viewable, and the interface reports how much area was
removed.

---

## Known limitations

**The indicator taxonomy is not independent of the label.** No category appears under
both classes: `engineered*` is always stable, `erosion`/`scarp`/`debris`/`crack` always
unstable. The feature was named because of the verdict, so the operational question is
closer to "is this an engineered structure or an eroded surface?" than "is this slope
failing?". This partly explains the semantic shortcut below — the label scheme rewards
exactly the separation CLIP is good at. Fixing it needs indicator labels assigned blind
to the class.

**It cannot see underground.** The most important one. Two slopes can look identical
and behave completely differently.

**The shortcut moved rather than vanished.** Under ResNet18 features the model treated
bare rock texture as evidence of instability: `stable_rock` failed 50% of the time,
`stable_cliff` 67%. CLIP features cut `stable_rock` to 9% — and raised `stable_cut`
from 19% to **47%**.

CLIP separates "engineered cutting" from "natural cliff" more reliably than it
separates failing ground from intact ground. The same geometry is visible directly in
the embedding space: `stable_rock_011` sits at cosine **0.9453** from
`unstable_cliff_016`, closer than most same-label pairs. The counterexamples that would
break it are unstable engineered cuts and stable natural cliffs.

**Accuracy is a range, not a number.** ±2.9 across folds. Earlier single-split figures
in this project were optimistic by 8–10 points; that is why `cross_validate.py` and
`frozen_features.py` exist and why a single held-out split is not quoted anywhere.

**Training was unseeded until it was fixed.** Six seeds on identical data produced
recall gaps of 3.5–43.9 percentage points — noise exceeding every dataset-driven
effect measured. All run-to-run comparisons predating the fix are meaningless.

**The heatmap does not always concentrate.** On one debris-fan image the attribution
performed *worse* than random. Diffuse indicators may have genuinely distributed
evidence, or the map fails there. And faithful is not the same as correct: on images
the model gets wrong, a faithful heatmap will faithfully show bad reasoning.

**Some dataset images lack licence provenance.** Several rows carry `source=unknown`
in a public repository. Unresolved.

**Indicator naming was attempted and rejected.** Telling the user *which* feature was
spotted was tested three ways — zero-shot CLIP 40%, zero-shot with centering 38%,
supervised head 41% ± 8 — against a 40% majority-class baseline. Not shippable. The
cause is data: per-indicator counts run as low as 3, and roughly 30–50 per class would
be needed.

---

## Acknowledgements

Dataset images are drawn from field photography and Wikimedia Commons (CC BY-SA and
compatible licences), recorded per image in the `source` column of `labels.csv`.

Pretrained models: [CLIP](https://github.com/openai/CLIP) (OpenAI) and
[SegFormer](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512) (NVIDIA,
ADE20K).
