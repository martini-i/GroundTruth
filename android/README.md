# GroundTruth — Android client

Field companion to the GroundTruth slope surface-indicator classifier. Photograph a
slope, get the same assessment the web app gives, on the same model.

Kotlin + Jetpack Compose, single module, no navigation library.

---

## What this is, and what it is not

**It is a thin client.** Nothing is classified on the phone. The app prepares a
JPEG and posts it to the existing FastAPI backend.

That is a deliberate choice, not a shortcut. The deployed classifier is a
calibrated probe on frozen CLIP ViT-B/32 features, with SegFormer removing sky and
water first — roughly 600MB of weights plus a scikit-learn estimator. Running it
on-device means converting two transformers to ONNX or TFLite and reimplementing
the sigmoid calibration in Kotlin.

The stronger reason is correctness. The decision thresholds are asymmetric on
purpose — the model flags at P(unstable) ≥ 0.35 rather than 0.5, because missing an
unstable slope costs more than a false alarm — and there are three assessment
bands, not two. Duplicating that logic on-device creates two places for it to
drift, and a phone that quietly disagrees with the web app about the same
photograph is worse than a phone that needs a network.

So the app reads **both** thresholds out of the `/predict` response and derives the
band from the numbers. Change them on the server and the phone follows.

---

## Running it

### 1. Make the backend reachable from the phone

**This is the step that will bite you.** The launch config runs:

```
uvicorn backend.main:app --port 8000
```

which binds `127.0.0.1` — reachable only from the dev machine itself. A phone on
the same Wi-Fi cannot connect to it. For device testing, bind all interfaces:

```
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Then find the machine's LAN address (`ipconfig` on Windows) and allow it through
the firewall on port 8000.

CORS is not involved — the backend's `allow_origins` list only governs browsers,
and a native client sends no `Origin` header.

### 2. Point the app at it

Open **Server** in the app and enter the address. `http://` is added if you leave
it off.

| Running on | Address |
|---|---|
| Android emulator | `10.0.2.2:8000` (the host's loopback, as seen from the emulator) |
| Physical phone | the dev machine's LAN address, e.g. `192.168.1.42:8000` |

`localhost` from a phone means *the phone*, which is the most common mistake here.

### 3. Allow the plaintext address

Android blocks cleartext HTTP by default from API 28 onward. `10.0.2.2`,
`localhost` and `127.0.0.1` are already permitted in
`app/src/main/res/xml/network_security_config.xml`. **A LAN address must be added
there too**, or the request is blocked before it leaves the device and surfaces as
a generic connection failure:

```xml
<domain includeSubdomains="true">192.168.1.42</domain>
```

Release builds keep the default HTTPS-only policy. Putting a real deployment
behind TLS is a prerequisite for shipping, not an optional extra.

---

## Layout

```
data/
  Models.kt      Prediction, Band, typed ApiError
  Api.kt         OkHttp client over /health /predict /cropped /gradcam
  Settings.kt    DataStore-backed server address
util/
  ImagePrep.kt   downscale + EXIF rotation before upload
ui/
  AnalyzeScreen.kt   capture, analyse, results
  SettingsScreen.kt  server address and reachability
  Components.kt      AssessmentCard, ConfidenceBars, Notice
  theme/Theme.kt     Material scheme + the three band colours
AnalyzeViewModel.kt  state, health polling, error mapping
```

### Things worth knowing

**Photos are downscaled and de-rotated before upload.** A phone camera produces
12–50MP files against a 12MB server cap, and a 50MP image decoded at full size is
~200MB in memory. `ImagePrep` decodes with `inSampleSize` to a 1600px long edge.

It also applies the EXIF orientation tag. This is not cosmetic: phones record
portrait photos as landscape pixels plus a rotation tag, and the classifier was
fitted entirely on upright photographs. A sideways frame is out of distribution and
the prediction is unreliable in a way nothing else would surface.

**Warming is distinguished from unreachable.** A cold backend loads CLIP and
SegFormer lazily, so the first analysis blocks for tens of seconds — on a phone,
indistinguishable from a freeze. The ViewModel polls `/health` and the UI reports
"loading the vision models" (amber) separately from "can't reach the server" (red).

**Errors are typed.** `ApiError` separates unreachable, rate-limited, rejected and
server failure, and FastAPI's `detail` string is surfaced verbatim, so a rate limit
reads "Rate limit reached (40 per 10 minutes)" rather than "HTTP 429".

**Captures go to the app's cache**, not the device gallery, via `FileProvider`.
Gallery selection uses the modern photo picker, so the app needs no storage
permission and can only see what the user explicitly chose.

---

## Not built yet

- **No tests.** No instrumented or unit tests exist.
- **No offline mode.** The app is useless without a reachable backend, which is a
  real limitation for the field use this is meant for — see below.
- **No result history.** Each analysis is discarded when the next photo is picked.
  Field use probably wants a saved log with GPS and timestamp.
- **No GPS capture.** Location would need a runtime permission and a privacy
  decision about where coordinates are stored.
- **Icon is a placeholder** vector, not a designed asset.

## If offline inference becomes necessary

Realistic path, in order of difficulty:

1. **Drop SegFormer.** Sky cropping is the heavier half of the pipeline. Either
   ship a much smaller segmentation model or skip cropping on-device — but note
   that would break the train/serve symmetry the project depends on, since every
   fitted image was cropped. Measure the accuracy cost before accepting it.
2. **Export CLIP's vision tower to ONNX** (`openai/clip-vit-base-patch32`,
   ~350MB as float32, ~90MB quantised to int8) and run it with ONNX Runtime
   Mobile. Verify the embeddings match the Python ones to a tight tolerance —
   silent preprocessing differences are the usual failure here.
3. **Port the probe.** It is 512 weights and a bias plus a `StandardScaler`, so
   the linear part is trivial. The sigmoid calibration is the fiddly bit and must
   be exported explicitly, not re-fitted.
4. **Keep the thresholds server-sourced anyway**, cached locally, so the two
   surfaces cannot disagree.

Expect roughly 100–400MB of app size and a few seconds per inference on a mid-range
device.
