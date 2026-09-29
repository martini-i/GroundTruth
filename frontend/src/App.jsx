import { useState, useEffect } from "react";
import ImageUpload from "./components/ImageUpload";
import ConfidenceBars from "./components/ConfidenceBars";
import AssessmentCard from "./components/AssessmentCard";
import HowItWorks from "./components/HowItWorks";
import About from "./components/About";
import { predictImage, croppedImage, fetchHealth } from "./api";
import "./App.css";

const EXAMPLES = [
  { label: "Natural rock", tone: "stable", path: "/examples/stable_cliff_001.jpg" },
  { label: "Engineered", tone: "stable", path: "/examples/stable_engineered_001.jpg" },
  { label: "Crack", tone: "unstable", path: "/examples/unstable_crack_001.jpg" },
  { label: "Scarp", tone: "unstable", path: "/examples/unstable_scarp_007.jpg" },
];

// Mirrors the backend's own cap. Rejecting here means a 12MB phone photo fails
// instantly with an explanation instead of after a long upload.
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

const VIEWS = [
  ["analyze", "Analyze"],
  ["how", "How it works"],
  ["about", "About"],
];

function Spinner() {
  return <span className="spinner" aria-hidden="true" />;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [croppedUrl, setCroppedUrl] = useState(null);
  const [loading, setLoading] = useState(null); // null | "analyze" | "cropped" | "example"
  const [error, setError] = useState(null);
  const [warm, setWarm] = useState(null); // null unknown | false warming | true ready
  const [view, setView] = useState("analyze");
  // Respect the reader's OS setting on a first visit; a stored choice always wins
  // after that. Defaulting everyone to dark ignored light-mode users entirely.
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") return saved;
    return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
  });

  useEffect(() => {
    localStorage.setItem("theme", theme);
  }, [theme]);

  // The backend loads CLIP and SegFormer in a background thread at startup. Until
  // that finishes the first analysis blocks for tens of seconds, which is
  // indistinguishable from a hang. Poll until ready so the UI can say which it is.
  useEffect(() => {
    let cancelled = false;
    let timer;

    async function check() {
      try {
        const h = await fetchHealth();
        if (cancelled) return;
        setWarm(!!h.models_ready);
        if (!h.models_ready) timer = setTimeout(check, 2000);
      } catch {
        if (!cancelled) {
          setWarm(false);
          timer = setTimeout(check, 5000);
        }
      }
    }
    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  // Paste a screenshot straight in, rather than making the user save it to disk
  // first. Bound to the document so it works without focusing the dropzone.
  useEffect(() => {
    function handlePaste(e) {
      const item = [...(e.clipboardData?.items ?? [])].find((i) =>
        i.type.startsWith("image/")
      );
      if (!item) return;
      const pasted = item.getAsFile();
      if (pasted) {
        setView("analyze");
        handleFileSelected(pasted);
      }
    }
    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, []);

  // Every object URL created below is revoked before being replaced and on
  // unmount. Without this each new photo and crop leaks its blob for
  // the lifetime of the tab.
  function releaseDerived() {
    setCroppedUrl((url) => {
      if (url) URL.revokeObjectURL(url);
      return null;
    });
  }

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      if (croppedUrl) URL.revokeObjectURL(croppedUrl);
    };
  }, [previewUrl, croppedUrl]);

  function handleFileSelected(selectedFile) {
    // Validate before uploading. The backend rejects these too, but doing it here
    // turns a round trip and a status code into an instant, specific message.
    if (!selectedFile.type.startsWith("image/")) {
      setError("That file isn't an image. Choose a JPEG or PNG photo.");
      return;
    }
    if (selectedFile.size > MAX_UPLOAD_BYTES) {
      const mb = (selectedFile.size / 1024 / 1024).toFixed(1);
      setError(`That image is ${mb} MB. The limit is 12 MB — try a smaller version.`);
      return;
    }

    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(selectedFile);
    });
    setFile(selectedFile);
    setResult(null);
    releaseDerived();
    setError(null);
  }

  function handleClear() {
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return null;
    });
    setFile(null);
    setResult(null);
    releaseDerived();
    setError(null);
  }

  async function handleExampleClick(path) {
    // Fetching the example is a network call and can fail. Unhandled, it rejected
    // silently and the click appeared to do nothing at all.
    setLoading("example");
    setError(null);
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`Couldn't load that example (${res.status}).`);
      const blob = await res.blob();
      handleFileSelected(
        new File([blob], path.split("/").pop(), { type: blob.type || "image/jpeg" })
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(null);
    }
  }

  async function handleShowCropped() {
    if (!file) return;
    setLoading("cropped");
    setError(null);
    try {
      const url = await croppedImage(file);
      setCroppedUrl((old) => {
        if (old) URL.revokeObjectURL(old);
        return url;
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(null);
    }
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading("analyze");
    setError(null);
    try {
      setResult(await predictImage(file));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(null);
    }
  }


  return (
    <div className="app" data-theme={theme}>
      <header>
        <div className="top-bar">
          <div className="brand">
            <span className="brand-mark" aria-hidden="true">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
                <path d="M2 19h20L15 6l-4 7-3-4-6 10z" fill="currentColor" opacity="0.9" />
              </svg>
            </span>
            <div>
              <h1>GroundTruth</h1>
              <p className="subtitle">Slope Surface Indicator Classifier</p>
            </div>
          </div>
          <button
            className="theme-toggle"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            {theme === "dark" ? "☀ Light mode" : "☾ Dark mode"}
          </button>
        </div>

        <nav className="tabs" aria-label="Sections">
          {VIEWS.map(([id, label]) => (
            <button
              key={id}
              className={`tab${view === id ? " tab--active" : ""}`}
              aria-current={view === id ? "page" : undefined}
              onClick={() => setView(id)}
            >
              {label}
            </button>
          ))}
        </nav>
      </header>

      {view === "analyze" && (
        <>
          <p className="looks-for">
            Looks for signs such as cracks · scarps · debris · disturbed soil · undercutting · rockfall
          </p>
          <blockquote>
            Identifies visible surface indicators of potential instability. It does{" "}
            <strong>not</strong> predict landslides or assess subsurface conditions. Research
            prototype, not a safety determination — consult a qualified geotechnical professional.
          </blockquote>

          {warm === false && (
            <p className="warming" role="status">
              <span className="spinner" aria-hidden="true" />
              Loading the vision models — the first analysis will be slow until this finishes.
            </p>
          )}

          <main className="main-grid">
            <section className="upload-section">
              <ImageUpload onFileSelected={handleFileSelected} previewUrl={previewUrl} />

              <div className="button-row">
                <button onClick={handleAnalyze} disabled={!file || loading} className="btn-primary">
                  {loading === "analyze" && <Spinner />}
                  {loading === "analyze" ? "Working…" : "Analyze"}
                </button>
                <button
                  onClick={handleShowCropped}
                  disabled={!file || loading}
                  className="btn-secondary"
                  title="See the image after sky and water are removed"
                >
                  {loading === "cropped" && <Spinner />}
                  {loading === "cropped" ? "Working…" : "What the model sees"}
                </button>
                {file && (
                  <button onClick={handleClear} disabled={loading} className="btn-ghost">
                    Clear
                  </button>
                )}
              </div>

              <div className="examples">
                <span className="examples-label">Try an example</span>
                <div className="examples-grid">
                  {EXAMPLES.map((ex) => (
                    <button
                      key={ex.path}
                      className="example-card"
                      onClick={() => handleExampleClick(ex.path)}
                      disabled={!!loading}
                      title={ex.label}
                    >
                      <img src={ex.path} alt={ex.label} />
                      <span className={`example-tag example-tag--${ex.tone}`}>{ex.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="results-section" aria-live="polite" aria-busy={!!loading}>
              {error && <p className="error">{error}</p>}
              {!result && !croppedUrl && !error && (
                <div className="placeholder">
                  <svg width="34" height="34" viewBox="0 0 24 24" fill="none"
                       stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
                    <path d="M3 17l5-6 4 4 3-3 6 5" strokeLinecap="round" strokeLinejoin="round" />
                    <circle cx="8.5" cy="7.5" r="1.6" />
                  </svg>
                  <p>Results will appear here.</p>
                  <span>Choose a photo, then press Analyze.</span>
                </div>
              )}
              {result && (
                <>
                  <ConfidenceBars
                    scores={result.scores}
                    threshold={result.unstable_threshold}
                  />
                  <AssessmentCard
                    assessment={result.assessment}
                    pUnstable={result.scores?.unstable}
                    threshold={result.unstable_threshold}
                    highConfidence={result.unstable_high_confidence}
                  />
                  {result.sky_cropped > 0 && (
                    <p className="crop-note">
                      Sky and open water cropped before analysis — model saw{" "}
                      {Math.round((1 - result.sky_cropped) * 100)}% of the photo.
                    </p>
                  )}
                </>
              )}
              {croppedUrl && (
                <div className="crop-panel">
                  <h3>What the model sees</h3>
                  <img src={croppedUrl} alt="The photo after sky and water removal"
                       className="crop-image" />
                  <p className="crop-hint">
                    Sky and open water are removed before analysis so the classifier reads the
                    ground, not the weather. If this crop lost the slope itself, treat the result
                    with suspicion.
                  </p>
                </div>
              )}
            </section>
          </main>

          <footer>
            <p>
              The Assessment is the actual call: it flags at 35% P(unstable), not 50%, because
              missing an unstable slope costs more than a false alarm. It can disagree with the
              highest confidence bar — that's intentional.
            </p>
            <ul>
              <li>P(unstable) ≥ 65% → Potentially Unstable</li>
              <li>35% ≤ P(unstable) &lt; 65% → Potentially Unstable (borderline) — inspect further</li>
              <li>P(unstable) &lt; 35% → Stable</li>
            </ul>
          </footer>
        </>
      )}

      {view === "how" && <HowItWorks />}
      {view === "about" && <About />}
    </div>
  );
}
