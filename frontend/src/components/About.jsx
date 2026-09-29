import { useEffect, useState } from "react";
import { fetchStats } from "../api";

// Accuracy is read from /stats, which serves metrics.json as written by
// frozen_features.py. It used to be hardcoded here and went stale silently —
// the page showed 82% +/- 7 long after the measurement had moved twice.
const LIMITS = [
  {
    title: "It cannot see underground",
    body:
      "Slope stability depends on water content, bedding orientation and loading history — none of " +
      "which appear in a photograph. Two slopes can look identical and behave differently.",
  },
  {
    title: "Accuracy is a range, not a number",
    body:
      "Cross-validation folds vary around the mean, so the spread is quoted alongside it.",
  },
];

export default function About() {
  const [stats, setStats] = useState(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetchStats().then(setStats).catch(() => setFailed(true));
  }, []);

  const d = stats?.dataset;
  const m = stats?.metrics;

  return (
    <div className="page">
      <section className="page-block">
        <h2>Measured performance</h2>
        <div className="stat-row">
          <div className="stat">
            <span className="stat-value">
              {m?.accuracy != null ? (
                <>
                  {Math.round(m.accuracy)}%{" "}
                  <span className="stat-spread">±{Math.round(m.spread)}</span>
                </>
              ) : failed ? "—" : "…"}
            </span>
            <span className="stat-label">accuracy, grouped cross-validation</span>
          </div>
          <div className="stat">
            <span className="stat-value">{d ? d.total : failed ? "—" : "…"}</span>
            <span className="stat-label">labelled images</span>
          </div>
          <div className="stat">
            <span className="stat-value">
              {d ? `${d.stable} / ${d.unstable}` : failed ? "—" : "…"}
            </span>
            <span className="stat-label">stable / unstable</span>
          </div>
        </div>
        {m?.deployed?.unstable_recall != null && (
          <div className="stat-row">
            <div className="stat">
              <span className="stat-value">{Math.round(m.deployed.unstable_recall)}%</span>
              <span className="stat-label">
                unstable slopes caught, at the threshold this app actually flags at
              </span>
            </div>
            <div className="stat">
              <span className="stat-value">{Math.round(m.deployed.unstable_precision)}%</span>
              <span className="stat-label">of flagged slopes are genuinely unstable</span>
            </div>
            <div className="stat">
              <span className="stat-value">{Math.round(m.deployed.borderline_rate)}%</span>
              <span className="stat-label">land in the borderline band</span>
            </div>
          </div>
        )}
        <p className="page-note">
          The first figure is accuracy at the even 50% split, which is what makes it comparable with
          other models. The row below it is the operating point you get: flagging starts well below
          50%, which catches more unstable slopes at the cost of more false alarms.
        </p>
        <p className="page-note">
          Folds are split by site, so photographs of the same location never appear on both sides of a
          split. Without that, near-duplicate images inflate the score.
        </p>
      </section>

      {d?.top_indicators?.length > 0 && (
        <section className="page-block">
          <h2>What the dataset contains</h2>
          <div className="tag-row">
            {d.top_indicators.map(([name, count]) => (
              <span className="tag" key={name}>
                {name.replace(/_/g, " ")}
                <em>{count}</em>
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="page-block">
        <h2>Limitations</h2>
        <div className="limit-grid">
          {LIMITS.map((l) => (
            <div className="limit" key={l.title}>
              <h3>{l.title}</h3>
              <p>{l.body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="page-block">
        <h2>Intended use</h2>
        <p>
          A research prototype exploring whether visible surface indicators can be classified from
          ordinary ground-level photographs. It is not a safety tool — for decisions about whether
          ground is safe to occupy, traverse or build on, consult a qualified geotechnical
          professional.
        </p>
      </section>
    </div>
  );
}
