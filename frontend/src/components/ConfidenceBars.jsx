export default function ConfidenceBars({ scores, threshold }) {
  if (!scores) return null;

  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);

  return (
    <div className="confidence-bars">
      <h3>Model Confidence</h3>
      {entries.map(([label, value]) => (
        <div className="confidence-row" key={label}>
          <div className="confidence-label">
            <span>{label}</span>
            <span>{(value * 100).toFixed(1)}%</span>
          </div>
          <div className="confidence-track">
            <div
              className={`confidence-fill confidence-fill--${label}`}
              style={{ width: `${value * 100}%` }}
            />
            {/* Marks the decision cutoff on the unstable bar, so it's visible why the
                Assessment can differ from whichever class scores highest. */}
            {label === "unstable" && threshold != null && (
              <span
                className="confidence-threshold"
                style={{ left: `${threshold * 100}%` }}
                title={`Flagged at ${(threshold * 100).toFixed(0)}%`}
              />
            )}
          </div>
        </div>
      ))}
      {threshold != null && (
        <p className="confidence-note">
          Marker = {(threshold * 100).toFixed(0)}% flag threshold
        </p>
      )}
    </div>
  );
}
