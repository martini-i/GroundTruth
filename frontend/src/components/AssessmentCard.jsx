// The backend produces three bands, not two. Collapsing the middle one into
// "unstable" showed an uncertain result in full alarm styling, which overstates
// what the model actually claims — the opposite of what this project is for.
//
// The band is derived from the probability and the two thresholds the API
// returns, rather than by matching on the assessment prose, so rewording the
// text can never silently change how a result is presented.
const BANDS = {
  stable: {
    tone: "stable",
    heading: "Stable",
    note: "No significant visible surface indicators detected.",
  },
  borderline: {
    tone: "borderline",
    heading: "Borderline",
    note: "Some possible indicators. The model is genuinely uncertain here — field inspection recommended.",
  },
  unstable: {
    tone: "unstable",
    heading: "Potentially Unstable",
    note: "Visible surface indicators present — cracks, scarps, loose debris or disturbed soil.",
  },
};

function bandFor(pUnstable, threshold, high) {
  if (pUnstable == null || threshold == null) return null;
  if (pUnstable >= (high ?? 0.65)) return BANDS.unstable;
  if (pUnstable >= threshold) return BANDS.borderline;
  return BANDS.stable;
}

export default function AssessmentCard({ assessment, pUnstable, threshold, highConfidence }) {
  if (!assessment) return null;

  // Falls back to the server's prose if the probability is unavailable, so the
  // card still renders rather than disappearing.
  const band = bandFor(pUnstable, threshold, highConfidence);
  const tone = band ? band.tone : assessment.startsWith("Potentially Unstable") ? "unstable" : "stable";

  return (
    <div className={`assessment-card assessment-card--${tone}`}>
      <h3>Assessment</h3>
      {band ? (
        <>
          <p className="assessment-headline">{band.heading}</p>
          <p>{band.note}</p>
        </>
      ) : (
        <p>{assessment}</p>
      )}
    </div>
  );
}
