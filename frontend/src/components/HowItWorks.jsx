const STEPS = [
  {
    n: "01",
    title: "Sky removal",
    body:
      "A pretrained segmentation model finds the sky and crops it away. Without this the classifier " +
      "reads cloud texture as terrain — an early version keyed on clouds instead of the slope.",
  },
  {
    n: "02",
    title: "Feature extraction",
    body:
      "The cropped photo passes through a frozen CLIP vision transformer, which turns it into a " +
      "512-number description of what the image contains. Nothing here is trained on slope data.",
  },
  {
    n: "03",
    title: "Classification",
    body:
      "A small calibrated classifier reads that description and estimates the probability of visible " +
      "instability indicators. It has roughly a thousand parameters, not millions.",
  },
  {
    n: "04",
    title: "Threshold",
    body:
      "The probability is compared against a 35% cutoff rather than 50%, because missing an unstable " +
      "slope is treated as a costlier error than a false alarm.",
  },
];

const INDICATORS = [
  ["Tension cracks", "Open cracks running across or along a slope surface"],
  ["Scarps", "Fresh, steep breaks where ground has dropped away"],
  ["Debris and talus", "Loose material accumulated below a face"],
  ["Disturbed soil", "Bare or churned ground where cover has been stripped"],
  ["Undercutting", "Material removed from the base, leaving an overhang"],
  ["Rockfall evidence", "Detached blocks on or below the slope"],
];

export default function HowItWorks() {
  return (
    <div className="page">
      <section className="page-block">
        <h2>The pipeline</h2>
        <p className="page-lead">
          Four stages run on every image. Each one is deliberately simple and inspectable.
        </p>
        <ol className="steps">
          {STEPS.map((s) => (
            <li key={s.n} className="step">
              <span className="step-num">{s.n}</span>
              <div>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section className="page-block">
        <h2>What counts as an indicator</h2>
        <p className="page-lead">
          These are surface features a person can point at in a photograph — not inferences about
          what is happening underground.
        </p>
        <div className="indicator-grid">
          {INDICATORS.map(([name, desc]) => (
            <div className="indicator" key={name}>
              <h3>{name}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="page-block">
        <h2>Checking what the model saw</h2>
        <p>
          Sky and open water are cropped away before analysis, so the classifier reads ground
          rather than weather. "What the model sees" shows the result of that step. It is worth
          checking on an unusual photograph: if the crop has removed the slope itself, the
          result below it is describing whatever was left.
        </p>
      </section>
    </div>
  );
}
