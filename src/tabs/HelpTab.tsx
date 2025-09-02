import { EDGE_MIN, VALUE_MIN } from "./constants";

export default function HelpTab() {
  return (
    <section className="card">
      <div className="card-title">Indicators & Definitions</div>
      <div className="note">
        <ul>
          <li><b>Book-style spread</b>: Negative favors <b>home</b>; positive favors <b>away</b>.</li>
          <li><b>Model (H)</b>: Our predicted book-style spread for the home team.</li>
          <li><b>Market (H)</b>: Consensus market spread (home perspective).</li>
          <li><b>Expected (H)</b>: Market level implied by our advantage mapping.</li>
          <li><b>Edge</b>: Model − Market (book). Positive = away lean; negative = home lean.</li>
          <li><b>Value</b>: Market − Expected. Positive = value to away; negative = value to home.</li>
          <li><b>Qualified</b>: ✓ when model & expected agree and |Edge| ≥ {EDGE_MIN} & |Value| ≥ {VALUE_MIN}.</li>
          <li><b>Backtest</b>: 2024 season calculated each build; tab shows W/L/Push + Hit% per week.</li>
        </ul>
      </div>
    </section>
  );
}