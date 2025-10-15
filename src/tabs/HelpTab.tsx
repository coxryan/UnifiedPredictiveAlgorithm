import { EDGE_MIN, VALUE_MIN } from "./constants";

export default function HelpTab() {
  return (
    <>
      <section className="card">
        <div className="card-title">Indicators &amp; Definitions</div>
        <div className="note">
          <ul>
            <li><b>Book-style spread</b>: Negative favors <b>home</b>; positive favors <b>away</b>.</li>
            <li><b>Model (H)</b>: Our predicted book-style spread for the home team.</li>
            <li><b>Market (H)</b>: Consensus market spread (home perspective).</li>
            <li><b>Expected (H)</b>: Market level implied by our advantage mapping.</li>
            <li><b>Edge</b>: Model − Market (book). Positive = away lean; negative = home lean.</li>
            <li><b>Value</b>: Market − Expected. Positive = value to away; negative = value to home.</li>
            <li><b>Qualified</b>: ✓ when model &amp; expected agree and |Edge| ≥ {EDGE_MIN} &amp; |Value| ≥ {VALUE_MIN}.</li>
            <li><b>Backtest</b>: 2024 season calculated each build; tab shows W/L/Push + Hit% per week.</li>
            <li><b>QB Impact</b>: Flagged when CFBD player usage shows a meaningful quarterback availability drop (either side).</li>
          </ul>
        </div>
      </section>

      <section className="card" style={{ marginTop: 16 }}>
        <div className="card-title">Workflow &amp; Cache Quick Reference</div>
        <div className="note">
          <ul>
            <li><b>Deploy</b> (`deploy.yml`): runs on every push. Rebuilds schedule, markets, predictions, residual model, live edge report, and regenerates the site bundle.</li>
            <li><b>Live markets</b> (`fetch_markets_live.yml`): runs every Fri/Sat 5 min (and manually). Refreshes current-week markets + live scores, relying on cached FanDuel data.</li>
            <li><b>FanDuel snapshot</b> (`fetch-fanduel-odds.yml`): scheduled Thu 00:00 PT, Fri 00:00 PT, Sat 07:00 PT (14:00 UTC) with manual dispatch optional. Refreshes `.cache_odds/&lt;year&gt;` and commits the odds snapshot.</li>
            <li><b>Backtest</b> (`build_backtest.yml`): manual. Rebuilds historic seasons (default 2024) with optional offline mode.</li>
            <li><b>CFBD cache</b>: `.cache_cfbd/&lt;year&gt;` (90d TTL) stores schedule/teams/stats; purge only when stale.</li>
            <li><b>Odds cache</b>: `.cache_odds/&lt;year&gt;` (2d TTL) stores Odds API payloads; refreshed by the scheduled snapshot workflow.</li>
            <li><b>`upa_data.sqlite`</b>: regenerated or updated each run; copied into `dist/data/` alongside debug logs for download/audits.</li>
          </ul>
        </div>
      </section>

      <section className="card" style={{ marginTop: 16 }}>
        <div className="card-title">How We Price Games</div>
        <div className="note">
          <ol>
            <li><b>Baseline rating model</b>: blends WRPS, Talent, SRS, and efficiency scores to create a home-minus-away rating differential plus contextual home-field advantage.</li>
            <li><b>Availability adjustments</b>: CFBD player-usage translates into offense/defense/special/QB availability metrics. Low QB availability triggers alerts and dampens confidence.</li>
            <li><b>Residual ensemble</b>: ridge + gradient-boosted stumps learn historical residuals (market vs final margin) and output calibrated adjustments.</li>
            <li><b>Anchoring</b>: calibrated residuals, scaled by source weights and confidence, produce `market_adjustment`. We add it to the bookmaker spread (or baseline if the market is missing) for `model_spread_book`.</li>
            <li><b>Expected blend</b>: the dampened line `expected_market_spread_book = λ*M_market + (1-λ)*M_model` derives value/edge metrics and drives qualification thresholds.</li>
          </ol>
        </div>
      </section>
    </>
  );
}
