# UPA-F Development Manual

> **Agent instructions**  
> 1. Read this file at the start of every session.  
> 2. Follow the workflows, validation steps, and operational guardrails below.  
> 3. Update this document whenever you ship changes that affect process, data flow, or automation.

---

## Overview

Unified Predictive Algorithm — Full Repo (UPA-F) ships a static dashboard (Vite + React + TypeScript) backed by a Python data pipeline. GitHub Actions build the SQLite datastore (`data/upa_data.sqlite`) and JSON status artifacts; the frontend loads them with `sql.js`, so the site can be deployed as static pages without an API.

The dashboard currently exposes seven tabs (Status, Team, Predictions, Live Results, Recommended Bets, Backtest 2024, Help) that all read from the same SQLite database plus a few JSON diagnostics under `data/`.

---

## Architecture Snapshot

- **Frontend**: `src/` React app, built with Vite. Data access lives in `src/lib/db.ts` (SQLite via `sql.js`) and `src/lib/csv.ts` (helpers for number formatting, JSON pulls).
- **Data store**: `agents/storage/sqlite_store.py` wraps SQLite writes/reads, persists both tabular datasets and JSON blobs (`data_json` table). Primary database lives at `data/upa_data.sqlite`.
- **Collectors & model pipeline**: Modules under `agents/collect/` produce schedule, markets, team inputs, predictions, backtests, status JSON, and diagnostics. `agents/collect/helpers.write_dataset` centralises market backfill, grading, and artifact bookkeeping.
- **Jobs**: `agents/jobs/*.py` provide narrow entry points for automation (daily/weekly refreshes, live markets, residual model training).
- **Tools**: `tools/` contains CLI utilities used by workflows (`build_site_data`, `build_backtest_data`, `fetch_fanduel_odds`, `validate_site_data`).
- **Static delivery**: Deploy workflow copies everything in `data/` into `dist/data/` so the published Pages bundle always carries the latest SQLite + JSON payloads.

---

## Data Flow Summary

1. **Schedule & team inputs**: `agents.collect.team_inputs.build_team_inputs_datadriven` ingests CFBD team metadata, player usage, and efficiency stats (via `agents.collect.stats_cfbd`). Outputs are written with `write_dataset`, which also records artifact metadata.
2. **Markets & live scores**: `agents.collect.markets.get_market_lines_for_current_week` (FanDuel first, CFBD fallback) and `agents.collect.live_scores.update_live_scores` hydrate bookmaker spreads plus recent ESPN scoreboard results.
3. **Predictions**: `agents.collect.predictions.build_predictions_for_year` merges schedule, markets, team inputs, and live scores, applies the residual spread ensemble (`agents.collect.spread_model`), and flags completed games. `write_dataset` backfills missing spreads (FanDuel → CFBD → legacy → model) and grades results into `model_result`.
4. **Status & diagnostics**: `_upsert_status_market_source` in `agents.collect.status` refreshes `data/status.json` with run metadata, market source, fallback reason, and generated timestamp. Additional debug JSON (`market_predictions_backfill.json`, `market_unmatched.json`) are emitted via helper modules.
5. **Publishing**: `tools.build_site_data` copies curated artifacts into `dist/data/`; `tools.validate_site_data` fails the build if required datasets are empty or stale.

---

## Automation & Schedules

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `deploy` | Push to `main` and manual dispatch | End-to-end build: collector, residual model, validation, UI build, commit data, publish Pages. Supports optional cache reuse and backtest pass. |
| `refresh-cfbd-daily` | 09:00 UTC daily | Refresh CFBD schedule/lines snapshots via `agents.jobs.refresh_cfbd_daily`; commits `data/upa_data.sqlite` if updated. |
| `refresh-cfbd-weekly` | 10:00 UTC Mondays | Weekly CFBD refresh (`agents.jobs.refresh_cfbd_weekly`) to capture roster/team metadata updates. |
| `refresh-cfbd-stats-weekly` | 12:00 UTC Tuesdays | Rebuild team stat features powering `team_inputs`. |
| `refresh-markets-live` | Every 5 min on Fri/Sat | Pull current FanDuel odds (cache-only mode) plus ESPN live scores via `agents.jobs.refresh_markets_live`. |
| `fetch-fanduel-odds` | Thu/Fri 07:00 UTC and Sat 14:00 UTC (manual enabled) | Full FanDuel pull with `tools.fetch_fanduel_odds`, updates `data/debug/fanduel_odds_snapshot.json` and appends odds history to SQLite. |
| `train-spread-model` | 11:00 UTC Tuesdays | Train residual spread ensemble (`agents.jobs.train_spread_model`) and persist metrics + calibrated adjustments to the DB. |
| `build-backtest` | Manual | Generate historical season datasets with `tools.build_backtest_data` (online or offline cache mode), rebuild UI, and publish Pages artifact. |

---

## Key Data Artifacts

| Artifact | Produced by | Notes |
| --- | --- | --- |
| `data/upa_data.sqlite` | All collector/tooling flows | Canonical SQLite store consumed by the UI (`sql.js`). Contains tables such as `upa_predictions`, `upa_team_inputs_datadriven_v0`, `cfb_schedule`, `market_debug`, `live_edge_report`, backtest tables, and JSON payloads in `data_json`. |
| `data/status.json` | `_upsert_status_market_source` | Dashboard status card: timestamps, counts, market source, fallback reason. |
| `data/market_debug.json` & `data/market_debug.csv` | Market ingestion + `write_dataset` | FanDuel + CFBD spread history, used by Status downloads and for FanDuel backfill audits. |
| `data/market_unmatched.json` | Markets normaliser | List of events that failed FanDuel ↔ schedule matching; surfaced on Status tab for alias triage. |
| `data/market_predictions_backfill.json` | `write_dataset("upa_predictions")` | Summary of how many predictions rows were filled by FanDuel vs fallbacks (observability for synthetic spreads). |
| `data/debug/collector.log` | `agents.collect_cfbd_all` (file logger) | DEBUG-level log for the most recent collector run. |
| `data/debug/fanduel_odds_snapshot.json` | `tools.fetch_fanduel_odds` | Raw Odds API response archive used to troubleshoot market coverage. |

All artifacts are copied into `dist/data/` during deploy so the published site references static paths.

---

## Recent Changes (October 2025)

- Status tab now surfaces spread-band accuracy for each completed week and highlights market fallback reasons.
- Recommended Bets tab defaults to double-digit spreads, shares spread band filters with the Status tab, and honours tightened qualification thresholds (`src/tabs/constants.tsx`).
- Prediction builder integrates CFBD player-usage availability metrics and residual spread adjustments; `write_dataset` grades outcomes into `model_result` for MAE/weekly accuracy tables.
- FanDuel odds ingestion runs on a dedicated workflow with cache reuse knobs (`FANDUEL_CACHE_ONLY`) so deploy runs can anchor to the latest snapshot without hammering the Odds API.
- Deploy workflow rebases with `--autostash`, validates datasets both pre/post residual training, and uploads an auxiliary site bundle for the live markets workflow.

---

## Model Inputs & Feature Blending

The `build_team_inputs_datadriven` pipeline now blends:

- **Returning production (WRPS)**: offensive/defensive/overall percentages plus a 0–100 composite (`wrps_percent_0_100`).
- **Talent rating**: CFBD “talent” metric scaled to 0–100.
- **Current-season SRS**: rating + rank + scaled score (`srs_score_0_100`).
- **SP+ Ratings (2025 in-season)**:
  - Overall rating/rank and opponent-adjusted strength of schedule (`sp_rating`, `sp_ranking`, `sp_sos` plus 0–100 scaled variants).
  - Offensive/defensive SP+ components (`sp_off_rating`, `sp_def_rating`, success subscores) mapped to 0–100.
- **Advanced efficiency (CFBD advanced season stats)**:
  - Points per drive, success rates, total PPA, and derived yards-per-play after opponent adjustments (`stat_off_ppd_adj`, `stat_def_success_adj`, etc.).
  - Situational splits for passing/standard downs, points per opportunity, havoc pressure, and field-position expectation (e.g., `stat_off_pd_success`, `stat_off_havoc_front`).
- **Rolling form metrics**: 4-game opponent-adjusted trends for success, points-per-drive, PPA, and YPP (`rolling_off_ppd_4`, `rolling_def_success_rate_4`).
- **Legacy efficiency set**: points-per-game, yards-per-play, success rate, explosiveness for offense/defense, special-teams points/play (still pulled from CFBD team stats).
- **Availability & usage concentration**: CFBD player-usage availability scores plus top-contributor shares and QB usage (`availability_off_top3_usage_pct`, `availability_qb_usage_pct`).
- **Cross-model power ratings**: SP+, FPI (efficiencies + resume ranks), and Elo combined to anchor baselines (`sp_rating_0_100`, `fpi_rating_0_100`, `elo_rating_0_100`).
- **Portal placeholder**: `portal_net_*` columns remain for future integration.
- **Confidence calibration**: Weekly reliability curves (weeks 4+) generate a conservative `confidence_calibrated` value per pick, using Wilson lower bounds, spread-band adjustments, and market-source penalties. Weeks 1–3 remain purely preseason-weighted, then the model pivots to stats-backed calibration. `confidence_play_flag` marks recommendations once calibrated probability ≥ 0.62 with the usual qualification filters still applied.
- **Seasonal weighting note**: The first three weeks lean almost entirely on preseason priors (WRPS, talent, recruiting history). Starting Week 4 the collector blends in live-season stats, rolling form, and advanced efficiencies; the calibration step explicitly excludes Weeks 1–3 when it builds the reliability curves so the confidence signal reflects in-season performance only.

Position grades (QB/WR/RB/OL/DL/LB/DB/ST) now weight SP+, advanced efficiency, and availability metrics in addition to the historical WRPS/talent/SRS inputs.

---

## In-Progress Changes *(updated 2025-10-16)*

- [ ] Snapshot weekly pre-kick predictions into a longitudinal history table so we can audit drift and realized error week-to-week.
- [ ] Extend CFBD stat ingestion with situational splits (3rd/4th down, red zone) and rolling deltas; add regression tests around the new payloads.
- [ ] Produce `raw_positional_ratings` derived from roster usage + player metrics and merge into team inputs/model dataset.
- [ ] Document a troubleshooting playbook for deploy/validation failures (schedule stale, markets empty, cache mismatch) with pointers to logs and artifacts.

When a session is interrupted, resume with the first unchecked item unless context dictates otherwise. Update this checklist as work progresses.

---

## Operational Notes

### Market sourcing & fallbacks
- Primary feed is FanDuel via Odds API (`MARKET_SOURCE=fanduel`). `write_dataset` fills `market_spread_book` from FanDuel (`market_debug`), then CFBD odds, then legacy schedule columns, finally the model baseline. Rows filled without a bookmaker line set `market_is_synthetic=1`.
- `market_spread_source` is normalised (`fanduel`, `cfbd`, `model`, `unknown`) and forced to `fanduel` whenever a numeric FanDuel spread matches the stored value within tolerance.
- `expected_market_spread_book = λ*market + (1-λ)*model` (λ=0.6). Edge/value thresholds in both Python and UI share the same defaults; override via env (`EXPECTED_MARKET_LAMBDA`, `EDGE_POINTS_QUALIFY_MIN`, etc.).
- `model_result` / `expected_result` are derived in `_apply_book_grades` using final scores from the scoreboard snapshot so accuracy tables stay in sync with predictions.

### Cache management
- CFBD cache path defaults to `.cache_cfbd/<year>` with a soft ~90 day TTL; odds cache lives at `.cache_odds/<year>` with a two-day TTL (`ODDS_CACHE_TTL_DAYS`). Both are configurable via env in workflows.
- Deploy workflow can optionally restore caches (manual dispatch with `enable_cache=true`). Purging caches in CI requires bumping the cache key version (`CACHE_VERSION`) or deleting the directories before restoring.
- Manual reset: `rm -rf .cache_cfbd/2025 .cache_odds/2025` before re-running collectors. Backtest caches are year-scoped—avoid deleting other seasons unless you intend to refresh them.

### Validation & monitoring
- `tools.validate_site_data` fails builds if required tables/JSON are empty (schedule, market_debug, predictions, status).
- `python -m pytest` covers collectors, market joins, residual model, and helper utilities (see `tests/collect/`).
- Status tab exposes generated time, next-run ETA, dataset counts, and download links for `upa_data.sqlite` and `collector.log`.
- Monitor `market_predictions_backfill.json` to ensure FanDuel coverage; synthetic rate creeping upward usually means caches are stale or joins are failing.

---

## Metrics & Definitions (Spread/Prediction Datasets)

All spreads use bookmaker sign (negative favours the home team). Metrics live primarily in `upa_predictions` and mirrored backtest tables.

| Metric | Type | Definition / Calculation | Primary Usage |
| --- | --- | --- | --- |
| `week` | int | CFBD week number associated with the matchup. | Tab filtering, weekly accuracy tables. |
| `model_spread_baseline` | float | Rating differential (team inputs + home field) before any market anchoring. | Diagnose residual adjustments; fallback when markets missing. |
| `market_spread_book` | float | Latest bookmaker line after FanDuel/CFBD joins and backfill logic. | Canonical market comparison for edges/value; displayed in UI. |
| `market_spread_fanduel` | float | Raw FanDuel spread (when available). | Audit FanDuel coverage; forces `market_spread_source="fanduel"` when equal. |
| `market_spread_cfbd` | float | CFBD odds API spread normalised to book sign. | Fallback when FanDuel absent; surfaced in debug downloads. |
| `market_spread_source` | string | Origin tag for `market_spread_book` (`fanduel`, `cfbd`, `model`, `unknown`). | Status tab transparency, filtering Bets tab by source. |
| `market_spread_effective` | float | Spread actually used in calculations (`market_spread_book` or model fallback). | Ensures calculations stay defined even when market is synthetic. |
| `model_spread_book` | float | Baseline + residual adjustment anchored to `market_spread_book`; falls back to baseline when market missing. | Model price shown in UI; feeds edge/value math. |
| `market_adjustment` | float | Calibrated residual adjustment applied to the market (clipped ±8). | Explains deviation between market and model. |
| `expected_market_spread_book` | float | Blended expectation: `λ*market_spread_book + (1-λ)*model_spread_book` (λ=0.6). | Value calculation (`value_points_book`); dampens market noise. |
| `edge_points_book` | float | `model_spread_book - market_spread_effective`. Positive means model likes away team. | Core signal for Bets/Predictions tabs; qualification input. |
| `value_points_book` | float | `market_spread_effective - expected_market_spread_book`. | Indicates how far the book is from expectation. |
| `model_confidence` | float | Confidence score (0–1) from residual ensemble, scaled by source and synthetic status. | Bets tab confidence buckets; qualification gating. |
| `qualified_edge_flag` | int (0/1) | 1 when edge/value/confidence thresholds and directional agreement are met. | Drives recommended plays; surfaced in predictions download. |
| `market_is_synthetic` | int (0/1) | 1 when a bookmaker price was not available (legacy/model backfill). | Used to discount confidence and track synthetic rate on Status tab. |
| `played` | int (0/1) | 1 when both teams have final scores (merged from ESPN scoreboard). | Filters historical accuracy/MAE calculations. |
| `model_result` | string | `"CORRECT"`, `"INCORRECT"`, `"P"`, or empty, graded via `_apply_book_grades`. | Weekly win/loss tables, accuracy reporting. |
| `expected_result` | string | Grade for the `expected` pick side (same grading as `model_result`). | Backtest diagnostics; rarely surfaced in UI. |
| `availability_*_score` | float | Team availability metrics (0–100) derived from CFBD player usage by unit (overall/offense/defense/ST/QB). | Residual feature inputs; surfaced in team inputs datasets. |
| `availability_flag_qb_low` | int | 1 when quarterback usage drops below threshold. | Reduces confidence; displayed in debug exports. |
| `stat_off_ppd_adj` / `stat_def_ppd_adj` | float | Opponent-adjusted points per drive (scaled 0–100). | Captures drive efficiency beyond raw scoring rates. |
| `stat_off_pd_success` / `stat_def_pd_success` | float | Passing-down success splits (scaled; defense inverted). | Highlights situational strengths/weaknesses. |
| `stat_off_havoc_front` / `stat_def_havoc_front` | float | Havoc pressure (front-seven) rates from advanced stats. | Signals disruptive units impacting spread error. |
| `rolling_off_ppd_4` / `rolling_def_ppd_4` | float | Opponent-adjusted 4-game rolling points per drive (prior games). | Adds recent form trends to the residual adjustment. |
| `availability_off_top3_usage_pct` | float | Share of total offensive usage accounted for by top 3 contributors. | Flags concentration risk when key players are unavailable. |
| `fpi_rating_0_100` / `elo_rating_0_100` | float | FPI and Elo ratings rescaled to 0–100. | Blends additional power models alongside SP+. |
| `confidence_calibrated` | float | Conservative probability after calibrating `model_confidence` with historical weeks ≥ 4, spread bands, and market source penalties. | Drives bet gating and mirrors what shows in recommendations. |
| `confidence_play_flag` | int (0/1) | 1 when `confidence_calibrated ≥ 0.62` and the row is qualified. | Quickly filters the bet slate in downstream tooling. |

For full column lists see `agents/collect/predictions.py` (final `cols` array) and `tests/collect/test_predictions.py` for assertions.

---

## Local Development

- **Run collector locally** (requires `CFBD_BEARER_TOKEN`, optionally `ODDS_API_KEY`):  
  ```bash
  python -m agents.collect_cfbd_all --market-source fanduel --year 2025
  ```
- **Fetch FanDuel snapshot manually**:  
  ```bash
  python -m tools.fetch_fanduel_odds --year 2025
  ```
- **Rebuild backtest dataset**:  
  ```bash
  python -m tools.build_backtest_data --year 2024 --offline  # drop --offline to hit APIs
  ```
- **Run unit tests**: `python -m pytest --maxfail=1 --disable-warnings`
- **Start UI locally**: `npm install` (or `npm ci`) then `npm run dev`

Environment shortcuts (see `agents/collect/config.py` for defaults): `MARKET_SOURCE`, `CACHE_DIR`, `ODDS_CACHE_DIR`, `EXPECTED_MARKET_LAMBDA`, `EDGE_POINTS_QUALIFY_MIN`, `HIGH_CONFIDENCE_MIN`, `FANDUEL_CACHE_ONLY`, `INCLUDE_GRADE_FEATURES`.

---

## Open Items / Near-Term TODO

- Snapshot weekly pre-kick predictions into a history table (year/week partition) to support drift analysis and long-term MAE tracking.
- Expand CFBD feature ingestion with situational splits (3rd/4th down, red zone) and rolling deltas; add regression tests around stat normalisation.
- Produce positional rating datasets (`raw_positional_ratings`) that combine roster/usage data for richer unit grades.
- Expose rolling baseline metrics and alerting hooks on the Status tab (drift thresholds, synthetic rate, cache staleness).
- Document troubleshooting steps for deploy failures triggered by validation scripts (e.g., schedule stale, market_debug empty) and add quick links to relevant logs/artifacts.

Keep this list current—each session should either address an item or update it with new context.
