
# Unified Predictive Algorithm — Full Repo (with Predictions)

This repo bundles:
- **Dashboard** (Vite + React + TS) with tabs: **Status**, **Teams**, **Predictions**, **Live Edge**
- **Collector** (Python via GitHub Actions) that writes:
  - `data/upa_team_inputs_datadriven_v0.csv`
  - `data/status.json`
  - (if `data/cfb_schedule.csv` exists) `data/upa_predictions.csv` and `data/live_edge_report.csv`
- **Pages deploy** integrated into the same workflow

## One-time setup
1. Create a repo named **UnifiedPredictiveAlgorithm** (or update `vite.config.ts` `base` to match your repo name).
2. Unzip this folder into a clean repo directory.
3. Push:
   ```bash
   git init
   git add .
   git commit -m "init: UPA-F full repo (collector + dashboards + pages)"
   git branch -M main
   git remote add origin https://github.com/coxryan/UnifiedPredictiveAlgorithm.git
   git push -u origin main
   ```
4. In **Settings → Secrets and variables → Actions**, add:
   - Secret **BEARER_TOKEN** = your CollegeFootballData API token
   - *(Optional)* Secret **GOOGLE_SERVICE_ACCOUNT_JSON** = full JSON (for Sheets upsert)
   - *(Optional)* Variable **SHEET_ID** = your Sheet ID

## Provide a schedule (optional but recommended)
Place `data/cfb_schedule.csv` with columns such as:
```
week,date,home_team,away_team,neutral_site,market_spread
```
If present, the collector will emit `data/upa_predictions.csv` + `data/live_edge_report.csv` every run.

## Run
- Go to **Actions → Agent Collect & Update CSVs (All FBS) → Run workflow**.
- Pages will deploy to: `https://<your-username>.github.io/<repo-name>/` (base is set for `UnifiedPredictiveAlgorithm`).

## Local dev
```bash
npm install
npm run dev
```
