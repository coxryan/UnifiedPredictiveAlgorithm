name: Deploy site

on:
  workflow_dispatch:
  push:
    branches: [ "main" ]
  schedule:
    # Nightly (UTC). Adjust as needed.
    - cron: "30 4 * * *"

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

env:
  # Optional – if not set, the collector will fall back internally.
  MARKET_SOURCE: ${{ vars.MARKET_SOURCE }}
  ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
  ODDS_CACHE_DIR: .cache_odds
  ODDS_CACHE_TTL_DAYS: 2

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Configure Pages
        uses: actions/configure-pages@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install -r agents/requirements.txt

      - name: Restore API caches (CFBD & Odds)
        uses: actions/cache@v4
        with:
          path: |
            .cache_cfbd
            .cache_odds
          key: api-cache-${{ runner.os }}-${{ hashFiles('agents/collect_cfbd_all.py') }}-${{ env.MARKET_SOURCE }}
          restore-keys: |
            api-cache-${{ runner.os }}-

      - name: Run collector (writes data/*.csv + status.json)
        run: |
          python agents/collect_cfbd_all.py

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: "20.x"

      - name: Install frontend deps
        run: npm install --no-audit --no-fund

      - name: Build site
        run: npm run build

      - name: Bundle data into site
        run: |
          mkdir -p dist/data
          cp -R data/* dist/data/ || true

      - name: Upload artifact for Pages
        uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
import React, { useEffect, useState } from "react";

interface StatusJson {
  last_updated?: string;
  next_run_eta?: string;
  season?: string | number;
  requested_market?: string;
  market_source_used?: string;
  fallback_reason?: string;
  teams?: number;
  games?: number;
  pred_rows?: number;
}

const STATUS_URL = "/data/status.json";
const UNMATCHED_ODDS_CSV_URL = "/data/unmatched_odds.csv";

function formatTimestamp(ts?: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return ts;
  return d.toLocaleString();
}

const StatusTab: React.FC = () => {
  const [status, setStatus] = useState<StatusJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [csvAvailable, setCsvAvailable] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(STATUS_URL)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load status (${res.status})`);
        return res.json();
      })
      .then((data: StatusJson) => {
        if (!cancelled) {
          setStatus(data);
          setLoading(false);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setError(e.message || "Failed to load status.");
          setLoading(false);
        }
      });
    // Check for unmatched_odds.csv
    fetch(UNMATCHED_ODDS_CSV_URL, { method: "HEAD" })
      .then((res) => {
        if (!cancelled) setCsvAvailable(res.ok);
      })
      .catch(() => {
        if (!cancelled) setCsvAvailable(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <div className="card"><div className="row">Loading status...</div></div>;
  }
  if (error) {
    return <div className="card"><div className="row">Error: {error}</div></div>;
  }
  if (!status) {
    return <div className="card"><div className="row">No status data found.</div></div>;
  }

  return (
    <div className="card" style={{ maxWidth: 600, margin: "2rem auto", padding: "1.5rem" }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Last updated:</strong></label>
        <span style={{ marginLeft: 8 }}>{formatTimestamp(status.last_updated)}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Next run ETA:</strong></label>
        <span style={{ marginLeft: 8 }}>{formatTimestamp(status.next_run_eta)}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Season:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.season ?? "-"}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Requested market:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.requested_market ?? "-"}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Market source used:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.market_source_used ?? "-"}</span>
      </div>
      {status.fallback_reason && status.fallback_reason.trim().length > 0 && (
        <div className="row" style={{ marginBottom: 12 }}>
          <label><strong>Fallback reason:</strong></label>
          <span style={{ marginLeft: 8 }}>{status.fallback_reason}</span>
        </div>
      )}
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Teams:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.teams ?? "-"}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Games:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.games ?? "-"}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Prediction rows:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.pred_rows ?? "-"}</span>
      </div>
      <div className="row" style={{ marginTop: 24 }}>
        <div style={{ borderTop: "1px solid #333", paddingTop: 12, width: "100%" }}>
          <strong>Diagnostics</strong>
          <div style={{ marginTop: 8 }}>
            {csvAvailable ? (
              <a
                href={UNMATCHED_ODDS_CSV_URL}
                download
                className="download-link"
                style={{ color: "#4af", textDecoration: "underline" }}
              >
                [Download unmatched odds CSV]
              </a>
            ) : (
              <span style={{ color: "#888" }}>[Unmatched odds CSV not available]</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatusTab;