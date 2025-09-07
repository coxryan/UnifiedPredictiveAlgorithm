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

// Helper to resolve paths against Vite's BASE_URL (works on GH Pages)
// Ensures exactly one slash between base and path.
function withBase(path: string): string {
  const base = (import.meta as any)?.env?.BASE_URL ?? "/";
  const b = String(base).replace(/\/+$/, "");
  const p = String(path).replace(/^\/+/, "");
  return `${b}/${p}`;
}

const STATUS_URL = withBase("data/status.json");
const UNMATCHED_ODDS_CSV_URL = withBase("data/unmatched_odds.csv");

function formatTimestamp(ts?: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d.getTime()) ? ts : d.toLocaleString();
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

    // Probe for unmatched_odds.csv
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
    return (
      <div className="card">
        <div className="row">Loading status...</div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="card">
        <div className="row">Error: {error}</div>
      </div>
    );
  }
  if (!status) {
    return (
      <div className="card">
        <div className="row">No status data found.</div>
      </div>
    );
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