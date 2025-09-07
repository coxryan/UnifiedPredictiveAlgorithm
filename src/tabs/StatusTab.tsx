import React, { useEffect, useState } from "react";

interface StatusJson {
  // new schema
  last_updated?: string;
  next_run_eta?: string;
  market_source_used?: string;
  requested_market?: string;

  // legacy/alt schema
  generated_at_utc?: string;
  next_run_eta_utc?: string;
  market_source?: string;
  market_requested?: string;

  // common
  season?: string | number;
  teams?: number;
  games?: number;
  pred_rows?: number;
}

/** Build a URL that respects Vite/GitHub Pages BASE_URL. */
function withBase(path: string): string {
  const base = (import.meta as any)?.env?.BASE_URL ?? "/";
  const b = String(base).replace(/\/+$/, "");
  const p = String(path).replace(/^\/+/, "");
  return `${b}/${p}`;
}

/** Try a list of URLs in order until one succeeds (non-404). */
async function fetchJsonWithFallback<T = any>(urls: string[]): Promise<T> {
  let lastErr: any;
  for (const url of urls) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) {
        lastErr = new Error(`HTTP ${res.status} for ${url}`);
        continue;
      }
      return (await res.json()) as T;
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr ?? new Error("Failed to load JSON");
}

async function headExists(urls: string[]): Promise<boolean> {
  for (const url of urls) {
    try {
      const res = await fetch(url, { method: "HEAD", cache: "no-store" });
      if (res.ok) return true;
    } catch {
      // continue
    }
  }
  return false;
}

/** Friendly date/time. */
function formatTimestamp(ts?: string): string {
  if (!ts) return "";
  const d = new Date(ts);
  return isNaN(d.getTime()) ? ts : d.toLocaleString();
}

/** Field pickers (support both old and new keys) */
function pickLastUpdated(s: StatusJson) {
  return s.last_updated ?? s.generated_at_utc ?? "";
}
function pickNextEta(s: StatusJson) {
  return s.next_run_eta ?? s.next_run_eta_utc ?? "";
}
function pickRequestedMarket(s: StatusJson) {
  return s.requested_market ?? s.market_requested ?? "-";
}
function pickMarketUsed(s: StatusJson) {
  return s.market_source_used ?? s.market_source ?? "-";
}

const STATUS_PRIMARY = withBase("data/status.json");
const STATUS_FALLBACK = "data/status.json";

const UNMATCHED_CSV_PRIMARY = withBase("data/unmatched_odds.csv");
const UNMATCHED_CSV_FALLBACK = "data/unmatched_odds.csv";

const StatusTab: React.FC = () => {
  const [status, setStatus] = useState<StatusJson | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [csvAvailable, setCsvAvailable] = useState<boolean>(false);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setLoading(true);
        // Try base-aware then relative
        const s = await fetchJsonWithFallback<StatusJson>([
          STATUS_PRIMARY,
          STATUS_FALLBACK,
        ]);
        if (!cancelled) {
          setStatus(s);
          setLoading(false);
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.message || "Failed to load status.");
          setLoading(false);
        }
      }

      // Probe CSV existence
      const csvExists = await headExists([UNMATCHED_CSV_PRIMARY, UNMATCHED_CSV_FALLBACK]);
      if (!cancelled) setCsvAvailable(csvExists);
    })();

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

  const lastUpdated = pickLastUpdated(status);
  const nextEta = pickNextEta(status);
  const requestedMarket = pickRequestedMarket(status);
  const marketUsed = pickMarketUsed(status);

  const csvUrl = csvAvailable ? UNMATCHED_CSV_PRIMARY : UNMATCHED_CSV_FALLBACK;

  return (
    <div className="card" style={{ maxWidth: 600, margin: "2rem auto", padding: "1.5rem" }}>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Last updated:</strong></label>
        <span style={{ marginLeft: 8 }}>{formatTimestamp(lastUpdated)}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Next run ETA:</strong></label>
        <span style={{ marginLeft: 8 }}>{formatTimestamp(nextEta)}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Season:</strong></label>
        <span style={{ marginLeft: 8 }}>{status.season ?? "-"}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Requested market:</strong></label>
        <span style={{ marginLeft: 8 }}>{requestedMarket}</span>
      </div>
      <div className="row" style={{ marginBottom: 12 }}>
        <label><strong>Market source (used):</strong></label>
        <span style={{ marginLeft: 8 }}>{marketUsed}</span>
      </div>
      {status["fallback_reason"] && String(status["fallback_reason"]).trim().length > 0 && (
        <div className="row" style={{ marginBottom: 12 }}>
          <label><strong>Fallback reason:</strong></label>
          <span style={{ marginLeft: 8 }}>{status["fallback_reason"]}</span>
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
                href={csvUrl}
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