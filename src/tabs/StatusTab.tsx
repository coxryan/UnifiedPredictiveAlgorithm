import { useEffect, useMemo, useState } from "react";
import { loadText, loadCsv, fmtNum, fmtPct01, toNum, playedBool } from "../lib/csv";
import { Badge } from "../lib/ui";

type Status = {
  generated_at_utc: string;
  year: number;
  teams: number;
  games?: number;
  pred_rows?: number;
  next_run_eta_utc: string;

  // market info written by the collector
  market_source?: string;           // (back-compat) the source actually used
  market_source_used?: string;      // preferred: the source actually used
  market_source_config?: string;    // what we requested (env/flag), may differ if fallback happened
  market_fallback_reason?: string;  // <-- NEW: human-readable reason when a fallback occurred
};

type PredRow = { week: string; market_spread_book?: string; model_spread_book?: string; played?: any; model_result?: string; };

export default function StatusTab() {
  const [status, setStatus] = useState<Status | null>(null);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [err, setErr] = useState<string>("");

  useEffect(() => { (async () => {
    try { setStatus(JSON.parse(await loadText("data/status.json"))); }
    catch (e: any) { setErr(String(e?.message || e)); }
    try { setPreds(await loadCsv("data/upa_predictions.csv") as PredRow[]); } catch {}
  })(); }, []);

  const mae = useMemo(() => {
    if (!preds.length) return { overall: NaN, byBucket: [] as {bucket:string, mae:number}[], lastWeek: NaN };
    const mask = preds.filter(r => r.market_spread_book && r.model_spread_book);
    const diffs = mask.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
    const overall = diffs.length ? diffs.reduce((a,b)=>a+b,0)/diffs.length : NaN;

    const buckets = [
      { name: "0–3", min: 0, max: 3 },
      { name: "3–7", min: 3, max: 7 },
      { name: "7–14", min: 7, max: 14 },
      { name: "14+", min: 14, max: 999 },
    ].map(b => {
      const rows = mask.filter(r => {
        const m = Math.abs(toNum(r.market_spread_book));
        return Number.isFinite(m) && m >= b.min && m < b.max;
      });
      const err = rows.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
      const m = err.length ? err.reduce((a,v)=>a+v,0)/err.length : NaN;
      return { bucket: b.name, mae: m };
    });

    const played = preds.filter(r => playedBool(r.played));
    const lastWk = played.length ? Math.max(...played.map(r => Number(r.week))) : NaN;
    const lastRows = mask.filter(r => Number(r.week) === lastWk);
    const lastDiffs = lastRows.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
    const lastWeek = lastDiffs.length ? lastDiffs.reduce((a,b)=>a+b,0)/lastDiffs.length : NaN;

    return { overall, byBucket: buckets, lastWeek };
  }, [preds]);

  const weekly = useMemo(() => {
    const by: Record<string, {w:number,l:number,p:number}> = {};
    for (const r of preds) {
      if (!playedBool(r.played)) continue;
      const wk = r.week;
      const res = r.model_result;
      if (!by[wk]) by[wk] = { w:0, l:0, p:0 };
      if (res === "CORRECT") by[wk].w++;
      else if (res === "INCORRECT") by[wk].l++;
      else by[wk].p++;
    }
    const rows = Object.keys(by).sort((a,b)=>Number(a)-Number(b)).map(wk => {
      const {w,l,p} = by[wk];
      const tot = w + l; const acc = tot ? w/tot : NaN;
      return { week: wk, wins: w, losses: l, pushes: p, accuracy: acc };
    });
    return rows;
  }, [preds]);

  // Normalize raw market strings to pretty labels
  const prettyMarket = (s?: string) => {
    const t = (s || "").trim().toLowerCase();
    if (!t) return "";
    if (t === "fanduel") return "FanDuel";
    if (t === "cfbd") return "CFBD";
    return s || "";
  };

  // Show both requested and used values; flag a fallback if they differ
  const { marketCombined, marketUsed, marketRequested, marketMismatch } = useMemo(() => {
    const usedRaw = (status?.market_source_used ?? status?.market_source) || "";
    const reqRaw = status?.market_source_config || "";

    const used = prettyMarket(usedRaw);
    const req = prettyMarket(reqRaw);

    const mismatch =
      !!used && !!req && used.toLowerCase() !== req.toLowerCase();

    const combined =
      mismatch ? `${used} (fallback from ${req})`
      : (used || req || "CFBD");

    return {
      marketCombined: combined,
      marketUsed: used,
      marketRequested: req,
      marketMismatch: mismatch,
    };
  }, [status]);

  // Cache-busting query string for download links based on status timestamp
  const v = status?.generated_at_utc ? `?v=${encodeURIComponent(status.generated_at_utc)}` : "";

  return (
    <section className="card">
      <div className="card-title">Collector Status</div>
      {!status ? (
        <div className="note">{err ? `Status unavailable (${err})` : "Loading…"}</div>
      ) : (
        <>
          <div className="grid2">
            <div className="kv"><div className="k">Last updated</div><div className="v">{status.generated_at_utc}</div></div>
            <div className="kv"><div className="k">Next run ETA</div><div className="v">{status.next_run_eta_utc}</div></div>
            <div className="kv"><div className="k">Season</div><div className="v">{status.year}</div></div>

            <div className="kv">
              <div className="k">Market source (used)</div>
              <div className="v">
                {marketCombined}
                {marketMismatch ? <Badge tone="warn" style={{ marginLeft: 8 }}>fallback</Badge> : null}
              </div>
            </div>

            <div className="kv">
              <div className="k">Requested market</div>
              <div className="v">{marketRequested || "—"}</div>
            </div>

            {/* NEW: explicit fallback reason */}
            <div className="kv">
              <div className="k">Fallback reason</div>
              <div className="v">{status.market_fallback_reason ? status.market_fallback_reason : "—"}</div>
            </div>

            <div className="kv"><div className="k">Teams</div><div className="v">{status.teams}</div></div>
            {!!status.games && <div className="kv"><div className="k">Games</div><div className="v">{status.games}</div></div>}
            {!!status.pred_rows && <div className="kv"><div className="k">Pred rows</div><div className="v">{status.pred_rows}</div></div>}
          </div>

          <div className="subcards">
            <div className="subcard">
              <div className="subcard-title">MAE (Model vs Market)</div>
              <div className="kv"><div className="k">Overall</div><div className="v">{fmtNum(mae.overall)}</div></div>
              <div className="kv"><div className="k">Last completed week</div><div className="v">{fmtNum(mae.lastWeek)}</div></div>
              <div className="kv"><div className="k">By market size</div>
                <div className="v">
                  {mae.byBucket.map(b => <span key={b.bucket} style={{marginRight:8}}><b>{b.bucket}</b>: {fmtNum(b.mae)}</span>)}
                </div>
              </div>
            </div>

            <div className="subcard">
              <div className="subcard-title">Weekly Accuracy (Model)</div>
              {!weekly.length ? <div className="note">No completed games yet.</div> : (
                <div className="table-wrap">
                  <table className="tbl compact">
                    <thead><tr><th>Week</th><th>W</th><th>L</th><th>Push</th><th>Accuracy</th></tr></thead>
                    <tbody>
                      {weekly.map(r => (
                        <tr key={r.week}>
                          <td>{r.week}</td><td>{r.wins}</td><td>{r.losses}</td><td>{r.pushes}</td><td>{fmtPct01(r.accuracy)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>

          <div className="note" style={{ marginTop: 8 }}>
            <a href={`data/upa_team_inputs_datadriven_v0.csv${v}`} target="_blank" rel="noreferrer">team inputs CSV ↗</a>{" • "}
            <a href={`data/cfb_schedule.csv${v}`} target="_blank" rel="noreferrer">schedule CSV ↗</a>{" • "}
            <a href={`data/upa_predictions.csv${v}`} target="_blank" rel="noreferrer">predictions CSV ↗</a>{" • "}
            <a href={`data/live_edge_report.csv${v}`} target="_blank" rel="noreferrer">live edge CSV ↗</a>{" • "}
            <a href={`data/market_unmatched.csv${v}`} target="_blank" rel="noreferrer">unmatched market CSV ↗</a>{" • "}
            <a href={`data/2024/upa_predictions_2024_backtest.csv${v}`} target="_blank" rel="noreferrer">backtest predictions (2024) CSV ↗</a>{" • "}
            <a href={`data/2024/backtest_summary_2024.csv${v}`} target="_blank" rel="noreferrer">backtest summary (2024) CSV ↗</a>{" • "}
            <a href={`data/backtest_predictions_2024.csv${v}`} target="_blank" rel="noreferrer">alt: backtest predictions (2024) CSV ↗</a>
          </div>
        </>
      )}
    </section>
  );
}