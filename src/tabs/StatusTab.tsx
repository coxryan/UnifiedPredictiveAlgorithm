import { useEffect, useMemo, useState } from "react";
import { loadJson, loadTable, fmtNum, fmtPct01, toNum, playedBool } from "../lib/csv";
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

const SPREAD_BANDS = [
  { label: "<3", min: 0, max: 3 },
  { label: "3-5", min: 3, max: 5 },
  { label: "5-10", min: 5, max: 10 },
  { label: "10-15", min: 10, max: 15 },
  { label: "15-20", min: 15, max: 20 },
  { label: "20+", min: 20, max: Number.POSITIVE_INFINITY },
];

export default function StatusTab() {
  const [status, setStatus] = useState<Status | null>(null);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [err, setErr] = useState<string>("");

  useEffect(() => { (async () => {
    try { setStatus((await loadJson("data/status.json")) as Status); }
    catch (e: any) { setErr(String(e?.message || e)); }
    try { setPreds(await loadTable("upa_predictions") as PredRow[]); } catch {}
  })(); }, []);

  const mae = useMemo(() => {
    if (!preds.length) return { overall: NaN, byBucket: [] as {bucket:string, mae:number}[], lastWeek: NaN };
    const mask = preds.filter(r =>
      Number.isFinite(toNum(r.market_spread_book)) &&
      Number.isFinite(toNum(r.model_spread_book))
    );
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
    const by: Record<string, {w:number,l:number,p:number, bands: Record<string,{wins:number,total:number}>}> = {};
    for (const r of preds) {
      if (!playedBool(r.played)) continue;
      const wk = r.week;
      const res = r.model_result;
      if (!by[wk]) {
        const bandInit: Record<string,{wins:number,total:number}> = {};
        for (const band of SPREAD_BANDS) bandInit[band.label] = { wins: 0, total: 0 };
        by[wk] = { w:0, l:0, p:0, bands: bandInit };
      }
      let bandLabel: string | null = null;
      const marketAbs = Math.abs(toNum(r.market_spread_book));
      if (Number.isFinite(marketAbs)) {
        const band = SPREAD_BANDS.find(b => marketAbs >= b.min && marketAbs < b.max);
        if (band) bandLabel = band.label;
      }
      if (res === "CORRECT") by[wk].w++;
      else if (res === "INCORRECT") by[wk].l++;
      else by[wk].p++;
      if (bandLabel) {
        const bucket = by[wk].bands[bandLabel];
        if (res === "CORRECT") {
          bucket.wins += 1;
          bucket.total += 1;
        } else if (res === "INCORRECT") {
          bucket.total += 1;
        }
      }
    }
    const rows = Object.keys(by).sort((a,b)=>Number(a)-Number(b)).map(wk => {
      const {w,l,p,bands} = by[wk];
      const tot = w + l; const acc = tot ? w/tot : NaN;
      const bandRates = SPREAD_BANDS.map(b => {
        const stats = bands[b.label];
        const rate = stats.total ? stats.wins / stats.total : NaN;
        return { label: b.label, wins: stats.wins, total: stats.total, rate };
      });
      return { week: wk, wins: w, losses: l, pushes: p, accuracy: acc, bandRates };
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
                    <thead><tr><th>Week</th><th>W</th><th>L</th><th>Push</th><th>Accuracy</th><th>Spread Bands</th></tr></thead>
                    <tbody>
                      {weekly.map(r => (
                        <tr key={r.week}>
                          <td>{r.week}</td>
                          <td>{r.wins}</td>
                          <td>{r.losses}</td>
                          <td>{r.pushes}</td>
                          <td>{fmtPct01(r.accuracy)}</td>
                          <td>
                            {r.bandRates.some(b => b.total > 0)
                              ? r.bandRates.filter(b => b.total > 0).map(b => (
                                  <span key={b.label} style={{ marginRight: 8 }}>
                                    <b>{b.label}</b>: {fmtPct01(b.rate)}
                                  </span>
                                ))
                              : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="subcard">
              <div className="subcard-title">Downloads &amp; Debug</div>
              <div className="table-wrap">
                <table className="tbl compact">
                  <thead>
                    <tr><th>File</th><th>Description</th></tr>
                  </thead>
                  <tbody>
                    {[
                      { href: `data/upa_data.sqlite${v}`, label: "upa_data.sqlite ↗", desc: "Primary SQLite database containing predictions, schedule, markets, team inputs, status, live scores, and supporting tables." },
                      { href: `data/debug/collector.log${v}`, label: "collector.log ↗", desc: "Latest collector run log (DEBUG level)." },
                    ].map(item => (
                      <tr key={item.href}>
                        <td><a href={item.href} target="_blank" rel="noreferrer">{item.label}</a></td>
                        <td>{item.desc}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
