import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, fmtPct01, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel } from "../lib/ui";

type PredRow = {
  week: string; date: string; away_team: string; home_team: string;
  model_spread_book?: string; market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string;
  qualified_edge_flag?: string; played?: any; model_result?: string;
  home_rank?: string; away_rank?: string;
};

async function loadFirst(paths: string[]) {
  for (const p of paths) {
    try {
      const rows = await loadCsv(p);
      if (rows && rows.length) return rows;
    } catch {}
  }
  return [];
}

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome;
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function BacktestTab() {
  const [summary, setSummary] = useState<any[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [week, setWeek] = useState<string>("ALL");
  const [err, setErr] = useState<string>("");

  useEffect(() => { (async () => {
    try {
      const summaryRows = await loadFirst([
        "data/2024/backtest_summary_2024.csv",
        "data/backtest_summary_2024.csv",
      ]);
      setSummary(summaryRows);
    } catch (e:any) {
      setSummary([]); setErr(String(e?.message || e));
    }

    try {
      const predRows = await loadFirst([
        "data/2024/upa_predictions_2024_backtest.csv",
        "data/2024/backtest_predictions_2024.csv",
        "data/backtest_predictions_2024.csv",
      ]) as PredRow[];
      setPreds(predRows);
    } catch (e:any) {
      setPreds([]); setErr(prev => prev || String(e?.message || e));
    }
  })(); }, []);

  const weeks = useMemo(() => {
    const s = new Set<string>();
    (summary || []).forEach((r:any) => s.add(String(r.week)));
    if (!s.size) (preds || []).forEach((r:any) => s.add(String(r.week)));
    s.add("ALL");
    return Array.from(s).sort((a,b) => {
      if (a === "ALL") return -1;
      if (b === "ALL") return 1;
      return Number(a) - Number(b);
    });
  }, [summary, preds]);

  const counts = useMemo(() => {
    const rows = week === "ALL" ? preds : preds.filter(r => String(r.week) === String(week));
    let w=0,l=0,p=0;
    for (const r of rows) {
      if (!playedBool(r.played)) continue;
      const res = r.model_result;
      if (res === "CORRECT") w++;
      else if (res === "INCORRECT") l++;
      else p++;
    }
    const tot = w + l; const hit = tot ? w/tot : NaN;
    return { w,l,p, hit, rows: rows.filter(r => playedBool(r.played)).length };
  }, [preds, week]);

  const tableRows = useMemo(
    () => (week === "ALL" ? preds : preds.filter(r => String(r.week) === String(week))),
    [preds, week]
  ).map((r:any) => {
    const model = toNum(r.model_spread_book);
    const market = toNum(r.market_spread_book);
    const pick = valueSide(model, market, r.home_team, r.away_team);
    return { ...r, _model:model, _market:market, _pick: pick.side };
  });

  return (
    <section className="card">
      <div className="card-title">Backtest — 2024</div>
      <div className="controls">
        <label>Week
          <select className="input" value={week} onChange={(e)=>setWeek(e.target.value)}>
            {weeks.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
        <Badge tone="muted">Games: {fmtNum(counts.rows)}</Badge>
        <Badge tone="pos">W: {counts.w}</Badge>
        <Badge tone="neg">L: {counts.l}</Badge>
        <Badge tone="muted">Push: {counts.p}</Badge>
        <Badge>Hit: {fmtPct01(counts.hit)}</Badge>
      </div>

      {!preds.length && (
        <div className="note" style={{marginBottom:8}}>
          No backtest data found. Looking for: <code>data/2024/upa_predictions_2024_backtest.csv</code> or <code>data/backtest_predictions_2024.csv</code>.
        </div>
      )}

      <div className="table-wrap">
        <table className="tbl compact">
          <thead>
            <tr>
              <th>Week</th><th>Date</th><th colSpan={2}>Matchup</th>
              <th>Model (H)</th><th>Market (H)</th><th>Value Side</th>
              <th>Qualified</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r:any, i:number)=>(
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i%2?"alt":undefined}>
                <td>{r.week}</td>
                <td>{r.date}</td>
                <td style={{ textAlign: "right" }}>
                  <TeamLabel home={false} team={r.away_team} neutral={false} />
                  {Number.isFinite(Number(r.away_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.away_rank)}</span>
                  ) : null}
                </td>
                <td style={{ textAlign: "left" }}>
                  <TeamLabel home={true} team={r.home_team} neutral={false} />
                  {Number.isFinite(Number(r.home_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.home_rank)}</span>
                  ) : null}
                </td>
                <td>{fmtNum(r._model)}</td>
                <td>{fmtNum(r._market)}</td>
                <td>{r._pick}</td>
                <td>{r.qualified_edge_flag==="1" ? "✓" : "—"}</td>
                <td>{r.model_result==="CORRECT" ? "✓ Model" : r.model_result==="INCORRECT" ? "✗ Model" : "—"}</td>
              </tr>
            ))}
            {!tableRows.length && (
              <tr><td colSpan={9} style={{textAlign:"center",padding:12}}>No rows to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="note" style={{marginTop:8}}>
        <b>Value side</b> uses edge = model − market (home perspective). edge&gt;0 ⇒ market too heavy on home ⇒ value = away; edge&lt;0 ⇒ value = home.
      </div>
    </section>
  );
}