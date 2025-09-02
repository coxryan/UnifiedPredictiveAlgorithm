import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, fmtPct01, toNum, playedBool } from "../lib/csv";
import { Badge } from "../lib/ui";

type PredRow = {
  week: string; date: string; away_team: string; home_team: string;
  model_spread_book?: string; market_spread_book?: string;
  edge_points_book?: string; qualified_edge_flag?: string;
  played?: any; model_result?: string;
};

export default function BacktestTab() {
  const [summary,setSummary] = useState<any[]>([]);
  const [preds,setPreds] = useState<PredRow[]>([]);
  const [week,setWeek] = useState<string>("ALL");

  useEffect(()=>{(async()=>{
    try{ setSummary(await loadCsv("data/2024/backtest_summary_2024.csv")); }catch{ setSummary([]); }
    try{ setPreds(await loadCsv("data/2024/upa_predictions_2024_backtest.csv") as PredRow[]); }catch{ setPreds([]); }
  })();},[]);

  const weeks = useMemo(()=>{
    const s = new Set<string>();
    if (summary.length) summary.forEach(r => s.add(String(r.week)));
    if (!s.size && preds.length) preds.forEach(r => s.add(String(r.week)));
    if (!s.has("ALL")) s.add("ALL");
    return Array.from(s).sort((a,b)=>{
      if (a==="ALL") return -1;
      if (b==="ALL") return 1;
      return Number(a)-Number(b);
    });
  }, [summary, preds]);

  const counts = useMemo(()=>{
    const rows = week==="ALL" ? preds : preds.filter(r => String(r.week)===String(week));
    let w=0,l=0,p=0;
    for (const r of rows) {
      if (!playedBool(r.played)) continue;
      const res = r.model_result;
      if (res === "CORRECT") w++;
      else if (res === "INCORRECT") l++;
      else p++;
    }
    const tot = w + l; const hit = tot ? w / tot : NaN;
    return { w,l,p, hit, rows: rows.filter(r=>playedBool(r.played)).length };
  }, [preds, week]);

  const tableRows = useMemo(()=> (week==="ALL" ? preds : preds.filter(r=>String(r.week)===String(week))), [preds, week]);

  return (
    <section className="card">
      <div className="card-title">Backtest — 2024 (Weeks 1–15)</div>
      <div className="controls">
        <label>Week
          <select className="input" value={week} onChange={(e)=>setWeek(e.target.value)}>
            {weeks.map(w=><option key={w} value={w}>{w}</option>)}
          </select>
        </label>
        <Badge tone="muted">Games: {fmtNum(counts.rows)}</Badge>
        <Badge tone="pos">W: {counts.w}</Badge>
        <Badge tone="neg">L: {counts.l}</Badge>
        <Badge tone="muted">Push: {counts.p}</Badge>
        <Badge>Hit: {fmtPct01(counts.hit)}</Badge>
      </div>

      <div className="table-wrap">
        <table className="tbl compact">
          <thead>
            <tr>
              <th>Week</th><th>Date</th><th>Away</th><th>Home</th>
              <th>Model (H)</th><th>Market (H)</th><th>Edge</th><th>Qualified</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r,i)=>(
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i%2?"alt":undefined}>
                <td>{r.week}</td><td>{r.date}</td><td>{r.away_team}</td><td>{r.home_team}</td>
                <td>{fmtNum(r.model_spread_book)}</td>
                <td>{fmtNum(r.market_spread_book)}</td>
                <td className={Number(toNum(r.edge_points_book))>0?"pos":"neg"}>{fmtNum(r.edge_points_book)}</td>
                <td>{r.qualified_edge_flag==="1" ? "✓" : "—"}</td>
                <td>{r.model_result==="CORRECT" ? "✓ Model" : r.model_result==="INCORRECT" ? "✗ Model" : "—"}</td>
              </tr>
            ))}
            {!tableRows.length && <tr><td colSpan={9} style={{textAlign:"center",padding:12}}>No backtest data found.</td></tr>}
          </tbody>
        </table>
      </div>

      <div className="note" style={{marginTop:8}}>
        Source: <a href="data/2024/upa_predictions_2024_backtest.csv" target="_blank" rel="noreferrer">predictions</a> •{" "}
        <a href="data/2024/backtest_summary_2024.csv" target="_blank" rel="noreferrer">summary</a>
      </div>
    </section>
  );
}