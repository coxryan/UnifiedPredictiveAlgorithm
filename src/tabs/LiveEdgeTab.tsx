import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum } from "../lib/csv";
import { Badge } from "../lib/ui";

export default function LiveEdgeTab() {
  const [rows, setRows] = useState<any[]>([]);
  const [q, setQ] = useState("");

  useEffect(()=>{(async()=>{
    try { setRows((await loadCsv("data/live_edge_report.csv")) as any[]); } catch { setRows([]); }
  })();},[]);

  const filtered = useMemo(()=>{
    const ql = q.trim().toLowerCase();
    if (!ql) return rows;
    return rows.filter(
      (r) => r.home_team?.toLowerCase().includes(ql) || r.away_team?.toLowerCase().includes(ql)
    );
  },[rows,q]);

  return (
    <section className="card">
      <div className="card-title">Live Edge (Book-style) — 2025</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e)=>setQ(e.target.value)} />
        <Badge>{filtered.length} rows</Badge>
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Wk</th><th>Date</th><th>Away</th><th>Home</th><th>Model (H)</th><th>Market (H)</th><th>Edge</th><th>Qualified</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r,i)=>(
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i%2?"alt":undefined}>
                <td>{r.week}</td><td>{r.date}</td><td>{r.away_team}</td><td>{r.home_team}</td>
                <td>{fmtNum(r.model_spread_book)}</td>
                <td>{fmtNum(r.market_spread_book)}</td>
                <td className={Number(r.edge_points_book)>0?"pos":"neg"}>{fmtNum(r.edge_points_book)}</td>
                <td>{r.qualified_edge_flag==="1"?"✓":"—"}</td>
              </tr>
            ))}
            {!filtered.length && <tr><td colSpan={8} style={{textAlign:"center",padding:12}}>No rows</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}