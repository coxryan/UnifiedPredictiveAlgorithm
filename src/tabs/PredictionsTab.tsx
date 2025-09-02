import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, scoreText } from "../lib/ui";

type PredRow = {
  game_id: string; week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; home_points?: string; away_points?: string; played?: any;
  market_spread_book?: string; model_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string; qualified_edge_flag?: string; model_result?: string;
};

export default function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [week, setWeek] = useState<string>("");

  useEffect(() => { (async () => {
    try {
      const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
      setRows(r);
      const last = r.length ? Math.max(...r.map(x => Number(x.week))) : null;
      setWeek(last ? String(last) : "");
    } catch { setRows([]); }
  })(); }, []);

  const weeks = useMemo(
    () => Array.from(new Set(rows.map((r) => r.week))).sort((a, b) => Number(a) - Number(b)),
    [rows]
  );
  const filtered = useMemo(() => rows.filter(r => String(r.week) === String(week)), [rows, week]);

  return (
    <section className="card">
      <div className="card-title">Predictions (Book-style) — 2025</div>
      <div className="controls">
        <label>Week
          <select className="input" value={week} onChange={(e)=>setWeek(e.target.value)}>
            {weeks.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
        <Badge>{filtered.length} rows</Badge>
      </div>

      <div className="table-wrap">
        <table className="tbl wide">
          <thead>
            <tr>
              <th>Wk</th><th>Date</th><th colSpan={2}>Matchup</th><th>Score (A @ H)</th>
              <th>Model (H)</th><th>Market (H)</th><th>Expected (H)</th><th>Edge</th><th>Value</th><th>Qualified</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const model = toNum(r.model_spread_book);
              const market = toNum(r.market_spread_book);
              const expected = toNum(r.expected_market_spread_book);
              const edge = toNum(r.edge_points_book);
              const value = toNum(r.value_points_book);
              const played = playedBool(r.played);
              const neutral = r.neutral_site === "1" || r.neutral_site === "true";
              const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
              const badge = played ? (
                r.model_result === "CORRECT" ? <Badge tone="pos">✓ Model</Badge> :
                r.model_result === "INCORRECT" ? <Badge tone="neg">✗ Model</Badge> : null
              ) : null;

              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td style={{ textAlign: "right" }}><TeamLabel home={false} team={r.away_team} neutral={false} /></td>
                  <td style={{ textAlign: "left" }}><TeamLabel home={true} team={r.home_team} neutral={!!neutral} /></td>
                  <td>{played ? scoreText(r.away_points, r.home_points) : "—"}</td>
                  <td>{fmtNum(model)}</td>
                  <td>{fmtNum(market)}</td>
                  <td>{fmtNum(expected)}</td>
                  <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                  <td className={Number.isFinite(value) ? (value > 0 ? "pos" : "neg") : undefined}>{fmtNum(value)}</td>
                  <td>{qual}</td>
                  <td>{badge}</td>
                </tr>
              );
            })}
            {!filtered.length && <tr><td colSpan={12} style={{ textAlign:"center", padding:12 }}>No rows</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}