import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel, nextUpcomingWeek } from "../lib/ui";

// -----------------------------
// Types
// -----------------------------
type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string;
  expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string;
  qualified_edge_flag?: string;
  // optional ranks (if present)
  home_rank?: string; away_rank?: string;
  home_ap_rank?: string; away_ap_rank?: string;
  home_coaches_rank?: string; away_coaches_rank?: string;
  // optional scores (if present for settled games)
  home_points?: string; away_points?: string;
};

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [wk, setWk] = useState<number | null>(null);
  const [onlyQualified, setOnlyQualified] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      try {
        const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
        setRows(r);
        const nextWk = nextUpcomingWeek(r as any);
        // Default to upcoming if exists; otherwise to min available
        if (nextWk) setWk(nextWk);
        else {
          const w = Array.from(new Set(r.map(x => Number(x.week)).filter(x => Number.isFinite(x)))).sort((a,b)=>a-b);
          setWk(w.length ? w[0] : null);
        }
      } catch {
        setRows([]);
        setWk(null);
      }
    })();
  }, []);

  // available week list for dropdown
  const weekOptions = useMemo(() => {
    const w = Array.from(
      new Set(
        rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x))
      )
    ).sort((a, b) => a - b);
    return w as number[];
  }, [rows]);

  const tableRows = useMemo(() => {
    const filtered = rows.filter((r) => (wk ? Number(r.week) === wk : true));
    const f2 = onlyQualified ? filtered.filter((r:any)=> r.qualified_edge_flag === "1") : filtered;
    return f2.map((r:any) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const expected = toNum(r.expected_market_spread_book);
      const edge = Number.isFinite(toNum(r.edge_points_book))
        ? toNum(r.edge_points_book)
        : Number.isFinite(model) && Number.isFinite(market)
        ? model - market
        : NaN;
      const value = Number.isFinite(toNum(r.value_points_book))
        ? toNum(r.value_points_book)
        : Number.isFinite(market) && Number.isFinite(expected)
        ? market - expected
        : NaN;

      const pick = valueSide(model, market, r.home_team, r.away_team);
      const hp = toNum(r.home_points);
      const ap = toNum(r.away_points);
      const score = (Number.isFinite(hp) && Number.isFinite(ap)) ? `${fmtNum(ap,{maximumFractionDigits:0})} @ ${fmtNum(hp,{maximumFractionDigits:0})}` : "—";
      const finalDiff = (Number.isFinite(hp) && Number.isFinite(ap)) ? (hp - ap) : NaN; // home minus away

      return { ...r, _model:model, _market:market, _expected:expected, _edge:edge, _value:value, _pick: pick.side, _score: score, _finalDiff: finalDiff };
    });
  }, [rows, wk, onlyQualified]);

  return (
    <section className="card">
      <div className="card-title">Predictions — 2025</div>
      <div className="controls">
        <label>Week
          <select className="input" value={wk ?? ""} onChange={(e)=> setWk(e.target.value ? Number(e.target.value) : null)}>
            {(weekOptions.length ? weekOptions : [wk ?? 1]).map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </label>
        <label className="chk">
          <input type="checkbox" checked={onlyQualified} onChange={(e)=> setOnlyQualified(e.target.checked)} />
          Exclude non-qualified (show only ✓)
        </label>
        <Badge tone="muted">Rows: {tableRows.length}</Badge>
      </div>

      {!rows.length && (
        <div className="note" style={{marginBottom:8}}>No predictions found. Expecting <code>data/upa_predictions.csv</code>.</div>
      )}

      <div className="table-wrap">
        <table className="tbl wide">
          <thead>
            <tr>
              <th>Week</th>
              <th>Date</th>
              <th>Score (A @ H)</th>
              <th>Final (H)</th>
              <th colSpan={2}>Matchup</th>
              <th>Model (H)</th>
              <th>Market (H)</th>
              <th>Expected (H)</th>
              <th>Edge</th>
              <th>Value</th>
              <th>Qualified</th>
              {/* intentionally no Result column on Predictions per request */}
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r:any, i:number)=> (
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i%2?"alt":undefined}>
                <td>{r.week}</td>
                <td>{r.date}</td>
                <td>{r._score}</td>
                <td>{fmtNum(r._finalDiff)}</td>
                <td style={{ textAlign: "right" }}>
                  <TeamLabel home={false} team={r.away_team} neutral={false} />
                  {Number.isFinite(Number(r.away_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.away_rank)}</span>
                  ) : null}
                  {Number.isFinite(Number(r.away_ap_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.6, fontSize: "0.8em" }}>(AP #{Number(r.away_ap_rank)})</span>
                  ) : Number.isFinite(Number(r.away_coaches_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.6, fontSize: "0.8em" }}>(Coaches #{Number(r.away_coaches_rank)})</span>
                  ) : null}
                </td>
                <td style={{ textAlign: "left" }}>
                  <TeamLabel home={true} team={r.home_team} neutral={r.neutral_site === "1" || r.neutral_site === "true"} />
                  {Number.isFinite(Number(r.home_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.home_rank)}</span>
                  ) : null}
                  {Number.isFinite(Number(r.home_ap_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.6, fontSize: "0.8em" }}>(AP #{Number(r.home_ap_rank)})</span>
                  ) : Number.isFinite(Number(r.home_coaches_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.6, fontSize: "0.8em" }}>(Coaches #{Number(r.home_coaches_rank)})</span>
                  ) : null}
                </td>
                <td>{fmtNum(r._model)}</td>
                <td>{fmtNum(r._market)}</td>
                <td>{fmtNum(r._expected)}</td>
                <td className={Number.isFinite(r._edge) ? (r._edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._edge)}</td>
                <td className={Number.isFinite(r._value) ? (r._value > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._value)}</td>
                <td>{r.qualified_edge_flag === "1" ? "✓" : "—"}</td>
              </tr>
            ))}
            {!tableRows.length && (
              <tr><td colSpan={12} style={{textAlign:"center", padding:12}}>No rows to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads shown (negative = home favorite). <b>Edge</b> = model − market (home perspective). <b>Value</b> = market − expected.
        Games marked with a score are complete; otherwise they are upcoming or in-progress.
      </div>
    </section>
  );
}