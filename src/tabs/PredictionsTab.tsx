import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel } from "../lib/ui";
import { EDGE_MIN, VALUE_MIN } from "./constants";

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string;
  model_spread_book?: string; market_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string; qualified_edge_flag?: string;
};

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome;
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [week, setWeek] = useState<string>("ALL");
  const [conf, setConf] = useState<string>("All");

  useEffect(() => {
    (async () => {
      try {
        const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
        setRows(r);
      } catch {
        setRows([]);
      }
    })();
  }, []);

  const withComputed = useMemo(() => {
    return rows.map((r) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const expected = toNum(r.expected_market_spread_book);
      const edge = Number.isFinite(toNum(r.edge_points_book))
        ? toNum(r.edge_points_book)
        : (Number.isFinite(model) && Number.isFinite(market)) ? (model - market) : NaN;
      const value = Number.isFinite(toNum(r.value_points_book))
        ? toNum(r.value_points_book)
        : (Number.isFinite(market) && Number.isFinite(expected)) ? (market - expected) : NaN;
      const pick = valueSide(model, market, r.home_team, r.away_team);
      const neutral = r.neutral_site === "1" || r.neutral_site === "true";
      return { ...r, _model: model, _market: market, _edge: edge, _value: value, _pick: pick.side, _neutral: neutral };
    });
  }, [rows]);

  const weeks = useMemo(() => {
    const set = new Set<string>();
    rows.forEach((r) => set.add(String(r.week)));
    return ["ALL", ...Array.from(set).sort((a, b) => Number(a) - Number(b))];
  }, [rows]);

  const filtered = useMemo(() => {
    let r = withComputed;
    if (week !== "ALL") r = r.filter((x) => String(x.week) === week);
    if (conf !== "All") {
      // (Optional) if you add conference to the predictions CSV, filter here.
    }
    return r;
  }, [withComputed, week, conf]);

  return (
    <section className="card">
      <div className="card-title">Predictions — 2025</div>
      <div className="controls">
        <label>Week
          <select className="input" value={week} onChange={(e)=>setWeek(e.target.value)}>
            {weeks.map((w) => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
        <Badge tone="muted">Rows: {filtered.length}</Badge>
        <Badge>Qualified rule: |Edge| ≥ {EDGE_MIN}, |Value| ≥ {VALUE_MIN}, side agreement</Badge>
      </div>

      <div className="table-wrap">
        <table className="tbl wide">
          <thead>
            <tr>
              <th>Week</th>
              <th>Date</th>
              <th colSpan={2}>Matchup</th>
              <th>Model (H)</th>
              <th>Market (H)</th>
              <th>Expected (H)</th>
              <th>Edge</th>
              <th>Value</th>
              <th>Qualified</th>
              <th>Recommended Side (value)</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td style={{ textAlign: "right" }}>
                    <TeamLabel home={false} team={r.away_team} neutral={false} />
                  </td>
                  <td style={{ textAlign: "left" }}>
                    <TeamLabel home={true} team={r.home_team} neutral={r._neutral} />
                  </td>
                  <td>{fmtNum(r._model)}</td>
                  <td>{fmtNum(r._market)}</td>
                  <td>{fmtNum(toNum(r.expected_market_spread_book))}</td>
                  <td className={r._edge > 0 ? "pos" : "neg"}>{fmtNum(r._edge)}</td>
                  <td className={r._value > 0 ? "pos" : "neg"}>{fmtNum(r._value)}</td>
                  <td>{qual}</td>
                  <td>{r._pick}</td>
                </tr>
              );
            })}
            {!filtered.length && (
              <tr><td colSpan={11} style={{textAlign:"center", padding:12}}>No rows to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads (negative = home favorite). <b>Recommended side</b> is the value side:
        edge = model − market (home perspective). edge&gt;0 ⇒ value = away; edge&lt;0 ⇒ value = home.
      </div>
    </section>
  );
}