import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel } from "../lib/ui";
import { EDGE_MIN, VALUE_MIN } from "./constants";

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string; qualified_edge_flag?: string;
  home_rank?: string; away_rank?: string;
};

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome;
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function LiveEdgeTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [limit, setLimit] = useState<number>(40);

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

  const future = useMemo(() => {
    const mapped = rows
      .filter((r) => !playedBool(r.played))
      .map((r) => {
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
      })
      .filter((r) => Number.isFinite(r._edge) && Number.isFinite(r._value));

    // rank by |Edge| desc then |Value| desc
    mapped.sort((a, b) => {
      const ea = Math.abs(a._edge), eb = Math.abs(b._edge);
      if (ea !== eb) return eb - ea;
      const va = Math.abs(a._value), vb = Math.abs(b._value);
      if (va !== vb) return vb - va;
      return (a.date || "").localeCompare(b.date || "");
    });

    return mapped.slice(0, limit);
  }, [rows, limit]);

  return (
    <section className="card">
      <div className="card-title">Live Edge (Top)</div>
      <div className="controls">
        <label>Show Top
          <input className="input" type="number" min={5} step={5} value={limit} onChange={(e)=>setLimit(Math.max(5, Number(e.target.value)||40))} />
        </label>
        <Badge>Qualified rule: |Edge| ≥ {EDGE_MIN}, |Value| ≥ {VALUE_MIN}</Badge>
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
              <th>Edge</th>
              <th>Value</th>
              <th>Qualified</th>
              <th>Value Side</th>
            </tr>
          </thead>
          <tbody>
            {future.map((r, i) => (
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i % 2 ? "alt" : undefined}>
                <td>{r.week}</td>
                <td>{r.date}</td>
                <td style={{ textAlign: "right" }}>
                  <TeamLabel home={false} team={r.away_team} neutral={false} />
                  {Number.isFinite(Number(r.away_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.away_rank)}</span>
                  ) : null}
                </td>
                <td style={{ textAlign: "left" }}>
                  <TeamLabel home={true} team={r.home_team} neutral={r._neutral} />
                  {Number.isFinite(Number(r.home_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.home_rank)}</span>
                  ) : null}
                </td>
                <td>{fmtNum(r._model)}</td>
                <td>{fmtNum(r._market)}</td>
                <td className={r._edge > 0 ? "pos" : "neg"}>{fmtNum(r._edge)}</td>
                <td className={r._value > 0 ? "pos" : "neg"}>{fmtNum(r._value)}</td>
                <td>{r.qualified_edge_flag === "1" ? "✓" : "—"}</td>
                <td>{r._pick}</td>
              </tr>
            ))}
            {!future.length && (
              <tr><td colSpan={10} style={{textAlign:"center", padding:12}}>No rows to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 8 }}>
        <b>Value side</b> uses edge = model − market (home perspective). edge&gt;0 ⇒ value = away; edge&lt;0 ⇒ value = home.
      </div>
    </section>
  );
}