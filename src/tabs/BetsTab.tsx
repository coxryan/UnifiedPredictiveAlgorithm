import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, downloadCsv, nextUpcomingWeek } from "../lib/ui";
import { EDGE_MIN, VALUE_MIN } from "./constants";

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string;
  expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string;
  qualified_edge_flag?: string;
  home_rank?: string; away_rank?: string;
};

type StakeMode = "flat" | "prop" | "kelly";

function probFromEdge(edgePts: number) {
  const x = Math.min(20, Math.max(0, Math.abs(edgePts)));
  return 0.50 + (x / 20) * 0.25; // 50%..75%
}

function kellyFraction(p: number, oddsAmerican: number) {
  let b: number;
  if (oddsAmerican > 0) b = oddsAmerican / 100;
  else b = 100 / Math.abs(oddsAmerican);
  const q = 1 - p;
  const f = (b * p - q) / b;
  return Math.max(0, Math.min(0.25, f)); // cap 25%
}

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function BetsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [wk, setWk] = useState<number | null>(null);

  // Controls
  const [bankroll, setBankroll] = useState<string>("1000");
  const [mode, setMode] = useState<StakeMode>("prop");
  const [odds, setOdds] = useState<string>("-110");
  const [topN, setTopN] = useState<string>("10");
  const [requireQualified, setRequireQualified] = useState<boolean>(true);

  useEffect(() => {
    (async () => {
      try {
        const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
        setRows(r);
        const nextWk = nextUpcomingWeek(r as any);
        setWk(nextWk);
      } catch {
        setRows([]);
        setWk(null);
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

  const candidatesRaw = useMemo(() => {
    if (!wk) return [];
    return withComputed.filter((r) =>
      Number(r.week) === wk &&
      !playedBool(r.played) &&
      Number.isFinite(r._model) &&
      Number.isFinite(r._market)
    );
  }, [withComputed, wk]);

  const strongFiltered = useMemo(() => {
    if (!candidatesRaw.length) return [];

    const edgeAbs = candidatesRaw
      .map((r) => Math.abs(r._edge))
      .filter((v) => Number.isFinite(v)) as number[];

    const mean = edgeAbs.length ? edgeAbs.reduce((a, b) => a + b, 0) / edgeAbs.length : 0;
    const variance = edgeAbs.length ? edgeAbs.reduce((a, b) => a + (b - mean) ** 2, 0) / edgeAbs.length : 0;
    const sd = Math.sqrt(variance);

    const isOutlier = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      if (!Number.isFinite(e) || !Number.isFinite(v)) return true;
      if (v > 40) return true;
      if (sd > 0 && e > mean + 2.5 * sd) return true;
      return false;
    };

    const meetsThresholds = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      const q = r.qualified_edge_flag === "1";
      if (requireQualified && !q) return false;
      if (!Number.isFinite(e) || !Number.isFinite(v)) return false;
      if (e < EDGE_MIN) return false;
      if (v < VALUE_MIN) return false;
      return true;
    };

    const kept = candidatesRaw.filter((r) => meetsThresholds(r) && !isOutlier(r));

    kept.sort((a, b) => {
      const qa = a.qualified_edge_flag === "1" ? 1 : 0;
      const qb = b.qualified_edge_flag === "1" ? 1 : 0;
      if (qa !== qb) return qb - qa;
      const ea = Math.abs(a._edge);
      const eb = Math.abs(b._edge);
      if (ea !== eb) return eb - ea;
      const va = Math.abs(a._value);
      const vb = Math.abs(b._value);
      if (va !== vb) return vb - va;
      return (a.date || "").localeCompare(b.date || "");
    });

    const limit = Math.max(1, Number(topN) || 10);
    return kept.slice(0, limit);
  }, [candidatesRaw, requireQualified, topN]);

  const staked = useMemo(() => {
    const bk = Math.max(0, Number(bankroll) || 0);
    const oddsAm = Number(odds) || -110;
    if (!bk || !strongFiltered.length) return [];

    const edges = strongFiltered.map((r) => Math.abs(r._edge));
    const maxEdge = Math.max(...edges.map((e) => (Number.isFinite(e) ? e : 0)));

    const weights = strongFiltered.map((r, i) => {
      const e = edges[i];
      if (!Number.isFinite(e) || e <= 0) return 0;
      if (mode === "flat") return 1;
      if (mode === "prop") return e / (maxEdge || 1);
      const p = probFromEdge(r._edge);
      return kellyFraction(p, oddsAm);
    });

    const sumW = weights.reduce((a, b) => a + b, 0);
    return strongFiltered.map((r, i) => {
      const w = weights[i];
      const stake = sumW ? bk * (w / sumW) : 0;
      return { ...r, _stake: Math.round(stake * 100) / 100 };
    });
  }, [strongFiltered, bankroll, mode, odds]);

  const totalStake = useMemo(
    () => staked.reduce((a, b) => a + (b as any)._stake, 0),
    [staked]
  );

  return (
    <section className="card">
      <div className="card-title">Recommended Bets (Strong Only) — 2025</div>
      <div className="controls" style={{ flexWrap: "wrap", gap: 8 }}>
        <Badge>{wk ? `Week ${wk}` : "No upcoming week detected"}</Badge>

        <label>Bankroll ($)
          <input className="input" type="number" min="0" step="1" value={bankroll} onChange={(e) => setBankroll(e.target.value)} />
        </label>

        <label>Stake Mode
          <select className="input" value={mode} onChange={(e) => setMode(e.target.value as StakeMode)}>
            <option value="flat">Flat (equal)</option>
            <option value="prop">Proportional (by edge)</option>
            <option value="kelly">Kelly-Lite (cap 25%)</option>
          </select>
        </label>

        <label>Odds (American)
          <input className="input" type="number" step="5" value={odds} onChange={(e) => setOdds(e.target.value)} />
        </label>

        <label>Show Top N
          <input className="input" type="number" min="1" step="1" value={topN} onChange={(e) => setTopN(e.target.value)} />
        </label>

        <label className="chk">
          <input type="checkbox" checked={requireQualified} onChange={(e) => setRequireQualified(e.target.checked)} />
          Require Qualified ✓ (|Edge| ≥ {EDGE_MIN}, |Value| ≥ {VALUE_MIN}, side agreement)
        </label>

        <Badge tone="muted">Total Staked: ${fmtNum(totalStake)}</Badge>

        <button
          className="btn"
          disabled={!staked.length}
          onClick={() =>
            downloadCsv(
              `week_${wk ?? "NA"}_bets.csv`,
              staked.map((r: any) => ({
                week: r.week,
                date: r.date,
                matchup: `${r.away_team} @ ${r.home_team}`,
                recommended_side: r._pick,
                model_line_home: r._model,
                market_line_home: r._market,
                edge_points: r._edge,
                value_points: r._value,
                qualified: r.qualified_edge_flag === "1" ? "YES" : "NO",
                stake: r._stake,
                odds: Number(odds) || -110,
              }))
            )
          }
        >
          Download betslip CSV
        </button>
      </div>

      {!staked.length ? (
        <div className="note">
          No strong, data-quality-safe candidates for the upcoming week. If you want to
          see more plays, uncheck “Require Qualified ✓” or increase “Top N”.
        </div>
      ) : (
        <div className="table-wrap">
          <table className="tbl wide">
            <thead>
              <tr>
                <th>Wk</th>
                <th>Date</th>
                <th colSpan={2}>Matchup</th>
                <th>Model (H)</th>
                <th>Market (H)</th>
                <th>Edge</th>
                <th>Value</th>
                <th>Qualified</th>
                <th>Recommended Side (value)</th>
                <th>Stake</th>
                <th>Odds</th>
              </tr>
            </thead>
            <tbody>
              {staked.map((r: any, i: number) => {
                const edge = r._edge;
                const value = r._value;
                const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
                return (
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
                    <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                    <td className={Number.isFinite(value) ? (value > 0 ? "pos" : "neg") : undefined}>{fmtNum(value)}</td>
                    <td>{qual}</td>
                    <td>{r._pick}</td>
                    <td>${fmtNum(r._stake, { maximumFractionDigits: 0 })}</td>
                    <td>{odds}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads shown (negative = home favorite). <b>Value side</b> is computed from
        edge = model − market (home perspective): edge&gt;0 ⇒ market too heavy on home ⇒ value is away;
        edge&lt;0 ⇒ market too heavy on away ⇒ value is home.
      </div>
    </section>
  );
}