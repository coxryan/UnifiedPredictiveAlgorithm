import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, downloadCsv, nextUpcomingWeek } from "../lib/ui";
import { EDGE_MIN, VALUE_MIN } from "./constants";

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

 type StakeMode = "flat" | "prop" | "kelly";

 // Probability of cover from edge (simple bounded mapping)
 function probFromEdge(edgePts: number) {
  const x = Math.min(20, Math.max(0, Math.abs(edgePts)));
  return 0.50 + (x / 20) * 0.25; // 50%..75%
 }

 function kellyFraction(p: number, oddsAmerican: number) {
  let b: number;
  if (oddsAmerican > 0) b = oddsAmerican / 100; else b = 100 / Math.abs(oddsAmerican);
  const q = 1 - p;
  const f = (b * p - q) / b;
  return Math.max(0, Math.min(0.25, f)); // cap 25%
 }

 function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
 }

 // Compute EV (expected ROI per $1) from edge via prob mapping and odds
 function evFromEdge(edgePts: number, oddsAmerican: number) {
  let b = oddsAmerican > 0 ? oddsAmerican / 100 : 100 / Math.abs(oddsAmerican); // net win per $1
  const p = probFromEdge(edgePts);
  return b * p - (1 - p); // expected ROI per $1 staked
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
  const [requirePositiveEV, setRequirePositiveEV] = useState<boolean>(true);

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

  // available week list for dropdown
  const weekOptions = useMemo(() => {
    const w = Array.from(
      new Set(
        rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x))
      )
    ).sort((a, b) => a - b);
    return w as number[];
  }, [rows]);

  const nextWk = useMemo(() => nextUpcomingWeek(rows as any), [rows]);

  const withComputed = useMemo(() => {
    return rows.map((r) => {
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
      const neutral = r.neutral_site === "1" || r.neutral_site === "true";
      const ev = Number.isFinite(edge) ? evFromEdge(edge, Number(odds) || -110) : NaN;
      return { ...r, _model: model, _market: market, _edge: edge, _value: value, _pick: pick.side, _neutral: neutral, _ev: ev } as any;
    });
  }, [rows, odds]);

  // Helper to compute W/L/P for a row when scores are present
  function resultForRow(r: any): "W" | "L" | "P" | "" {
    const hp = toNum(r.home_points);
    const ap = toNum(r.away_points);
    if (!Number.isFinite(hp) || !Number.isFinite(ap) || !Number.isFinite(r._market)) return "";
    // book-style: negative = home favorite. Home covers if (home - away + market) > 0
    const adj = (hp - ap) + r._market;
    const coverHome = adj > 0 ? 1 : adj < 0 ? -1 : 0; // 1 cover, -1 no cover, 0 push
    const pickHome = typeof r._pick === "string" && r._pick.includes("(home)");
    const pickAway = typeof r._pick === "string" && r._pick.includes("(away)");
    if (!pickHome && !pickAway) return "";
    if (coverHome === 0) return "P";
    return pickHome ? (coverHome > 0 ? "W" : "L") : (coverHome < 0 ? "W" : "L");
  }

  // Base candidate set for selected week
  const candidatesRaw = useMemo(() => {
    if (!wk) return [] as any[];
    const isFutureWeek = nextWk && wk >= nextWk; // for current/upcoming, hide already played
    return withComputed.filter((r: any) => {
      const sameWeek = Number(r.week) === wk;
      if (!sameWeek) return false;
      if (!isFutureWeek) return true; // past weeks: include played
      return !playedBool(r.played);
    });
  }, [withComputed, wk, nextWk]);

  // Strong-filtered list with outlier removal + thresholds + optional positive EV
  const strongFiltered = useMemo(() => {
    if (!candidatesRaw.length) return [] as any[];

    const edgeAbs = candidatesRaw
      .map((r: any) => Math.abs(r._edge))
      .filter((v: number) => Number.isFinite(v)) as number[];

    const mean = edgeAbs.length ? edgeAbs.reduce((a, b) => a + b, 0) / edgeAbs.length : 0;
    const variance = edgeAbs.length ? edgeAbs.reduce((a, b) => a + (b - mean) ** 2, 0) / edgeAbs.length : 0;
    const sd = Math.sqrt(variance);

    const isOutlier = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      if (!Number.isFinite(e) || !Number.isFinite(v)) return true;
      if (v > 40) return true; // unrealistically large value → likely data issue
      if (sd > 0 && e > mean + 2.5 * sd) return true; // z-score outlier
      return false;
    };

    const meetsThresholds = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      const q = r.qualified_edge_flag === "1";
      const evOk = !requirePositiveEV || (Number.isFinite(r._ev) && r._ev > 0);
      if (requireQualified && !q) return false;
      if (!Number.isFinite(e) || !Number.isFinite(v)) return false;
      if (e < EDGE_MIN) return false;
      if (v < VALUE_MIN) return false;
      if (!evOk) return false;
      return true;
    };

    const kept = candidatesRaw.filter((r: any) => meetsThresholds(r) && !isOutlier(r));

    kept.sort((a: any, b: any) => {
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
  }, [candidatesRaw, requireQualified, requirePositiveEV, topN]);

  // Stakes
  const staked = useMemo(() => {
    const bk = Math.max(0, Number(bankroll) || 0);
    const oddsAm = Number(odds) || -110;
    if (!bk || !strongFiltered.length) return [] as any[];

    const edges = strongFiltered.map((r: any) => Math.abs(r._edge));
    const maxEdge = Math.max(...edges.map((e: number) => (Number.isFinite(e) ? e : 0)));

    const weights = strongFiltered.map((r: any, i: number) => {
      const e = edges[i];
      if (!Number.isFinite(e) || e <= 0) return 0;
      if (mode === "flat") return 1;
      if (mode === "prop") return e / (maxEdge || 1);
      const p = probFromEdge(r._edge);
      return kellyFraction(p, oddsAm);
    });

    const sumW = weights.reduce((a, b) => a + b, 0);
    return strongFiltered.map((r: any, i: number) => {
      const w = weights[i];
      const stake = sumW ? bk * (w / sumW) : 0;
      const res = resultForRow(r);
      return { ...r, _stake: Math.round(stake * 100) / 100, _result: res };
    });
  }, [strongFiltered, bankroll, mode, odds]);

  const totalStake = useMemo(() => staked.reduce((a: number, b: any) => a + b._stake, 0), [staked]);

  // Summary W-L-P for the visible (staked) rows of the selected week
  const wlSummary = useMemo(() => {
    let W = 0, L = 0, P = 0;
    staked.forEach((r: any) => {
      if (r._result === "W") W++; else if (r._result === "L") L++; else if (r._result === "P") P++;
    });
    const tot = W + L + P;
    const pct = tot ? ((W + 0.5 * P) / tot) * 100 : 0;
    return { W, L, P, pct };
  }, [staked]);

  return (
    <section className="card">
      <div className="card-title">Recommended Bets (Strong Only) — 2025</div>
      <div className="controls" style={{ flexWrap: "wrap", gap: 8 }}>
        {/* Week selection */}
        <label>Week
          <select
            className="input"
            value={wk ?? ""}
            onChange={(e) => setWk(e.target.value ? Number(e.target.value) : null)}
          >
            {(weekOptions.length ? weekOptions : [wk ?? 1]).map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </label>
        {wk && nextWk && wk === nextWk ? (
          <Badge tone="muted">Upcoming</Badge>
        ) : wk ? (
          <Badge tone="muted">Historical</Badge>
        ) : null}

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

        <label className="chk">
          <input type="checkbox" checked={requirePositiveEV} onChange={(e) => setRequirePositiveEV(e.target.checked)} />
          Require positive EV (based on edge & odds)
        </label>

        <Badge tone="muted">Total Staked: ${fmtNum(totalStake)}</Badge>
        {wk && (
          <Badge tone="info">W-L-P: {wlSummary.W}-{wlSummary.L}-{wlSummary.P} · Acc: {fmtNum(wlSummary.pct, {maximumFractionDigits:1})}%</Badge>
        )}

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
                ev_roi: r._ev,
                qualified: r.qualified_edge_flag === "1" ? "YES" : "NO",
                stake: r._stake,
                result: r._result || "",
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
          No strong, data-quality-safe candidates for this week. If you want to
          see more plays, uncheck “Require Qualified ✓” / “Require positive EV” or increase “Top N”.
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
                <th>EV</th>
                <th>Qualified</th>
                <th>Recommended Side (value)</th>
                <th>Stake</th>
                <th>Result</th>
                <th>Odds</th>
              </tr>
            </thead>
            <tbody>
              {staked.map((r: any, i: number) => {
                const edge = r._edge;
                const value = r._value;
                const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
                const res = r._result || "";
                return (
                  <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i % 2 ? "alt" : undefined}>
                    <td>{r.week}</td>
                    <td>{r.date}</td>
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
                      <TeamLabel home={true} team={r.home_team} neutral={r._neutral} />
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
                    <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                    <td className={Number.isFinite(value) ? (value > 0 ? "pos" : "neg") : undefined}>{fmtNum(value)}</td>
                    <td className={Number.isFinite(r._ev) ? (r._ev > 0 ? "pos" : "neg") : undefined}>{fmtNum((r._ev as number) * 100, { maximumFractionDigits: 1 })}%</td>
                    <td>{qual}</td>
                    <td>{r._pick}</td>
                    <td>${fmtNum(r._stake, { maximumFractionDigits: 0 })}</td>
                    <td>{res}</td>
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
        edge = model − market (home perspective). Positive EV is calculated from a bounded probability
        mapping of edge to cover-probability and the American odds.
      </div>
    </section>
  );
}