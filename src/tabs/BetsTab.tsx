import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, downloadCsv } from "../lib/ui";

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string; edge_points_book?: string; value_points_book?: string;
  qualified_edge_flag?: string;
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
  const f = (b*p - q) / b;
  return Math.max(0, Math.min(0.25, f)); // cap 25%
}

export default function BetsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [bankroll, setBankroll] = useState<string>("1000");
  const [mode, setMode] = useState<StakeMode>("prop");
  const [odds, setOdds] = useState<string>("-110");
  const [topN, setTopN] = useState<string>("10");

  useEffect(()=>{(async()=>{
    try { setRows((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setRows([]); }
  })();},[]);

  const wk = useMemo(()=>{
    if (!rows.length) return null;
    const unplayed = rows.filter(r => !playedBool(r.played));
    if (unplayed.length) return Math.min(...unplayed.map(r => Number(r.week)));
    return Math.max(...rows.map(r => Number(r.week)));
  }, [rows]);

  const candidatesRaw = useMemo(()=>{
    if (!wk) return [];
    return rows.filter(r =>
      Number(r.week) === wk &&
      r.market_spread_book !== undefined && r.market_spread_book !== "" &&
      !playedBool(r.played)
    );
  }, [rows, wk]);

  const strongFiltered = useMemo(()=>{
    if (!candidatesRaw.length) return [];
    const edges = candidatesRaw.map(r => Math.abs(toNum(r.edge_points_book))).filter(Number.isFinite);
    const mean = edges.length ? edges.reduce((a,b)=>a+b,0)/edges.length : 0;
    const variance = edges.length ? edges.reduce((a,b)=>a+(b-mean)*(b-mean),0)/edges.length : 0;
    const sd = Math.sqrt(variance);

    const isDQ = (r: PredRow) => {
      const e = Math.abs(toNum(r.edge_points_book));
      const v = Math.abs(toNum(r.value_points_book));
      if (!Number.isFinite(e) || !Number.isFinite(v)) return true;
      if (e > mean + 2.5*sd) return true;   // outlier
      if (v > 40) return true;              // likely bad market
      return false;
    };

    const list = candidatesRaw.filter(r => !isDQ(r));
    const ranked = [...list].sort((a,b)=>{
      const qa = a.qualified_edge_flag === "1" ? 1 : 0;
      const qb = b.qualified_edge_flag === "1" ? 1 : 0;
      if (qa !== qb) return qb - qa;
      const ea = Math.abs(toNum(a.edge_points_book));
      const eb = Math.abs(toNum(b.edge_points_book));
      if (ea !== eb) return eb - ea;
      const va = Math.abs(toNum(a.value_points_book));
      const vb = Math.abs(toNum(b.value_points_book));
      return vb - va;
    });

    const limit = Math.max(1, Number(topN) || 10);
    return ranked.slice(0, limit);
  }, [candidatesRaw, topN]);

  const staked = useMemo(()=>{
    const bk = Math.max(0, Number(bankroll) || 0);
    const oddsAm = Number(odds) || -110;
    if (!bk || !strongFiltered.length) return [];
    const edges = strongFiltered.map(r => Math.abs(toNum(r.edge_points_book)));
    const maxEdge = Math.max(...edges.map(e => (Number.isFinite(e) ? e : 0)));
    const weights = strongFiltered.map((r, i) => {
      const e = edges[i];
      if (!Number.isFinite(e) || e <= 0) return 0;
      if (mode === "flat") return 1;
      if (mode === "prop") return e / (maxEdge || 1);
      const p = probFromEdge(toNum(r.edge_points_book));
      return kellyFraction(p, oddsAm);
    });
    const sumW = weights.reduce((a,b)=>a+b,0);
    const alloc = strongFiltered.map((r, i) => {
      const w = weights[i];
      const stake = sumW ? (bk * (w / sumW)) : 0;
      const side = toNum(r.model_spread_book) < 0 ? `${r.home_team} (home)` : `${r.away_team} (away)`;
      return {
        week: r.week, date: r.date, matchup: `${r.away_team} @ ${r.home_team}`,
        side, model_line_book: toNum(r.model_spread_book),
        market_line_book: toNum(r.market_spread_book),
        edge_points: toNum(r.edge_points_book), value_points: toNum(r.value_points_book),
        qualified: r.qualified_edge_flag === "1" ? "✓" : "—",
        odds: oddsAm, stake: Math.round(stake * 100) / 100,
      };
    });
    return alloc;
  }, [strongFiltered, bankroll, mode, odds]);

  const totalStake = useMemo(()=> staked.reduce((a,b)=>a + (b.stake||0), 0), [staked]);

  return (
    <section className="card">
      <div className="card-title">Recommended Bets (Strong Only) — 2025</div>
      <div className="controls" style={{ flexWrap: "wrap", gap: 8 }}>
        <Badge>{wk ? `Week ${wk}` : "No upcoming week detected"}</Badge>
        <label>Bankroll ($)
          <input className="input" type="number" min="0" step="1" value={bankroll} onChange={(e)=>setBankroll(e.target.value)} />
        </label>
        <label>Stake Mode
          <select className="input" value={mode} onChange={(e)=>setMode(e.target.value as StakeMode)}>
            <option value="flat">Flat (equal)</option>
            <option value="prop">Proportional (by edge)</option>
            <option value="kelly">Kelly-Lite (cap 25%)</option>
          </select>
        </label>
        <label>Odds (American)
          <input className="input" type="number" step="5" value={odds} onChange={(e)=>setOdds(e.target.value)} />
        </label>
        <label>Show Top N
          <input className="input" type="number" min="1" step="1" value={topN} onChange={(e)=>setTopN(e.target.value)} />
        </label>
        <Badge tone="muted">Total Staked: ${fmtNum(totalStake)}</Badge>
        <button className="btn" disabled={!staked.length} onClick={()=>downloadCsv(`week_${wk}_bets.csv`, staked)}>Download betslip CSV</button>
      </div>

      {!strongFiltered.length ? <div className="note">No strong, data-quality-safe candidates found for the upcoming week.</div> : (
        <div className="table-wrap">
          <table className="tbl wide">
            <thead>
              <tr>
                <th>Wk</th><th>Date</th><th colSpan={2}>Matchup</th>
                <th>Model (H)</th><th>Market (H)</th><th>Edge</th><th>Value</th><th>Qualified</th>
                <th>Recommended Side</th><th>Stake</th><th>Odds</th>
              </tr>
            </thead>
            <tbody>
              {strongFiltered.map((r,i)=>{
                const model = toNum((r as any).model_spread_book);
                const market = toNum((r as any).market_spread_book);
                const edge = toNum((r as any).edge_points_book);
                const value = toNum((r as any).value_points_book);
                const qual = (r as any).qualified_edge_flag === "1" ? "✓" : "—";
                const neutral = (r as any).neutral_site === "1" || (r as any).neutral_site === "true";
                const sideHome = model < 0;
                const side = sideHome ? `${(r as any).home_team} (home)` : `${(r as any).away_team} (away)`;
                const stakeRow = (staked[i] as any);

                return (
                  <tr key={`${r.week}-${(r as any).date}-${(r as any).home_team}-${(r as any).away_team}`} className={i%2?"alt":undefined}>
                    <td>{r.week}</td>
                    <td>{(r as any).date}</td>
                    <td style={{ textAlign: "right" }}><TeamLabel home={false} team={(r as any).away_team} neutral={false} /></td>
                    <td style={{ textAlign: "left" }}><TeamLabel home={true} team={(r as any).home_team} neutral={!!neutral} /></td>
                    <td>{fmtNum(model)}</td>
                    <td>{fmtNum(market)}</td>
                    <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                    <td className={Number.isFinite(value) ? (value > 0 ? "pos" : "neg") : undefined}>{fmtNum(value)}</td>
                    <td>{qual}</td>
                    <td>{side}</td>
                    <td>${fmtNum(stakeRow?.stake ?? 0, { maximumFractionDigits: 0 })}</td>
                    <td>{odds}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <div className="note" style={{marginTop:8}}>
        We exclude data-quality outliers (|edge| z-score &gt; 2.5, or extreme value) and rank Qualified ✓ first, then by |Edge| then Value.
      </div>
    </section>
  );
}