import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, scoreText } from "../lib/ui";

type TeamRow = { team: string; conference: string; };
type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; home_points?: string; away_points?: string; played?: any;
  market_spread_book?: string; model_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string; qualified_edge_flag?: string; model_result?: string;
};

export default function TeamTab() {
  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [rows, setRows] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => { (async () => {
    try { setTeams(await loadCsv("data/upa_team_inputs_datadriven_v0.csv") as TeamRow[]); } catch { setTeams([]); }
    try { setRows(await loadCsv("data/upa_predictions.csv") as PredRow[]); } catch { setRows([]); }
  })(); }, []);

  const gamesByTeam = useMemo(() => {
    const map = new Map<string, PredRow[]>();
    for (const r of rows) {
      if (!map.has(r.home_team)) map.set(r.home_team, []);
      if (!map.has(r.away_team)) map.set(r.away_team, []);
      map.get(r.home_team)!.push(r);
      map.get(r.away_team)!.push(r);
    }
    for (const arr of map.values()) arr.sort((a,b)=> (a.date||"").localeCompare(b.date||"") || Number(a.week)-Number(b.week));
    return map;
  }, [rows]);

  const sel = useMemo(() => {
    const ql = q.trim().toLowerCase();
    if (!ql) return null;
    return teams.find(t => t.team?.toLowerCase()===ql) || teams.find(t => t.team?.toLowerCase().includes(ql)) || null;
  }, [q, teams]);

  const games = useMemo(()=> sel ? (gamesByTeam.get(sel.team) ?? []) : [], [sel, gamesByTeam]);

  return (
    <section className="card">
      <div className="card-title">Team (Schedule) — 2025</div>
      <div className="controls">
        <input className="search" placeholder="Type a team name…" value={q} onChange={(e) => setQ(e.target.value)} />
        {sel ? <Badge>{sel.team}</Badge> : <Badge>{teams.length} teams</Badge>}
      </div>

      {!sel ? (
        <div className="note">Start typing a team to view schedule with scores & results.</div>
      ) : (
        <div className="table-wrap">
          <table className="tbl wide">
            <thead>
              <tr>
                <th>Wk</th><th>Date</th><th colSpan={2}>Matchup</th>
                <th>Score (A @ H)</th><th>Model (view)</th><th>Market (view)</th><th>Expected (view)</th>
                <th>Edge (view)</th><th>Value (view)</th><th>Qualified</th><th>Result</th>
              </tr>
            </thead>
            <tbody>
              {games.map((g,i)=>{
                const isHome = g.home_team===sel.team;
                const neutral = g.neutral_site === "1" || g.neutral_site === "true";
                const sgn = isHome ? 1 : -1;
                const model = toNum(g.model_spread_book) * sgn;
                const mkt   = toNum(g.market_spread_book)  * sgn;
                const exp   = toNum(g.expected_market_spread_book) * sgn;
                const edge  = toNum(g.edge_points_book) * sgn;
                const val   = toNum(g.value_points_book) * sgn;
                const played = playedBool(g.played);

                return (
                  <tr key={`${g.week}-${g.date}-${g.home_team}-${g.away_team}`} className={i%2?"alt":undefined}>
                    <td>{g.week}</td>
                    <td>{g.date}</td>
                    <td style={{ textAlign: "right" }}><TeamLabel home={false} team={g.away_team} neutral={false} /></td>
                    <td style={{ textAlign: "left" }}><TeamLabel home={true} team={g.home_team} neutral={!!neutral} /></td>
                    <td>{played ? scoreText(g.away_points, g.home_points) : "—"}</td>
                    <td className={Number.isFinite(model)?(model>0?"pos":"neg"):undefined}>{fmtNum(model)}</td>
                    <td className={Number.isFinite(mkt)?(mkt>0?"pos":"neg"):undefined}>{fmtNum(mkt)}</td>
                    <td className={Number.isFinite(exp)?(exp>0?"pos":"neg"):undefined}>{fmtNum(exp)}</td>
                    <td className={Number.isFinite(edge)?(edge>0?"pos":"neg"):undefined}>{fmtNum(edge)}</td>
                    <td className={Number.isFinite(val)?(val>0?"pos":"neg"):undefined}>{fmtNum(val)}</td>
                    <td>{g.qualified_edge_flag==="1"?"✓":"—"}</td>
                    <td>{g.model_result==="CORRECT" ? <Badge tone="pos">✓ Model</Badge> : g.model_result==="INCORRECT" ? <Badge tone="neg">✗ Model</Badge> : null}</td>
                  </tr>
                );
              })}
              {!games.length && <tr><td colSpan={12} style={{textAlign:"center",padding:12}}>No games</td></tr>}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}