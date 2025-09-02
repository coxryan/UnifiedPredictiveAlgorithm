import { useEffect, useMemo, useState } from "react";
import "./styles.css";

async function loadText(path: string) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to fetch ${path}: ${r.status}`);
  return r.text();
}
async function loadCsv(path: string) {
  const txt = await loadText(path);
  const lines = txt.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const cols = lines[0].split(",").map((c) => c.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const o: Record<string, string> = {};
    cols.forEach((c, i) => (o[c] = (cells[i] ?? "").trim()));
    return o;
  });
}
function fmtNum(v: any, opts: Intl.NumberFormatOptions = {}) {
  if (v === null || v === undefined || v === "" || v === "NaN") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 1, ...opts });
}
function Badge({ children }: { children: React.ReactNode }) {
  return <span className="badge">{children}</span>;
}

type Tab = "status" | "team" | "preds" | "edge";

type Status = {
  generated_at_utc: string;
  year: number;
  teams: number;
  games?: number;
  pred_rows?: number;
  next_run_eta_utc: string;
};
function StatusTab() {
  const [status, setStatus] = useState<Status | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const txt = await loadText("data/status.json");
        setStatus(JSON.parse(txt));
      } catch (e: any) {
        setErr(String(e?.message || e));
      }
    })();
  }, []);

  return (
    <section className="card">
      <div className="card-title">Collector Status</div>
      {!status ? (
        <div className="note">{err ? `Status unavailable (${err})` : "Loading…"}</div>
      ) : (
        <>
          <div className="grid2">
            <div className="kv"><div className="k">Last updated</div><div className="v">{status.generated_at_utc}</div></div>
            <div className="kv"><div className="k">Next run ETA</div><div className="v">{status.next_run_eta_utc}</div></div>
            <div className="kv"><div className="k">Season</div><div className="v">{status.year}</div></div>
            <div className="kv"><div className="k">Teams</div><div className="v">{status.teams}</div></div>
            {!!status.games && <div className="kv"><div className="k">Games</div><div className="v">{status.games}</div></div>}
            {!!status.pred_rows && <div className="kv"><div className="k">Pred rows</div><div className="v">{status.pred_rows}</div></div>}
          </div>
          <div className="note" style={{ marginTop: 8 }}>
            <a href={"data/upa_team_inputs_datadriven_v0.csv"} target="_blank" rel="noreferrer">team inputs CSV ↗</a>{" • "}
            <a href={"data/cfb_schedule.csv"} target="_blank" rel="noreferrer">schedule CSV ↗</a>{" • "}
            <a href={"data/upa_predictions.csv"} target="_blank" rel="noreferrer">predictions CSV ↗</a>{" • "}
            <a href={"data/live_edge_report.csv"} target="_blank" rel="noreferrer">live edge CSV ↗</a>{" • "}
            <a href={"data/diagnostics_summary.csv"} target="_blank" rel="noreferrer">diagnostics CSV ↗</a>
          </div>
        </>
      )}
    </section>
  );
}

type PredRow = {
  week: string; date: string; home_team: string; away_team: string;
  home_points?: string; away_points?: string; played?: string | boolean;
  market_spread_book?: string; model_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string;
  model_result?: string;
  qualified_edge_flag?: string;
};
function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => { (async () => {
    try { setRows((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setRows([]); }
  })(); }, []);

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    if (!ql) return rows;
    return rows.filter(r =>
      r.home_team?.toLowerCase().includes(ql) ||
      r.away_team?.toLowerCase().includes(ql)
    );
  }, [rows, q]);

  const resultBadge = (r: PredRow) => {
    const played = r.played === "True" || r.played === "true" || r.played === true;
    if (!played) return null;
    const won = r.model_result === "CORRECT";
    return <span className="badge" style={{ background: won ? "rgba(19,209,142,.15)" : "rgba(255,107,107,.15)", borderColor: won ? "#0fbf83" : "#ff6b6b" }}>
      {won ? "✓ Model" : "✗ Model"}
    </span>;
  };

  return (
    <section className="card">
      <div className="card-title">Predictions (Book-style)</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Badge>{filtered.length} rows</Badge>
      </div>

      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Wk</th><th>Date</th><th>Away</th><th>Home</th><th>Score</th>
              <th>Model (H)</th><th>Market (H)</th><th>Expected (H)</th>
              <th>Edge</th><th>Value</th><th>Qualified</th><th>Result</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const model = Number(r.model_spread_book);
              const market = Number(r.market_spread_book);
              const expected = Number(r.expected_market_spread_book);
              const edge = Number(r.edge_points_book);
              const value = Number(r.value_points_book);
              const played = r.played === "True" || r.played === "true" || r.played === true;
              const score = played ? `${r.away_points ?? "?"}–${r.home_points ?? "?"}` : "—";
              const badge = resultBadge(r);
              const qual = r.qualified_edge_flag === "1" ? "✓" : "—";

              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td>{r.away_team}</td>
                  <td>{r.home_team}</td>
                  <td>{score}</td>
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
            {!filtered.length && <tr><td colSpan={12} style={{ textAlign: "center", padding: 12 }}>No rows</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

type TeamRow = {
  team: string; conference: string;
  wrps_offense_percent?: string; wrps_defense_percent?: string; wrps_overall_percent?: string; wrps_percent_0_100?: string;
  talent_score_0_100?: string; portal_net_0_100?: string; sos_0_100?: string;
  team_power_0_100?: string; adv_score?: string; team_rating?: string;
};
type PredFull = PredRow & { neutral_site?: string };
function TeamTab() {
  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [rows, setRows] = useState<PredFull[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    (async () => {
      try { setTeams((await loadCsv("data/upa_team_inputs_datadriven_v0.csv")) as TeamRow[]); } catch { setTeams([]); }
      try { setRows((await loadCsv("data/upa_predictions.csv")) as PredFull[]); } catch { setRows([]); }
    })();
  }, []);

  const m = useMemo(() => {
    const map = new Map<string, PredFull[]>();
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

  const games = useMemo(()=> sel ? (m.get(sel.team) ?? []) : [], [sel, m]);

  return (
    <section className="card">
      <div className="card-title">Team (Schedule)</div>
      <div className="controls">
        <input className="search" placeholder="Type a team name…" value={q} onChange={(e) => setQ(e.target.value)} />
        {sel ? <Badge>{sel.team} • {sel.conference}</Badge> : <Badge>{teams.length} teams</Badge>}
      </div>

      {!sel ? <div className="note">Start typing a team to view schedule with scores & results.</div> : (
        <div className="table-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Wk</th><th>Date</th><th>Opp</th><th>H/A/N</th>
                <th>Score</th><th>Model (H)</th><th>Market (H)</th><th>Expected (H)</th>
                <th>Edge</th><th>Value</th><th>Qualified</th><th>Result</th>
              </tr>
            </thead>
            <tbody>
              {games.map((g,i)=>{
                const isHome = g.home_team===sel.team;
                const opp = isHome ? g.away_team : g.home_team;
                const home = Number(g.model_spread_book);
                const mkt  = Number(g.market_spread_book);
                const exp  = Number(g.expected_market_spread_book);
                const edge = Number(g.edge_points_book);
                const val  = Number(g.value_points_book);
                const played = g.played === "True" || g.played === "true" || g.played === true;
                const score = played ? `${g.away_points ?? "?"}–${g.home_points ?? "?"}` : "—";
                const qual = g.qualified_edge_flag === "1" ? "✓" : "—";
                const badge = (g.model_result==="CORRECT") ? <span className="badge" style={{background:"rgba(19,209,142,.15)",borderColor:"#0fbf83"}}>✓ Model</span>
                             : (g.model_result==="INCORRECT") ? <span className="badge" style={{background:"rgba(255,107,107,.15)",borderColor:"#ff6b6b"}}>✗ Model</span>
                             : null;
                // flip for team perspective if away
                const sgn = isHome ? 1 : -1;
                return (
                  <tr key={`${g.week}-${g.date}-${g.home_team}-${g.away_team}`} className={i%2?"alt":undefined}>
                    <td>{g.week}</td>
                    <td>{g.date}</td>
                    <td>{opp}</td>
                    <td>{g.neutral_site==="1" ? "N" : isHome ? "H" : "A"}</td>
                    <td>{score}</td>
                    <td>{fmtNum(sgn*home)}</td>
                    <td>{fmtNum(sgn*mkt)}</td>
                    <td>{fmtNum(sgn*exp)}</td>
                    <td className={Number.isFinite(edge) ? (sgn*edge>0?"pos":"neg"):undefined}>{fmtNum(sgn*edge)}</td>
                    <td className={Number.isFinite(val) ? (sgn*val>0?"pos":"neg"):undefined}>{fmtNum(sgn*val)}</td>
                    <td>{qual}</td>
                    <td>{badge}</td>
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

function LiveEdgeTab() {
  const [rows, setRows] = useState<any[]>([]);
  const [q, setQ] = useState("");

  useEffect(()=>{(async()=>{
    try { setRows((await loadCsv("data/live_edge_report.csv")) as any[]); } catch { setRows([]); }
  })();},[]);

  const filtered = useMemo(()=>{
    const ql = q.trim().toLowerCase();
    if (!ql) return rows;
    return rows.filter(r => r.home_team?.toLowerCase().includes(ql) || r.away_team?.toLowerCase().includes(ql));
  },[rows,q]);

  return (
    <section className="card">
      <div className="card-title">Live Edge (Book-style)</div>
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

export default function App() {
  const [tab, setTab] = useState<Tab>("preds");
  return (
    <div className="page">
      <header className="header">
        <h1>UPA-F Dashboard</h1>
        <p className="sub">Calibrated predictions (book-style), market-aligned value, and results</p>
      </header>
      <nav className="tabs">
        <button className={tab==="status"?"active":""} onClick={()=>setTab("status")}>Status</button>
        <button className={tab==="team"?"active":""} onClick={()=>setTab("team")}>Team (Schedule)</button>
        <button className={tab==="preds"?"active":""} onClick={()=>setTab("preds")}>Predictions</button>
        <button className={tab==="edge"?"active":""} onClick={()=>setTab("edge")}>Live Edge</button>
      </nav>
      {tab==="status" && <StatusTab/>}
      {tab==="team" && <TeamTab/>}
      {tab==="preds" && <PredictionsTab/>}
      {tab==="edge" && <LiveEdgeTab/>}
      <footer className="footer">© {new Date().getFullYear()} UPA-F</footer>
    </div>
  );
}