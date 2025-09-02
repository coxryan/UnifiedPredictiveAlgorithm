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
    // split preserving empty trailing fields
    const cells = line.split(","); // basic parser; your data avoids embedded commas
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
function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default"|"pos"|"neg"|"muted" }) {
  const style: any = { };
  if (tone === "pos") { style.background = "rgba(19,209,142,.12)"; style.borderColor = "#0fbf83"; }
  if (tone === "neg") { style.background = "rgba(255,107,107,.12)"; style.borderColor = "#ff6b6b"; }
  if (tone === "muted") { style.opacity = 0.85; }
  return <span className="badge" style={style}>{children}</span>;
}

type Tab = "status" | "team" | "preds" | "edge" | "help";

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
  home_points?: string; away_points?: string; played?: string | boolean; neutral_site?: string;
  market_spread_book?: string; model_spread_book?: string; expected_market_spread_book?: string;
  edge_points_book?: string; value_points_book?: string;
  model_result?: string; expected_result?: string; qualified_edge_flag?: string;
};
function TeamLabel({ home, team, neutral }: { home: boolean, team: string, neutral: boolean }) {
  return (
    <div style={{ fontWeight: home ? 700 : 500, display: "inline-flex", alignItems: "center", gap: 6 }}>
      {home ? <Badge tone="muted">HOME</Badge> : <Badge tone="muted">AWAY</Badge>}
      <span>{team}</span>
      {neutral && <Badge tone="muted">NEUTRAL</Badge>}
    </div>
  );
}
function scoreText(a?: string, h?: string) {
  if (a==null || h==null || a==="" || h==="") return "—";
  return `${a} @ ${h}`;
}
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
    return <Badge tone={won ? "pos" : "neg"}>{won ? "✓ Model" : "✗ Model"}</Badge>;
  };

  return (
    <section className="card">
      <div className="card-title">Predictions (Book-style)</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Badge>{filtered.length} rows</Badge>
      </div>

      <div className="table-wrap">
        <table className="tbl wide">
          <thead>
            <tr>
              <th>Wk</th><th>Date</th>
              <th colSpan={2}>Matchup</th>
              <th>Score (A @ H)</th>
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
              const neutral = r.neutral_site === "1" || r.neutral_site === "true";
              const badge = resultBadge(r);
              const qual = r.qualified_edge_flag === "1" ? "✓" : "—";

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
  adv_prior_strength_0_100?: string;
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
                const sgn = isHome ? 1 : -1; // flip to team view
                const model = Number(g.model_spread_book) * sgn;
                const mkt   = Number(g.market_spread_book) * sgn;
                const exp   = Number(g.expected_market_spread_book) * sgn;
                const edge  = Number(g.edge_points_book) * sgn;
                const val   = Number(g.value_points_book) * sgn;
                const played = g.played === "True" || g.played === "true" || g.played === true;

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

function HelpTab() {
  return (
    <section className="card">
      <div className="card-title">Indicators & Definitions</div>
      <div className="note">
        <p><b>Book-style spread</b>: Negative favors the <b>home</b> team; positive favors the <b>away</b> team.</p>
        <ul>
          <li><b>Model (H)</b>: Our predicted book-style spread for the home team. Combines an advanced-metrics feature model (ridge) and our rating model, then calibrated weekly.</li>
          <li><b>Market (H)</b>: Consensus market spread (home perspective). If missing, edges/values won’t populate.</li>
          <li><b>Expected (H)</b>: “What the market should be” based on our advantage signal (adv_gap) mapped to market.</li>
          <li><b>Edge</b>: Model − Market (book-style). Positive = we like the away side more than market; negative = we like home more.</li>
          <li><b>Value</b>: Market − Expected. Positive = price better than our expectation (value to the away side), negative = value to home.</li>
          <li><b>Qualified</b>: Shows “✓” only when model & expected agree on side and both Edge ≥ {EDGE_MIN} and Value ≥ {VALUE_MIN} by absolute value.</li>
          <li><b>HFA</b>: Home-field advantage, estimated by conference with shrinkage; neutral-site = 0.</li>
          <li><b>Advanced metrics used</b>: team PPA/EPA (off/def + rush/pass), Success Rate, Explosiveness, Starting Field Position, Havoc, Pregame WP (priors).</li>
        </ul>
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
        <p className="sub">Calibrated predictions (book-style), market-aligned value, results & advanced metrics</p>
      </header>
      <nav className="tabs">
        <button className={tab==="status"?"active":""} onClick={()=>setTab("status")}>Status</button>
        <button className={tab==="team"?"active":""} onClick={()=>setTab("team")}>Team (Schedule)</button>
        <button className={tab==="preds"?"active":""} onClick={()=>setTab("preds")}>Predictions</button>
        <button className={tab==="edge"?"active":""} onClick={()=>setTab("edge")}>Live Edge</button>
        <button className={tab==="help"?"active":""} onClick={()=>setTab("help")}>Help</button>
      </nav>
      {tab==="status" && <StatusTab/>}
      {tab==="team" && <TeamTab/>}
      {tab==="preds" && <PredictionsTab/>}
      {tab==="edge" && <LiveEdgeTab/>}
      {tab==="help" && <HelpTab/>}
      <footer className="footer">© {new Date().getFullYear()} UPA-F</footer>
    </div>
  );
}