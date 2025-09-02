import { useEffect, useMemo, useState } from "react";
import "./styles.css";

/* =================== CSV/JSON loaders (relative "data/…") =================== */
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

/* =================== Tabs =================== */
type Tab = "status" | "team" | "preds" | "edge";

/* =================== Status Tab =================== */
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
            {typeof status.games === "number" && (
              <div className="kv"><div className="k">Games</div><div className="v">{status.games}</div></div>
            )}
            {typeof status.pred_rows === "number" && (
              <div className="kv"><div className="k">Pred rows</div><div className="v">{status.pred_rows}</div></div>
            )}
          </div>
          <div className="note" style={{ marginTop: 8 }}>
            <a href={"data/upa_team_inputs_datadriven_v0.csv"} target="_blank" rel="noreferrer">team inputs CSV ↗</a>{" • "}
            <a href={"data/cfb_schedule.csv"} target="_blank" rel="noreferrer">schedule CSV ↗</a>{" • "}
            <a href={"data/upa_predictions.csv"} target="_blank" rel="noreferrer">predictions CSV ↗</a>{" • "}
            <a href={"data/live_edge_report.csv"} target="_blank" rel="noreferrer">live edge CSV ↗</a>
          </div>
        </>
      )}
    </section>
  );
}

/* =================== Shared Types =================== */
type PredRow = {
  game_id: string;
  week: string;
  date: string;
  home_team: string;
  away_team: string;
  home_conf?: string;
  away_conf?: string;
  neutral_site: string;
  home_points?: string;
  away_points?: string;
  played?: string | boolean;
  actual_home_margin?: string;

  // Advantage & rating
  home_adv_score?: string;
  away_adv_score?: string;
  adv_gap?: string;
  home_rating?: string;
  away_rating?: string;

  // HOME-POSITIVE (kept for reference, not displayed)
  market_spread_home?: string;
  model_spread_home?: string;
  model_spread_cal?: string;
  expected_market_spread?: string;

  // BOOK-STYLE (display everywhere)
  market_spread_book?: string;
  model_spread_book?: string;
  expected_market_spread_book?: string;

  // Edges / Value
  edge_points_book?: string;
  value_points_book?: string;

  // Picks & outcomes
  model_pick?: string;       // HOME/AWAY/PUSH (from model_book)
  model_result?: string;     // CORRECT/INCORRECT/None
  expected_pick?: string;    // HOME/AWAY/PUSH (from expected_book)
  expected_result?: string;  // CORRECT/INCORRECT/None
};

type TeamRow = {
  team: string;
  conference: string;
  wrps_offense_percent?: string;
  wrps_defense_percent?: string;
  wrps_overall_percent?: string;
  wrps_percent_0_100?: string;
  talent_score_0_100?: string;
  portal_net_0_100?: string;
  sos_0_100?: string;
  team_power_0_100?: string;
  adv_score?: string;
  team_rating?: string;
};

/* =================== Team (Schedule) =================== */
function TeamTab() {
  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [rows, setRows] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    (async () => {
      try { setTeams((await loadCsv("data/upa_team_inputs_datadriven_v0.csv")) as TeamRow[]); } catch { setTeams([]); }
      try { setRows((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setRows([]); }
    })();
  }, []);

  const scheduleForTeam = useMemo(() => {
    const m = new Map<string, PredRow[]>();
    for (const r of rows) {
      if (!m.has(r.home_team)) m.set(r.home_team, []);
      if (!m.has(r.away_team)) m.set(r.away_team, []);
      m.get(r.home_team)!.push(r);
      m.get(r.away_team)!.push(r);
    }
    for (const arr of m.values()) {
      arr.sort((a, b) => (a.date || "").localeCompare(b.date || "") || Number(a.week) - Number(b.week));
    }
    return m;
  }, [rows]);

  const selTeam = useMemo(() => {
    if (!q.trim()) return null;
    const ql = q.trim().toLowerCase();
    return teams.find(t => t.team?.toLowerCase() === ql) ||
           teams.find(t => t.team?.toLowerCase().includes(ql)) ||
           null;
  }, [q, teams]);

  // Model summary (from team inputs)
  const summary = useMemo(() => {
    if (!selTeam) return null;
    const off = Number(selTeam.wrps_offense_percent ?? selTeam.wrps_percent_0_100 ?? 50);
    const def = Number(selTeam.wrps_defense_percent ?? selTeam.wrps_percent_0_100 ?? 50);
    const tal = Number(selTeam.talent_score_0_100 ?? 50);
    const portal = Number(selTeam.portal_net_0_100 ?? 50);
    const sos = Number(selTeam.sos_0_100 ?? 50);
    const power = 0.40*off + 0.25*def + 0.20*tal + 0.10*portal + 0.05*sos;
    const rating = Number(selTeam.team_rating ?? (power - 50));
    return { off, def, tal, portal, sos, power: Math.round(power*10)/10, rating: Math.round(rating*10)/10 };
  }, [selTeam]);

  const games = useMemo(() => selTeam ? (scheduleForTeam.get(selTeam.team) ?? []) : [], [selTeam, scheduleForTeam]);

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
      <div className="card-title">Team (Schedule)</div>
      <div className="controls">
        <input className="search" placeholder="Type a team name (e.g., Michigan State)…" value={q} onChange={(e) => setQ(e.target.value)} />
        {selTeam ? <Badge>{selTeam.team} • {selTeam.conference}</Badge> : <Badge>{teams.length} teams loaded</Badge>}
      </div>

      {!selTeam ? (
        <div className="note">Start typing a team to view its model summary, schedule, scores, and results.</div>
      ) : (
        <>
          <div className="grid2">
            <div className="kv"><div className="k">Team</div><div className="v">{selTeam.team}</div></div>
            <div className="kv"><div className="k">Conference</div><div className="v">{selTeam.conference}</div></div>
            <div className="kv"><div className="k">WRPS Off%</div><div className="v">{fmtNum(selTeam.wrps_offense_percent ?? selTeam.wrps_percent_0_100)}</div></div>
            <div className="kv"><div className="k">WRPS Def%</div><div className="v">{fmtNum(selTeam.wrps_defense_percent ?? selTeam.wrps_percent_0_100)}</div></div>
            <div className="kv"><div className="k">Talent 0–100</div><div className="v">{fmtNum(selTeam.talent_score_0_100)}</div></div>
            <div className="kv"><div className="k">Portal 0–100</div><div className="v">{fmtNum(selTeam.portal_net_0_100)}</div></div>
            <div className="kv"><div className="k">SOS 0–100</div><div className="v">{fmtNum(selTeam.sos_0_100 ?? 50)}</div></div>
            <div className="kv"><div className="k">Power</div><div className="v">{fmtNum(summary?.power)}</div></div>
            <div className="kv"><div className="k">Rating</div><div className="v">{fmtNum(summary?.rating)}</div></div>
          </div>

          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Week</th>
                  <th>Date</th>
                  <th>Opponent</th>
                  <th>H/A/N</th>
                  <th>Score</th>
                  <th>Model (book)</th>
                  <th>Market (book)</th>
                  <th>Expected (book)</th>
                  <th>Edge</th>
                  <th>Value</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {games.map((g, i) => {
                  const isHome = g.home_team === selTeam.team;
                  const opp = isHome ? g.away_team : g.home_team;

                  // book-style (home anchor) → flip sign for team perspective if away
                  const modelHome = Number(g.model_spread_book);
                  const marketHome = Number(g.market_spread_book);
                  const expectedHome = Number(g.expected_market_spread_book);
                  const edgeHome = Number(g.edge_points_book);
                  const valueHome = Number(g.value_points_book);

                  const modelForTeam = isHome ? modelHome : -modelHome;
                  const marketForTeam = isHome ? marketHome : -marketHome;
                  const expectedForTeam = isHome ? expectedHome : -expectedHome;
                  const edgeForTeam = isHome ? edgeHome : -edgeHome;
                  const valueForTeam = isHome ? valueHome : -valueHome;

                  const played = g.played === "True" || g.played === "true" || (g.home_points && g.away_points);
                  const score = played ? `${g.away_points ?? "?"}–${g.home_points ?? "?"}` : "—";

                  const badge = resultBadge(g);

                  return (
                    <tr key={`${g.week}-${g.date}-${g.home_team}-${g.away_team}`} className={i % 2 ? "alt" : undefined}>
                      <td>{g.week}</td>
                      <td>{g.date}</td>
                      <td>{opp}</td>
                      <td>{g.neutral_site === "1" ? "N" : isHome ? "H" : "A"}</td>
                      <td>{score}</td>
                      <td>{fmtNum(modelForTeam)}</td>
                      <td>{fmtNum(marketForTeam)}</td>
                      <td>{fmtNum(expectedForTeam)}</td>
                      <td className={Number.isFinite(edgeForTeam) ? (edgeForTeam > 0 ? "pos" : "neg") : undefined}>{fmtNum(edgeForTeam)}</td>
                      <td className={Number.isFinite(valueForTeam) ? (valueForTeam > 0 ? "pos" : "neg") : undefined}>{fmtNum(valueForTeam)}</td>
                      <td>{badge}</td>
                    </tr>
                  );
                })}
                {!games.length && (
                  <tr><td colSpan={11} style={{ textAlign: "center", padding: 12 }}>No scheduled FBS games found.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

/* =================== Predictions (book-style) =================== */
function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");
  const [conf, setConf] = useState<"All" | "P5" | "G5">("All");

  useEffect(() => {
    (async () => {
      try { setRows((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setRows([]); }
    })();
  }, []);

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const p5 = new Set(["ACC", "SEC", "Big Ten", "Big 12", "Pac-12"]);
    return rows.filter((r) => {
      if (conf !== "All" && (r.home_conf || r.away_conf)) {
        const inP5 = p5.has(String(r.home_conf)) || p5.has(String(r.away_conf));
        if (conf === "P5" && !inP5) return false;
        if (conf === "G5" && inP5) return false;
      }
      if (!ql) return true;
      return r.home_team?.toLowerCase().includes(ql) || r.away_team?.toLowerCase().includes(ql);
    });
  }, [rows, q, conf]);

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
      <div className="card-title">Predictions (Book-style spreads)</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e) => setQ(e.target.value)} />
        <select value={conf} onChange={(e) => setConf(e.target.value as any)}>
          <option>All</option>
          <option>P5</option>
          <option>G5</option>
        </select>
        <Badge>{filtered.length} rows</Badge>
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Week</th>
              <th>Date</th>
              <th>Away</th>
              <th>Home</th>
              <th>Score</th>
              <th>Model (book)</th>
              <th>Market (book)</th>
              <th>Expected (book)</th>
              <th>Edge</th>
              <th>Value</th>
              <th>Result</th>
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
                  <td>{badge}</td>
                </tr>
              );
            })}
            {!filtered.length && (
              <tr><td colSpan={11} style={{ textAlign: "center", opacity: 0.7, padding: 12 }}>No rows (try clearing filters)</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="note">
        Book-style: **negative spreads = home favorite**.  
        Edge = model_book − market_book. Value = market_book − expected_book.  
        Score and ✓/✗ appear when final scores are available (e.g., Week 1).
      </div>
    </section>
  );
}

/* =================== Live Edge =================== */
type EdgeRow = {
  week: string;
  date: string;
  home_team: string;
  away_team: string;
  model_spread_book: string;
  market_spread_book: string;
  expected_market_spread_book?: string;
  edge_points_book: string;
  value_points_book?: string;
  home_points?: string;
  away_points?: string;
  played?: string | boolean;
  model_result?: string;
};
function LiveEdgeTab() {
  const [rows, setRows] = useState<EdgeRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => { (async () => {
    try { setRows((await loadCsv("data/live_edge_report.csv")) as EdgeRow[]); } catch { setRows([]); }
  })(); }, []);

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    if (!ql) return rows;
    return rows.filter(r =>
      r.home_team?.toLowerCase().includes(ql) ||
      r.away_team?.toLowerCase().includes(ql)
    );
  }, [rows, q]);

  const resultBadge = (r: EdgeRow) => {
    const played = r.played === "True" || r.played === "true" || r.played === true;
    if (!played) return null;
    const won = r.model_result === "CORRECT";
    return <span className="badge" style={{ background: won ? "rgba(19,209,142,.15)" : "rgba(255,107,107,.15)", borderColor: won ? "#0fbf83" : "#ff6b6b" }}>
      {won ? "✓ Model" : "✗ Model"}
    </span>;
  };

  return (
    <section className="card">
      <div className="card-title">Live Edge (Book-style)</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e) => setQ(e.target.value)} />
        <Badge>{filtered.length} rows</Badge>
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Week</th>
              <th>Date</th>
              <th>Away</th>
              <th>Home</th>
              <th>Score</th>
              <th>Model (book)</th>
              <th>Market (book)</th>
              <th>Edge</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const model = Number(r.model_spread_book);
              const market = Number(r.market_spread_book);
              const edge = Number(r.edge_points_book);
              const played = r.played === "True" || r.played === "true" || r.played === true;
              const score = played ? `${r.away_points ?? "?"}–${r.home_points ?? "?"}` : "—";
              const badge = resultBadge(r);

              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td>{r.away_team}</td>
                  <td>{r.home_team}</td>
                  <td>{score}</td>
                  <td>{fmtNum(model)}</td>
                  <td>{fmtNum(market)}</td>
                  <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                  <td>{badge}</td>
                </tr>
              );
            })}
            {!filtered.length && (
              <tr><td colSpan={9} style={{ textAlign: "center", padding: 12 }}>No rows</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* =================== App Shell =================== */
export default function App() {
  const [tab, setTab] = useState<Tab>("team");
  return (
    <div className="page">
      <header className="header">
        <h1>UPA-F Dashboard</h1>
        <p className="sub">Book-style spreads (home favorite negative), score outcomes, and value alignment</p>
      </header>

      <nav className="tabs">
        <button className={tab === "status" ? "active" : ""} onClick={() => setTab("status")}>Status</button>
        <button className={tab === "team" ? "active" : ""} onClick={() => setTab("team")}>Team (Schedule)</button>
        <button className={tab === "preds" ? "active" : ""} onClick={() => setTab("preds")}>Predictions</button>
        <button className={tab === "edge" ? "active" : ""} onClick={() => setTab("edge")}>Live Edge</button>
      </nav>

      {tab === "status" && <StatusTab />}
      {tab === "team" && <TeamTab />}
      {tab === "preds" && <PredictionsTab />}
      {tab === "edge" && <LiveEdgeTab />}

      <footer className="footer">© {new Date().getFullYear()} UPA-F</footer>
    </div>
  );
}