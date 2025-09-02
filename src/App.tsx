import { useEffect, useMemo, useState } from "react";
import "./styles.css";

/* ----------------------------- utils ----------------------------- */

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

/* ----------------------------- tabs ----------------------------- */

type Tab = "status" | "team" | "preds" | "edge";

/* ----------------------------- Status ----------------------------- */

type Status = {
  generated_at_utc: string;
  year: number;
  teams: number;
  fields: string[];
  next_run_eta_utc: string;
};

function StatusTab() {
  const [status, setStatus] = useState<Status | null>(null);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const txt = await loadText("/data/status.json");
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
          </div>
          <div className="note" style={{ marginTop: 8 }}>
            <a href="/data/upa_team_inputs_datadriven_v0.csv" target="_blank" rel="noreferrer">team inputs CSV ↗</a>{" • "}
            <a href="/data/upa_predictions.csv" target="_blank" rel="noreferrer">predictions CSV ↗</a>{" • "}
            <a href="/data/live_edge_report.csv" target="_blank" rel="noreferrer">live edge CSV ↗</a>
          </div>
        </>
      )}
    </section>
  );
}

/* ----------------------------- Team (Schedule) ----------------------------- */

type TeamRow = {
  team: string;
  conference: string;
  wrps_offense_percent?: string;
  wrps_defense_percent?: string;
  wrps_overall_percent?: string;
  talent_score_0_100?: string;
  portal_net_0_100?: string;
  wrps_percent_0_100?: string;
  sos_0_100?: string; // may be absent if not emitted; handled below
  team_power_0_100?: string; // may be absent; we display derived if missing
};

type PredRow = {
  week: string;
  date: string;
  home_team: string;
  away_team: string;
  neutral_site: string;
  home_conf?: string;
  away_conf?: string;
  model_spread_home: string;  // raw
  model_spread_cal: string;   // calibrated
  market_spread_home: string; // home-positive
  edge_points: string;        // model_cal - market_home
  cal_alpha?: string;
  cal_beta?: string;
};

function TeamTab() {
  const [teams, setTeams] = useState<TeamRow[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setTeams((await loadCsv("/data/upa_team_inputs_datadriven_v0.csv")) as TeamRow[]);
      } catch {
        setTeams([]);
      }
      try {
        setPreds((await loadCsv("/data/upa_predictions.csv")) as PredRow[]);
      } catch {
        setPreds([]);
      }
    })();
  }, []);

  // build quick index for schedule lines
  const scheduleForTeam = useMemo(() => {
    const map = new Map<string, PredRow[]>();
    for (const r of preds) {
      if (!map.has(r.home_team)) map.set(r.home_team, []);
      if (!map.has(r.away_team)) map.set(r.away_team, []);
      map.get(r.home_team)!.push(r);
      map.get(r.away_team)!.push(r);
    }
    // sort each by date, then week
    for (const [k, arr] of map.entries()) {
      arr.sort((a, b) => (a.date || "").localeCompare(b.date || "") || Number(a.week) - Number(b.week));
    }
    return map;
  }, [preds]);

  // find matching team
  const selTeam = useMemo(() => {
    if (!q.trim()) return null;
    const ql = q.trim().toLowerCase();
    return teams.find(t => t.team?.toLowerCase() === ql) ||
           teams.find(t => t.team?.toLowerCase().includes(ql)) ||
           null;
  }, [q, teams]);

  // compute a display “power / rating” if not present
  const teamSummary = useMemo(() => {
    if (!selTeam) return null;
    const off = Number(selTeam.wrps_offense_percent ?? 50);
    const def = Number(selTeam.wrps_defense_percent ?? 50);
    const tal = Number(selTeam.talent_score_0_100 ?? 50);
    const portal = Number(selTeam.portal_net_0_100 ?? 50);
    const sos = Number((selTeam as any).sos_0_100 ?? 50);
    // mimic collector weights (approx)
    const power =
      0.40 * off +
      0.25 * def +
      0.20 * tal +
      0.10 * portal +
      0.05 * sos;
    const rating = power - 50;
    return {
      off, def, tal, portal, sos,
      power: Math.round(power * 10) / 10,
      rating: Math.round(rating * 10) / 10,
    };
  }, [selTeam]);

  const games = useMemo(() => {
    if (!selTeam) return [];
    return scheduleForTeam.get(selTeam.team) ?? [];
  }, [selTeam, scheduleForTeam]);

  return (
    <section className="card">
      <div className="card-title">Team Schedule</div>
      <div className="controls">
        <input
          className="search"
          placeholder="Type a team name (e.g., Michigan State)…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        {selTeam ? (
          <Badge>{selTeam.team} • {selTeam.conference}</Badge>
        ) : (
          <Badge>{teams.length} teams loaded</Badge>
        )}
      </div>

      {!selTeam ? (
        <div className="note">Start typing a team to view its model summary and schedule.</div>
      ) : (
        <>
          <div className="grid2">
            <div className="kv"><div className="k">Team</div><div className="v">{selTeam.team}</div></div>
            <div className="kv"><div className="k">Conference</div><div className="v">{selTeam.conference}</div></div>
            <div className="kv"><div className="k">WRPS Off%</div><div className="v">{fmtNum(selTeam.wrps_offense_percent)}</div></div>
            <div className="kv"><div className="k">WRPS Def%</div><div className="v">{fmtNum(selTeam.wrps_defense_percent)}</div></div>
            <div className="kv"><div className="k">Talent 0–100</div><div className="v">{fmtNum(selTeam.talent_score_0_100)}</div></div>
            <div className="kv"><div className="k">Portal 0–100</div><div className="v">{fmtNum(selTeam.portal_net_0_100)}</div></div>
            <div className="kv"><div className="k">SOS 0–100</div><div className="v">{fmtNum((selTeam as any).sos_0_100 ?? 50)}</div></div>
            <div className="kv"><div className="k">Power</div><div className="v">{fmtNum(teamSummary?.power)}</div></div>
            <div className="kv"><div className="k">Rating</div><div className="v">{fmtNum(teamSummary?.rating)}</div></div>
          </div>

          <div className="table-wrap" style={{ marginTop: 12 }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>Week</th>
                  <th>Date</th>
                  <th>Opponent</th>
                  <th>H/A/N</th>
                  <th>Model (H)</th>
                  <th>Market (H)</th>
                  <th>Edge</th>
                </tr>
              </thead>
              <tbody>
                {games.map((g, i) => {
                  const isHome = g.home_team === selTeam.team;
                  const opp = isHome ? g.away_team : g.home_team;
                  // Model/Market are home-positive; if team is away, flip sign to show team-centric view
                  const modelHome = Number(g.model_spread_cal);
                  const marketHome = Number(g.market_spread_home);
                  const modelForTeam = isHome ? modelHome : -modelHome;
                  const marketForTeam = isHome ? marketHome : -marketHome;
                  const edgeForTeam = isHome ? Number(g.edge_points) : -Number(g.edge_points);

                  return (
                    <tr key={`${g.week}-${g.date}-${g.home_team}-${g.away_team}`} className={i % 2 ? "alt" : undefined}>
                      <td>{g.week}</td>
                      <td>{g.date}</td>
                      <td>{opp}</td>
                      <td>{g.neutral_site === "1" ? "N" : isHome ? "H" : "A"}</td>
                      <td>{fmtNum(modelForTeam)}</td>
                      <td>{fmtNum(marketForTeam)}</td>
                      <td className={Number.isFinite(edgeForTeam) ? (edgeForTeam > 0 ? "pos" : "neg") : undefined}>
                        {fmtNum(edgeForTeam)}
                      </td>
                    </tr>
                  );
                })}
                {!games.length && (
                  <tr><td colSpan={7} style={{ textAlign: "center", padding: 12 }}>No scheduled FBS games found.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

/* ----------------------------- Predictions (market + calibrated) ----------------------------- */

function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [q, setQ] = useState("");
  const [conf, setConf] = useState<"All" | "P5" | "G5">("All");
  const [bookStyle, setBookStyle] = useState(false); // sportsbook sign convention

  useEffect(() => {
    (async () => {
      try {
        setRows((await loadCsv("/data/upa_predictions.csv")) as PredRow[]);
      } catch {
        setRows([]);
      }
    })();
  }, []);

  const cal = useMemo(() => {
    if (!rows.length) return null;
    const r = rows.find((x) => x.cal_alpha && x.cal_beta) ?? rows[0];
    return r?.cal_alpha && r?.cal_beta ? { alpha: Number(r.cal_alpha), beta: Number(r.cal_beta) } : null;
  }, [rows]);

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

  return (
    <section className="card">
      <div className="card-title">Predictions</div>
      <div className="controls">
        <input className="search" placeholder="Search teams…" value={q} onChange={(e) => setQ(e.target.value)} />
        <select value={conf} onChange={(e) => setConf(e.target.value as any)}>
          <option>All</option>
          <option>P5</option>
          <option>G5</option>
        </select>
        <label title="If ON: show market with sportsbook sign (home favorite negative)">
          <input type="checkbox" checked={bookStyle} onChange={(e) => setBookStyle(e.target.checked)} style={{ marginRight: 6 }} />
          Book-style market sign
        </label>
        {cal && <Badge>α={cal.alpha.toFixed(2)} β={cal.beta.toFixed(2)}</Badge>}
      </div>
      <div className="table-wrap">
        <table className="tbl">
          <thead>
            <tr>
              <th>Week</th>
              <th>Date</th>
              <th>Away</th>
              <th>Home</th>
              <th>Neutral</th>
              <th>Model (H)</th>
              <th>Market (H)</th>
              <th>Edge</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const model = Number(r.model_spread_cal);
              let market = Number(r.market_spread_home);
              if (bookStyle && Number.isFinite(market)) market = -market;
              const edge = Number(r.edge_points);
              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td>{r.away_team}</td>
                  <td>{r.home_team}</td>
                  <td>{r.neutral_site === "1" ? "Y" : "—"}</td>
                  <td>{fmtNum(model)}</td>
                  <td>{fmtNum(market)}</td>
                  <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                </tr>
              );
            })}
            {!filtered.length && (
              <tr><td colSpan={8} style={{ textAlign: "center", opacity: 0.7, padding: 12 }}>No rows (try clearing filters)</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="note">
        “Model (H)” is the <em>calibrated</em> model (home-positive). “Market (H)” is the home-positive consensus.
        Toggle “Book-style” to display sportsbook sign convention (home favorites negative).
      </div>
    </section>
  );
}

/* ----------------------------- Live Edge ----------------------------- */

type EdgeRow = {
  week: string;
  date: string;
  home_team: string;
  away_team: string;
  model_spread_cal: string;
  market_spread_home: string;
  edge_points: string;
};

function LiveEdgeTab() {
  const [rows, setRows] = useState<EdgeRow[]>([]);
  const [q, setQ] = useState("");

  useEffect(() => {
    (async () => {
      try {
        setRows((await loadCsv("/data/live_edge_report.csv")) as EdgeRow[]);
      } catch {
        setRows([]);
      }
    })();
  }, []);

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    if (!ql) return rows;
    return rows.filter(r =>
      r.home_team?.toLowerCase().includes(ql) ||
      r.away_team?.toLowerCase().includes(ql)
    );
  }, [rows, q]);

  return (
    <section className="card">
      <div className="card-title">Live Edge (Top)</div>
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
              <th>Model (H)</th>
              <th>Market (H)</th>
              <th>Edge</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, i) => {
              const model = Number(r.model_spread_cal);
              const market = Number(r.market_spread_home);
              const edge = Number(r.edge_points);
              return (
                <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i % 2 ? "alt" : undefined}>
                  <td>{r.week}</td>
                  <td>{r.date}</td>
                  <td>{r.away_team}</td>
                  <td>{r.home_team}</td>
                  <td>{fmtNum(model)}</td>
                  <td>{fmtNum(market)}</td>
                  <td className={Number.isFinite(edge) ? (edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(edge)}</td>
                </tr>
              );
            })}
            {!filtered.length && (
              <tr><td colSpan={7} style={{ textAlign: "center", padding: 12 }}>No rows</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ----------------------------- App shell ----------------------------- */

export default function App() {
  const [tab, setTab] = useState<Tab>("team");

  return (
    <div className="page">
      <header className="header">
        <h1>UPA-F Dashboard</h1>
        <p className="sub">Automated FBS predictions (model calibrated to market) + live status</p>
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