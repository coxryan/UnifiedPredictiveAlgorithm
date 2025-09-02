import { useEffect, useMemo, useState } from "react";
import "./styles.css";

/* ------------------------- small utilities ------------------------- */
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
    // NOTE: assumes no embedded commas in data files we produce
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
function fmtPct01(n: number | string | undefined) {
  if (n === undefined || n === null || n === "" || n === "NaN") return "—";
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}
function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default"|"pos"|"neg"|"muted" }) {
  const style: any = { };
  if (tone === "pos") { style.background = "rgba(19,209,142,.12)"; style.borderColor = "#0fbf83"; }
  if (tone === "neg") { style.background = "rgba(255,107,107,.12)"; style.borderColor = "#ff6b6b"; }
  if (tone === "muted") { style.opacity = 0.85; }
  return <span className="badge" style={style}>{children}</span>;
}
function downloadCsv(filename: string, rows: Record<string, any>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const body = rows.map(r => cols.map(c => (r[c] ?? "")).join(",")).join("\n");
  const csv = cols.join(",") + "\n" + body + "\n";
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

/* ------------------------- types ------------------------- */
type Tab = "status" | "team" | "preds" | "edge" | "bets" | "help";

type Status = {
  generated_at_utc: string;
  year: number;
  teams: number;
  games?: number;
  pred_rows?: number;
  next_run_eta_utc: string;
};

type PredRow = {
  game_id: string;
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string;
  home_points?: string; away_points?: string; played?: string | boolean;
  market_spread_book?: string;
  model_spread_book?: string;
  expected_market_spread_book?: string;
  edge_points_book?: string;
  value_points_book?: string;
  qualified_edge_flag?: string;
  model_result?: string;
  expected_result?: string;
};

type TeamRow = {
  team: string; conference: string;
  wrps_offense_percent?: string; wrps_defense_percent?: string; wrps_overall_percent?: string; wrps_percent_0_100?: string;
  talent_score_0_100?: string; portal_net_0_100?: string; sos_0_100?: string;
  team_power_0_100?: string; adv_score?: string; team_rating?: string;
  adv_prior_strength_0_100?: string;
};

/* ------------------------- shared UI bits ------------------------- */
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
function playedBool(v: any) {
  return v === true || v === "True" || v === "true";
}
function toNum(v: any) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

/* ------------------------- helpers for weeks/bets ------------------------- */
function nextUpcomingWeek(rows: PredRow[]): number | null {
  if (!rows.length) return null;
  const today = new Date();
  const withDates = rows
    .map(r => ({ r, d: r.date ? new Date(r.date + "T00:00:00Z") : null }))
    .filter(x => x.d !== null) as {r: PredRow, d: Date}[];
  const future = withDates.filter(x => x.d.getTime() >= today.getTime());
  if (future.length) {
    const wk = Math.min(...future.map(x => Number(x.r.week)));
    return isFinite(wk) ? wk : null;
  }
  const unplayed = rows.filter(r => !playedBool(r.played));
  if (unplayed.length) {
    const wk = Math.min(...unplayed.map(r => Number(r.week)));
    return isFinite(wk) ? wk : null;
  }
  const anyWk = Math.min(...rows.map(r => Number(r.week)));
  return isFinite(anyWk) ? anyWk : null;
}
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
  return Math.max(0, Math.min(0.25, f)); // cap to 25%
}

/* ------------------------- Status tab ------------------------- */
function StatusTab() {
  const [status, setStatus] = useState<Status | null>(null);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [err, setErr] = useState<string>("");

  useEffect(() => {
    (async () => {
      try {
        const txt = await loadText("data/status.json");
        setStatus(JSON.parse(txt));
      } catch (e: any) {
        setErr(String(e?.message || e));
      }
      try {
        setPreds((await loadCsv("data/upa_predictions.csv")) as PredRow[]);
      } catch {}
    })();
  }, []);

  const mae = useMemo(() => {
    if (!preds.length) return { overall: NaN, byBucket: [] as {bucket:string, mae:number}[], lastWeek: NaN };
    const mask = preds.filter(r => r.market_spread_book && r.model_spread_book);
    const diffs = mask.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
    const overall = diffs.length ? diffs.reduce((a,b)=>a+b,0)/diffs.length : NaN;

    const buckets = [
      { name: "0–3", min: 0, max: 3 },
      { name: "3–7", min: 3, max: 7 },
      { name: "7–14", min: 7, max: 14 },
      { name: "14+", min: 14, max: 999 },
    ].map(b => {
      const rows = mask.filter(r => {
        const m = Math.abs(toNum(r.market_spread_book));
        return Number.isFinite(m) && m >= b.min && m < b.max;
      });
      const err = rows.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
      const mae = err.length ? err.reduce((a,b)=>a+b,0)/err.length : NaN;
      return { bucket: b.name, mae };
    });

    const played = preds.filter(r => playedBool(r.played));
    const lastWk = played.length ? Math.max(...played.map(r => Number(r.week))) : NaN;
    const lastRows = mask.filter(r => Number(r.week) === lastWk);
    const lastDiffs = lastRows.map(r => Math.abs(toNum(r.model_spread_book) - toNum(r.market_spread_book))).filter(Number.isFinite);
    const lastWeek = lastDiffs.length ? lastDiffs.reduce((a,b)=>a+b,0)/lastDiffs.length : NaN;

    return { overall, byBucket: buckets, lastWeek };
  }, [preds]);

  const weekly = useMemo(() => {
    const by: Record<string, {w:number,l:number,p:number}> = {};
    for (const r of preds) {
      if (!playedBool(r.played)) continue;
      const wk = r.week;
      const res = r.model_result;
      if (!by[wk]) by[wk] = { w:0, l:0, p:0 };
      if (res === "CORRECT") by[wk].w++;
      else if (res === "INCORRECT") by[wk].l++;
      else by[wk].p++;
    }
    const rows = Object.keys(by).sort((a,b)=>Number(a)-Number(b)).map(wk => {
      const {w,l,p} = by[wk];
      const tot = w + l;
      const acc = tot ? w/tot : NaN;
      return { week: wk, wins: w, losses: l, pushes: p, accuracy: acc };
    });
    return rows;
  }, [preds]);

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

          <div className="subcards">
            <div className="subcard">
              <div className="subcard-title">MAE (Model vs Market)</div>
              <div className="kv"><div className="k">Overall</div><div className="v">{fmtNum(mae.overall)}</div></div>
              <div className="kv"><div className="k">Last completed week</div><div className="v">{fmtNum(mae.lastWeek)}</div></div>
              <div className="kv"><div className="k">By market size</div>
                <div className="v">
                  {mae.byBucket.map(b => <span key={b.bucket} style={{marginRight:8}}><b>{b.bucket}</b>: {fmtNum(b.mae)}</span>)}
                </div>
              </div>
            </div>

            <div className="subcard">
              <div className="subcard-title">Weekly Accuracy (Model)</div>
              {!weekly.length ? <div className="note">No completed games yet.</div> : (
                <div className="table-wrap">
                  <table className="tbl compact">
                    <thead><tr><th>Week</th><th>W</th><th>L</th><th>Push</th><th>Accuracy</th></tr></thead>
                    <tbody>
                      {weekly.map(r => (
                        <tr key={r.week}>
                          <td>{r.week}</td><td>{r.wins}</td><td>{r.losses}</td><td>{r.pushes}</td>
                          <td>{fmtPct01(r.accuracy)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
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

/* ------------------------- Predictions tab (week selector) ------------------------- */
function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [week, setWeek] = useState<string>("");

  useEffect(() => { (async () => {
    try {
      const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
      setRows(r);
      const wk = nextUpcomingWeek(r);
      if (wk) setWeek(String(wk));
      else if (r.length) setWeek(String(Math.max(...r.map(x => Number(x.week)))));
    } catch { setRows([]); }
  })(); }, []);

  const weeks = useMemo(() => Array.from(new Set(rows.map(r => r.week))).sort((a,b)=>Number(a)-Number(b)), [rows]);

  const filtered = useMemo(() => {
    if (!week) return [];
    return rows.filter(r => String(r.week) === String(week));
  }, [rows, week]);

  return (
    <section className="card">
      <div className="card-title">Predictions (Book-style)</div>
      <div className="controls">
        <label>Week
          <select className="input" value={week} onChange={(e)=>setWeek(e.target.value)}>
            {weeks.map(w => <option key={w} value={w}>{w}</option>)}
          </select>
        </label>
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
              const model = toNum(r.model_spread_book);
              const market = toNum(r.market_spread_book);
              const expected = toNum(r.expected_market_spread_book);
              const edge = toNum(r.edge_points_book);
              const value = toNum(r.value_points_book);
              const played = playedBool(r.played);
              const neutral = r.neutral_site === "1" || r.neutral_site === "true";
              const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
              const badge = played ? (r.model_result === "CORRECT" ? <Badge tone="pos">✓ Model</Badge> : r.model_result === "INCORRECT" ? <Badge tone="neg">✗ Model</Badge> : null) : null;

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

/* ------------------------- Team (schedule) tab ------------------------- */
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
                const model = toNum(g.model_spread_book) * sgn;
                const mkt   = toNum(g.market_spread_book) * sgn;
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

/* ------------------------- Live Edge tab ------------------------- */
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

/* ------------------------- Recommended Bets (strong only, DQ via stdev) ------------------------- */
type StakeMode = "flat" | "prop" | "kelly";
function BetsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [bankroll, setBankroll] = useState<string>("1000");
  const [mode, setMode] = useState<StakeMode>("prop");
  const [odds, setOdds] = useState<string>("-110");
  const [topN, setTopN] = useState<string>("10");

  useEffect(()=>{(async()=>{
    try { setRows((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setRows([]); }
  })();},[]);

  const wk = useMemo(()=> nextUpcomingWeek(rows), [rows]);

  const candidatesRaw = useMemo(()=>{
    if (!wk) return [];
    return rows.filter(r =>
      Number(r.week) === wk &&
      r.market_spread_book !== undefined && r.market_spread_book !== "" &&
      !playedBool(r.played)
    );
  }, [rows, wk]);

  // Compute week distribution of |edge|
  const strongFiltered = useMemo(()=>{
    if (!candidatesRaw.length) return [];

    const edges = candidatesRaw.map(r => Math.abs(toNum(r.edge_points_book)));
    const mean = edges.filter(Number.isFinite).reduce((a,b)=>a+b,0) / Math.max(1, edges.filter(Number.isFinite).length);
    const variance = edges.filter(Number.isFinite).reduce((a,b)=>a + (b-mean)*(b-mean), 0) / Math.max(1, edges.filter(Number.isFinite).length);
    const sd = Math.sqrt(variance);

    const isDQ = (r: PredRow) => {
      const e = Math.abs(toNum(r.edge_points_book));
      const v = Math.abs(toNum(r.value_points_book));
      if (!Number.isFinite(e) || !Number.isFinite(v)) return true;
      if (e > mean + 2.5*sd) return true; // outlier edge
      if (v > 40) return true; // absurd value -> likely bad market scrape
      return false;
    };

    // Rank: Qualified ✓ first, then |edge| desc, then |value| desc; exclude DQ
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
        week: r.week,
        date: r.date,
        matchup: `${r.away_team} @ ${r.home_team}`,
        side,
        model_line_book: toNum(r.model_spread_book),
        market_line_book: toNum(r.market_spread_book),
        edge_points: toNum(r.edge_points_book),
        value_points: toNum(r.value_points_book),
        qualified: r.qualified_edge_flag === "1" ? "✓" : "—",
        odds: oddsAm,
        stake: Math.round(stake * 100) / 100,
      };
    });
    return alloc;
  }, [strongFiltered, bankroll, mode, odds]);

  const totalStake = useMemo(()=> staked.reduce((a,b)=>a + (b.stake||0), 0), [staked]);

  return (
    <section className="card">
      <div className="card-title">Recommended Bets (Strong Only)</div>
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
                const neutral = r.neutral_site === "1" || r.neutral_site === "true";
                const model = toNum(r.model_spread_book);
                const market = toNum(r.market_spread_book);
                const edge = toNum(r.edge_points_book);
                const value = toNum(r.value_points_book);
                const qual = r.qualified_edge_flag === "1" ? "✓" : "—";
                const sideHome = model < 0;
                const side = sideHome ? `${r.home_team} (home)` : `${r.away_team} (away)`;
                const stakeRow = staked[i];

                return (
                  <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}`} className={i%2?"alt":undefined}>
                    <td>{r.week}</td>
                    <td>{r.date}</td>
                    <td style={{ textAlign: "right" }}><TeamLabel home={false} team={r.away_team} neutral={false} /></td>
                    <td style={{ textAlign: "left" }}><TeamLabel home={true} team={r.home_team} neutral={!!neutral} /></td>
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

/* ------------------------- Help tab ------------------------- */
const EDGE_MIN = 2.0;
const VALUE_MIN = 1.0;
function HelpTab() {
  return (
    <section className="card">
      <div className="card-title">Indicators & Definitions</div>
      <div className="note">
        <p><b>Book-style spread</b>: Negative favors the <b>home</b> team; positive favors the <b>away</b> team.</p>
        <ul>
          <li><b>Model (H)</b>: Our predicted book-style spread for the home team. Combines an advanced-metrics feature model (ridge) and our rating model, then calibrated weekly.</li>
          <li><b>Market (H)</b>: Consensus market spread (home perspective). If missing, edges/values won’t populate.</li>
          <li><b>Expected (H)</b>: “What the market should be” based on our advantage signal mapped to market.</li>
          <li><b>Edge</b>: Model − Market (book-style). Positive = we like the away side more than market; negative = we like home more.</li>
          <li><b>Value</b>: Market − Expected. Positive = price better than our expectation (value to the away side), negative = value to home.</li>
          <li><b>Qualified</b>: “✓” only when model & expected agree on side and |Edge| ≥ {EDGE_MIN} & |Value| ≥ {VALUE_MIN}.</li>
          <li><b>HFA</b>: Home-field advantage, estimated by conference with shrinkage; neutral-site = 0.</li>
          <li><b>Advanced metrics used</b>: team PPA/EPA (off/def + rush/pass), Success Rate, Explosiveness, Starting Field Position, Havoc, Pregame WP (priors).</li>
        </ul>
      </div>
    </section>
  );
}

/* ------------------------- Shell ------------------------- */
export default function App() {
  const [tab, setTab] = useState<Tab>("preds");
  return (
    <div className="page">
      <header className="header">
        <h1>UPA-F Dashboard</h1>
        <p className="sub">Calibrated predictions (book-style), value, results & recommended bets</p>
      </header>
      <nav className="tabs">
        <button className={tab==="status"?"active":""} onClick={()=>setTab("status")}>Status</button>
        <button className={tab==="team"?"active":""} onClick={()=>setTab("team")}>Team (Schedule)</button>
        <button className={tab==="preds"?"active":""} onClick={()=>setTab("preds")}>Predictions</button>
        <button className={tab==="edge"?"active":""} onClick={()=>setTab("edge")}>Live Edge</button>
        <button className={tab==="bets"?"active":""} onClick={()=>setTab("bets")}>Recommended Bets</button>
        <button className={tab==="help"?"active":""} onClick={()=>setTab("help")}>Help</button>
      </nav>
      {tab==="status" && <StatusTab/>}
      {tab==="team" && <TeamTab/>}
      {tab==="preds" && <PredictionsTab/>}
      {tab==="edge" && <LiveEdgeTab/>}
      {tab==="bets" && <BetsTab/>}
      {tab==="help" && <HelpTab/>}
      <footer className="footer">© {new Date().getFullYear()} UPA-F</footer>
    </div>
  );
}