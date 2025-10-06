import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel } from "../lib/ui";

// Join ESPN live scores with our predictions by (away_school, home_school)
// and filter to FBS-vs-FBS using the current season team inputs file.

type LiveRow = {
  date?: string; state?: string; detail?: string; clock?: string; period?: string;
  venue?: string; event_id?: string;
  away_team?: string; home_team?: string;           // ESPN display (e.g., "Pittsburgh Panthers")
  away_school?: string; home_school?: string;       // normalized school (e.g., "Pittsburgh")
  away_points?: string | number; home_points?: string | number;
};

const providerLabel = (src: any): string => {
  const key = (src ?? "").toString().trim().toLowerCase();
  switch (key) {
    case "fanduel":
      return "FanDuel";
    case "cfbd":
      return "CFBD";
    case "model":
      return "Model";
    case "unknown":
      return "Unknown";
    case "none":
    case "":
      return "—";
    default:
      return key ? key.toUpperCase() : "—";
  }
};

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string;
  market_spread_source?: string | null;
};

type TeamInput = { team: string };

function pickFromModelMarket(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

function coverStatus(hp: number, ap: number, marketHome: number, pickSide: string | undefined) {
  if (!Number.isFinite(hp) || !Number.isFinite(ap) || !Number.isFinite(marketHome)) return "warn"; // missing info
  const adj = (hp - ap) + marketHome;               // book-style: <0 home fails to cover
  const homeCover = adj > 0;
  const pickHome = !!pickSide && pickSide.includes("(home)");
  const pickAway = !!pickSide && pickSide.includes("(away)");
  if (!pickHome && !pickAway) return "warn";
  const onTrack = pickHome ? homeCover : !homeCover;
  return onTrack ? "ok" : "bad";
}

const stateLabel = (state: string | undefined): string => {
  const key = (state || "").toString().toLowerCase();
  switch (key) {
    case "pre":
      return "Pre-game";
    case "in":
      return "In Progress";
    case "post":
      return "Final";
    default:
      return key ? key.toUpperCase() : "Status";
  }
};

const periodLabel = (period?: string | number): string | null => {
  const value = typeof period === "string" ? period.trim() : period;
  if (value === undefined || value === null || value === "") return null;
  const num = Number(value);
  if (!Number.isFinite(num)) return typeof value === "string" ? value : null;
  switch (num) {
    case 1: return "Q1";
    case 2: return "Q2";
    case 3: return "Q3";
    case 4: return "Q4";
    default:
      return num > 4 ? `OT${num - 4}` : `P${num}`;
  }
};

const clockLabel = (row: LiveRow): string => {
  const state = (row.state || "").toLowerCase();
  if (state === "pre") return row.detail || "Scheduled";
  if (state === "post") return row.detail || "Final";
  const clock = (row.clock || "").toString().trim();
  const period = periodLabel(row.period);
  if (clock && period) return `${period} ${clock}`;
  if (clock) return clock;
  if (period) return period;
  return row.detail || "Live";
};

export default function LiveResultsTab() {
  const [live, setLive] = useState<LiveRow[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [fbs, setFbs] = useState<Set<string>>(new Set());
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try { setLive((await loadTable("live_scores")) as LiveRow[]); } catch { setLive([]); }
      try { setPreds((await loadTable("upa_predictions")) as PredRow[]); } catch { setPreds([]); }
      try {
        const t = (await loadTable("upa_team_inputs_datadriven_v0")) as TeamInput[];
        setFbs(new Set((t || []).map((x:any)=> (x.team||"").toString().trim())));
      } catch { setFbs(new Set()); }
    })();
  }, []);

  const lastUpdated = useMemo(() => {
    const times = (live || [])
      .map(r => {
        try { return Date.parse(String(r.date || "")); } catch { return NaN; }
      })
      .filter((n) => Number.isFinite(n) && !Number.isNaN(n));
    if (!times.length) return null;
    return new Date(Math.max(...(times as number[])));
  }, [live]);

  const lastUpdatedLabel = useMemo(
    () => (lastUpdated ? lastUpdated.toLocaleString() : "—"),
    [lastUpdated]
  );

  const predByMatch = useMemo(() => {
    const key = (a:string,h:string)=> `${a}|${h}`;
    const map = new Map<string, { model:number, market:number, source:string, week?:string, date?:string }>();
    for (const r of preds) {
      const a = (r.away_team||"").toString().trim();
      const h = (r.home_team||"").toString().trim();
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const src = providerLabel((r as any).market_spread_source);
      if (a && h) map.set(key(a,h), { model, market, source: src, week: r.week, date: r.date });
    }
    return map;
  }, [preds]);

  const cards = useMemo(() => {
    const isFbs = (s?:string)=> !!s && fbs.has((s||"").toString().trim());

    return (live||[])
      .filter(r => isFbs(r.home_school||r.home_team) && isFbs(r.away_school||r.away_team))
      .map(r => {
        const aSchool = (r.away_school || r.away_team || "").toString().trim();
        const hSchool = (r.home_school || r.home_team || "").toString().trim();
        const match = predByMatch.get(`${aSchool}|${hSchool}`);
        const model = match ? match.model : NaN;
        const market = match ? match.market : NaN;
        const source = match ? match.source : "—";
        const week = match?.week || "—";
        const gameDate = match?.date || r.date || "";
        const pick = Number.isFinite(model) && Number.isFinite(market)
          ? pickFromModelMarket(model, market, hSchool, aSchool).side
          : "—";
        const hp = typeof r.home_points === 'string' ? Number(r.home_points) : (r.home_points as number);
        const ap = typeof r.away_points === 'string' ? Number(r.away_points) : (r.away_points as number);
        const tone = coverStatus(Number(hp), Number(ap), Number(market), pick);
        const edge = Number.isFinite(model) && Number.isFinite(market) ? model - market : NaN;

        return {
          ...r,
          _model: model,
          _market: market,
          _edge: edge,
          _pick: pick,
          _tone: tone,
          _marketSource: source,
          _week: week,
          _predDate: gameDate,
        } as any;
      });
  }, [live, predByMatch, fbs]);

  return (
    <section className="page">
      <div className="header">
        <div>
          <h1>Live Results</h1>
          <p className="sub">Scores joined with model & market context</p>
        </div>
        <Badge tone="muted">Updated {lastUpdatedLabel}</Badge>
      </div>

      {!cards.length && (
        <div className="note">No live rows found. Expecting dataset <code>live_scores</code>.</div>
      )}

      <div className="pred-grid">
        {cards.map((g:any, idx:number) => {
          const key = g.event_id || g.date || idx;
          const toneClass = g._tone === "ok" ? " pred-card--tone-ok" : g._tone === "bad" ? " pred-card--tone-bad" : " pred-card--tone-warn";
          const isOpen = expanded === key;
          const toggleDetails = () => setExpanded((prev) => (prev === key ? null : key));

          const homeScore = fmtNum(g.home_points, { maximumFractionDigits: 0 });
          const awayScore = fmtNum(g.away_points, { maximumFractionDigits: 0 });
          const state = stateLabel(g.state);
          const clock = clockLabel(g);
          const edge = Number.isFinite(g._edge) ? fmtNum(g._edge) : "—";

          return (
            <div key={key} className={`pred-card pred-card--live${toneClass}`}>
              <div className="pred-card__header">
                <div>
                  <div className="pred-card__week">Week {g._week || "—"}</div>
                  <div className="pred-card__date">{(g._predDate || g.date || "").split('T')[0] || ""}</div>
                </div>
                <div className="pred-card__header-meta">
                  <Badge tone="muted">{providerLabel(g._marketSource)}</Badge>
                </div>
              </div>

              <div className="pred-card__teams">
                <div className="pred-card__team pred-card__team--away">
                  <TeamLabel home={false} team={g.away_school || g.away_team} neutral={false} />
                  <div className="pred-card__score">{awayScore}</div>
                </div>
                <div className="pred-card__match-info">
                  <div className="pred-card__kick">{state}</div>
                  <div className="pred-card__scoreline">{clock}</div>
                  <div className="pred-card__live-tag">{g.detail || ""}</div>
                </div>
                <div className="pred-card__team pred-card__team--home">
                  <TeamLabel home={true} team={g.home_school || g.home_team} neutral={false} />
                  <div className="pred-card__score">{homeScore}</div>
                </div>
              </div>

              <div className="pred-card__summary">
                <div className="metric"><span className="label">Model (H)</span><span>{fmtNum(g._model)}</span></div>
                <div className="metric"><span className="label">Market (H)</span><span>{fmtNum(g._market)}</span></div>
                <div className="metric"><span className="label">Edge</span><span>{edge}</span></div>
                <div className="metric"><span className="label">Pick</span><span>{g._pick || "—"}</span></div>
                <div className="metric"><span className="label">Source</span><span>{providerLabel(g._marketSource)}</span></div>
                <div className="metric"><span className="label">Venue</span><span>{g.venue || "—"}</span></div>
              </div>

              <button className="pred-card__toggle" onClick={toggleDetails}>
                {isOpen ? "Hide details" : "More details"}
              </button>

              <div className={`pred-card__details${isOpen ? " open" : ""}`}>
                <div className="pred-card__notes">
                  <div className="pred-card__notes-title">Game Notes</div>
                  <div className="pred-card__notes-grid">
                    <div className="pred-card__notes-row"><span>State</span><span>{state}</span></div>
                    <div className="pred-card__notes-row"><span>Clock</span><span>{clock}</span></div>
                    <div className="pred-card__notes-row"><span>Detail</span><span>{g.detail || "—"}</span></div>
                    <div className="pred-card__notes-row"><span>Venue</span><span>{g.venue || "—"}</span></div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}

        {!cards.length && (
          <div className="note" style={{ padding: 12 }}>No rows to display.</div>
        )}
      </div>
    </section>
  );
}
