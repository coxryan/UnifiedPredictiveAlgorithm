import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel, nextUpcomingWeek } from "../lib/ui";

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

type LiveRow = {
  state?: string;
  away_school?: string;
  home_school?: string;
  away_points?: string | number;
  home_points?: string | number;
};

type PredRow = {
  week: string;
  date: string;
  neutral_site?: string;
  home_team: string;
  away_team: string;
  played?: any;
  model_spread_book?: string;
  model_spread_baseline?: string;
  market_adjustment?: string;
  model_confidence?: string;
  market_spread_book?: string;
  expected_market_spread_book?: string;
  edge_points_book?: string;
  value_points_book?: string;
  qualified_edge_flag?: string | number;
  market_spread_source?: string | null;
  home_points?: string;
  away_points?: string;
};

type PositionKey = "qb" | "rb" | "wr" | "ol" | "dl" | "lb" | "db" | "st";

const POSITION_KEYS: Array<{ key: PositionKey; label: string }> = [
  { key: "qb", label: "QB" },
  { key: "rb", label: "RB" },
  { key: "wr", label: "WR" },
  { key: "ol", label: "OL" },
  { key: "dl", label: "DL" },
  { key: "lb", label: "LB" },
  { key: "db", label: "DB" },
  { key: "st", label: "ST" },
];

const gradeClassName = (letter: string) => {
  const clean = (letter || "").toString().trim();
  if (!clean) return "";
  return ` grade-${clean.replace("+", "p").replace("-", "m").toLowerCase()}`;
};

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome;
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
}

export default function PredictionsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [wk, setWk] = useState<number | null>(null);
  const [onlyQualified, setOnlyQualified] = useState<boolean>(false);
  const [liveRows, setLiveRows] = useState<LiveRow[]>([]);
  const [kickKeyMap, setKickKeyMap] = useState<Record<string, string>>({});
  const [kickIdMap, setKickIdMap] = useState<Record<string, string>>({});
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const raw = (await loadTable("upa_predictions")) as PredRow[];
        const coerceNum = (v: any) => {
          const n = toNum(v);
          return Number.isFinite(n) ? n : null;
        };
        const normalizeSource = (v: any): string | null => {
          const key = (v ?? "").toString().trim().toLowerCase();
          return key ? key : null;
        };
        const flagVal = (v: any): number => {
          const n = Number(v);
          return Number.isFinite(n) ? n : 0;
        };
        const normalized = raw.map((x: PredRow) => ({
          ...x,
          model_spread_book: coerceNum(x.model_spread_book) as any,
          model_spread_baseline: coerceNum((x as any).model_spread_baseline) as any,
          market_adjustment: coerceNum((x as any).market_adjustment) as any,
          market_spread_book: coerceNum(x.market_spread_book) as any,
          expected_market_spread_book: coerceNum(x.expected_market_spread_book) as any,
          edge_points_book: coerceNum(x.edge_points_book) as any,
          value_points_book: coerceNum(x.value_points_book) as any,
          market_spread_source: normalizeSource((x as any).market_spread_source),
          model_confidence: coerceNum((x as any).model_confidence) as any,
          qualified_edge_flag: flagVal((x as any).qualified_edge_flag),
        }));
        setRows(normalized);

        const nextWk = nextUpcomingWeek(normalized as any);
        if (nextWk) {
          setWk(nextWk);
        } else {
          const weeks = Array.from(new Set(normalized.map((x) => Number(x.week)).filter((x) => Number.isFinite(x)))).sort((a, b) => a - b);
          setWk(weeks.length ? weeks[0] : null);
        }
      } catch {
        setRows([]);
        setWk(null);
      }

      try {
        const l = (await loadTable("live_scores")) as LiveRow[];
        setLiveRows(l || []);
      } catch {
        setLiveRows([]);
      }

      try {
        const sched = (await loadTable("cfb_schedule")) as any[];
        const byKey: Record<string, string> = {};
        const byId: Record<string, string> = {};
        const norm = (s: any) => (s ?? "").toString().trim();
        const key = (w: any, a: any, h: any) => `${Number(w) || 0}|${norm(a)}|${norm(h)}`;
        function combineDateTime(dateStr: string, timeStr: string): string {
          if (!dateStr) return "";
          if (!timeStr) return dateStr;
          if (/(\d{1,2}):(\d{2})/.test(dateStr)) return dateStr;
          return `${dateStr} ${timeStr}`;
        }
        for (const r of sched || []) {
          let dateStr = (r.kickoff_utc ?? r.start_date ?? r.datetime ?? r.date ?? "").toString().trim();
          if (!dateStr) continue;
          let timeStr = "";
          if (!/(\d{1,2}):(\d{2})/.test(dateStr)) {
            timeStr = (r.kickoff_et ?? r.kickoff_time ?? r.start_time ?? r.time ?? r.kick_time ?? "").toString().trim();
          }
          const dt = timeStr ? combineDateTime(dateStr, timeStr) : dateStr;
          byKey[key(r.week, r.away_team, r.home_team)] = dt;
          if (r.game_id != null && r.game_id !== "") byId[String(r.game_id)] = dt;
        }
        setKickKeyMap(byKey);
        setKickIdMap(byId);
      } catch {
        setKickKeyMap({});
        setKickIdMap({});
      }
    })();
  }, []);

  const weekOptions = useMemo(() => {
    const w = Array.from(new Set(rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x)))).sort((a, b) => a - b);
    return w as number[];
  }, [rows]);

  const liveMap = useMemo(() => {
    const m = new Map<string, { hp: number | null; ap: number | null; state: string }>();
    for (const r of liveRows || []) {
      const a = (r.away_school || "").toString().trim();
      const h = (r.home_school || "").toString().trim();
      if (!a || !h) continue;
      const hp = toNum(r.home_points);
      const ap = toNum(r.away_points);
      const st = (r.state || "").toString().toLowerCase();
      m.set(`${a}|${h}`, { hp: Number.isFinite(hp) ? hp : null, ap: Number.isFinite(ap) ? ap : null, state: st });
    }
    return m;
  }, [liveRows]);

  const keyOf = (w: any, a: any, h: any) => `${Number(w) || 0}|${String(a || "").trim()}|${String(h || "").trim()}`;

  function kickoffLabelET(dateStr?: string): string {
    if (!dateStr) return "";
    const s = dateStr.toString().trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return "";
    const clock = s.match(/(\d{1,2}):(\d{2})/);
    const hasTZ = /Z|[+-]\d{2}:?\d{2}$/.test(s);
    if (hasTZ) {
      const d = new Date(s);
      if (!isNaN(d.getTime())) {
        try {
          const t = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" }).format(d);
          return `${t} ET`;
        } catch {
          // ignore
        }
      }
      return clock ? `${clock[1]}:${clock[2]} ET` : "";
    }
    if (clock) return `${clock[1]}:${clock[2]} ET`;
    const d = new Date(s);
    if (!isNaN(d.getTime())) {
      try {
        const t = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", hour: "numeric", minute: "2-digit" }).format(d);
        return `${t} ET`;
      } catch {
        // ignore
      }
    }
    return "";
  }

  const cards = useMemo(() => {
    const filtered = rows.filter((r) => (wk ? Number(r.week) === wk : true));
    const baseline = onlyQualified ? filtered.filter((r) => Number(r.qualified_edge_flag) === 1) : filtered;
    return baseline.map((r: any) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const expected = toNum(r.expected_market_spread_book);
      const anchored = toNum(r.model_spread_book);
      const baselineModel = toNum((r as any).model_spread_baseline);
      const adjustment = toNum((r as any).market_adjustment);
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
      const marketSource = providerLabel((r as any).market_spread_source);
      const confidence = toNum((r as any).model_confidence);
      const confidencePct = Number.isFinite(confidence) ? Math.round(confidence * 100) : null;

      const schedDt = (r as any).kickoff_utc
        || (r as any).start_date
        || kickIdMap[String((r as any).game_id || "")]
        || kickKeyMap[keyOf(r.week, r.away_team, r.home_team)]
        || r.date;
      const kickLabel = kickoffLabelET(schedDt);

      const key = `${(r.away_team || "").toString().trim()}|${(r.home_team || "").toString().trim()}`;
      const live = liveMap.get(key);
      let hp = toNum(r.home_points);
      let ap = toNum(r.away_points);
      if (live && (live.state === "in" || live.state === "post")) {
        if (Number.isFinite(live.hp)) hp = live.hp as number;
        if (Number.isFinite(live.ap)) ap = live.ap as number;
      }

      const homePointsLabel = Number.isFinite(hp) ? fmtNum(hp, { maximumFractionDigits: 0 }) : "—";
      const awayPointsLabel = Number.isFinite(ap) ? fmtNum(ap, { maximumFractionDigits: 0 }) : "—";
      const scoreLabel = Number.isFinite(hp) && Number.isFinite(ap) ? `${awayPointsLabel} @ ${homePointsLabel}` : "—";

      const positions = POSITION_KEYS.map(({ key, label }) => {
        const awayLetter = ((r as any)[`away_grade_${key}_letter`] ?? "").toString().trim() || "—";
        const homeLetter = ((r as any)[`home_grade_${key}_letter`] ?? "").toString().trim() || "—";
        const awayScore = toNum((r as any)[`away_grade_${key}_score`]);
        const homeScore = toNum((r as any)[`home_grade_${key}_score`]);
        let advantage: "away" | "home" | "even" = "even";
        if (Number.isFinite(homeScore) && Number.isFinite(awayScore)) {
          const delta = homeScore - awayScore;
          if (delta > 1.5) advantage = "home";
          else if (delta < -1.5) advantage = "away";
        }
        return { key, label, awayLetter, homeLetter, advantage };
      });

      return {
        ...r,
        _model: model,
        _market: market,
        _anchored: anchored,
        _baseline: baselineModel,
        _adjustment: adjustment,
        _expected: expected,
        _edge: edge,
        _value: value,
        _pick: pick.side,
        _marketSource: marketSource,
        _kick: kickLabel,
        _scoreLabel: scoreLabel,
        _homePointsLabel: homePointsLabel,
        _awayPointsLabel: awayPointsLabel,
        _liveState: (live?.state || "").toString().toUpperCase(),
        _positions: positions,
        _confidence: confidence,
        _confidencePct: confidencePct,
      };
    });
  }, [rows, wk, onlyQualified, kickKeyMap, kickIdMap, liveMap]);

  return (
    <section className="page">
      <div className="header">
        <div>
          <h1>Predictions</h1>
          <p className="sub">Model-calibrated spreads, edge, and value</p>
        </div>
      </div>

      <div className="controls">
        <label>Week
          <select className="input" value={wk ?? ""} onChange={(e) => setWk(e.target.value ? Number(e.target.value) : null)}>
            {(weekOptions.length ? weekOptions : [wk ?? 1]).map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </label>
        <label className="chk">
          <input type="checkbox" checked={onlyQualified} onChange={(e) => setOnlyQualified(e.target.checked)} />
          Exclude non-qualified (show only ✓)
        </label>
        <Badge tone="muted">Rows: {cards.length}</Badge>
      </div>

      {!rows.length && (
        <div className="note" style={{ marginBottom: 8 }}>No predictions found. Expecting dataset <code>upa_predictions</code>.</div>
      )}

      <div className="pred-grid">
        {cards.map((card: any) => {
          const key = card.game_id || `${card.week}-${card.home_team}-${card.away_team}`;
          const qualified = Number(card.qualified_edge_flag) === 1;
          const isOpen = expanded === key;

          const toggleDetails = () => setExpanded((prev) => (prev === key ? null : key));

          const gradeBadge = (letter: string, highlight = false) => (
            <span className={`pred-card__grade${gradeClassName(letter)}${highlight ? " highlight" : ""}`}>
              {(letter || "—").toString().trim() || "—"}
            </span>
          );

          return (
            <div key={key} className={`pred-card${qualified ? " pred-card--qualified" : ""}`}>
              <div className="pred-card__header">
                <div>
                  <div className="pred-card__week">Week {card.week}</div>
                  <div className="pred-card__date">{card.date || "TBD"}</div>
                </div>
                <div className="pred-card__header-meta">
                  {qualified ? <Badge tone="pos">Qualified</Badge> : null}
                  <Badge tone="muted">{card._marketSource}</Badge>
                </div>
              </div>

              <div className="pred-card__teams">
                <div className="pred-card__team pred-card__team--away">
                  <div className="pred-card__team-role">Away</div>
                  <TeamLabel home={false} team={card.away_team} neutral={false} showTags={false} />
                  <div className="pred-card__score">{card._awayPointsLabel}</div>
                </div>
                <div className="pred-card__match-info">
                  <div className="pred-card__kick">{card._kick || "Kickoff TBA"}</div>
                  <div className="pred-card__scoreline">{card._scoreLabel}</div>
                  <div className="pred-card__live-tag">{card._liveState || ""}</div>
                </div>
                <div className="pred-card__team pred-card__team--home">
                  <div className="pred-card__team-role">Home</div>
                  <TeamLabel home={true} team={card.home_team} neutral={card.neutral_site === "1" || card.neutral_site === "true"} showTags={false} />
                  <div className="pred-card__score">{card._homePointsLabel}</div>
                  {card.neutral_site === "1" || card.neutral_site === "true" ? (
                    <div className="pred-card__team-tags"><Badge tone="muted">NEUTRAL</Badge></div>
                  ) : null}
                </div>
              </div>

              <div className="pred-card__summary">
                <div className="metric"><span className="label">Market</span><span>{fmtNum(card._market)}</span></div>
                <div className="metric"><span className="label">Adj Δ</span><span>{fmtNum(card._adjustment)}</span></div>
                <div className="metric"><span className="label">Anchored</span><span>{fmtNum(card._anchored)}</span></div>
                <div className="metric"><span className="label">Baseline</span><span>{fmtNum(card._baseline)}</span></div>
                <div className="metric"><span className="label">Edge</span><span>{fmtNum(card._edge)}</span></div>
                <div className="metric"><span className="label">Value</span><span>{fmtNum(card._value)}</span></div>
                <div className="metric"><span className="label">Confidence</span><span>{Number.isFinite(card._confidence) && card._confidencePct != null ? `${card._confidencePct}%` : "—"}</span></div>
                <div className="metric metric--pick"><span className="label">Pick</span><span>{card._pick || "—"}</span></div>
              </div>

              <button className="pred-card__toggle" onClick={toggleDetails}>
                {isOpen ? "Hide details" : "More details"}
              </button>

              <div className={`pred-card__details${isOpen ? " open" : ""}`}>
                <div className="pred-card__grades">
                  <div className="pred-card__grades-title">Positional Grades</div>
                  <div className="pred-card__grades-grid">
                    {card._positions.map((pos: any) => {
                      const awayStrong = pos.advantage === "away";
                      const homeStrong = pos.advantage === "home";
                      return (
                        <div key={pos.key} className="pred-card__grades-row">
                          {gradeBadge(pos.awayLetter, awayStrong)}
                          <span className="pred-card__grades-label">{pos.label}</span>
                          {gradeBadge(pos.homeLetter, homeStrong)}
                        </div>
                      );
                    })}
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

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads shown (negative = home favorite). <b>Adjustment</b> models the residual gap (market − model); the anchored model applies that delta to the live market line. <b>Edge</b> = model − market (home perspective). <b>Value</b> = market − expected. Positional letters compare normalized team units (A+ best, F worst), highlighting advantages head-to-head.
      </div>
    </section>
  );
}
