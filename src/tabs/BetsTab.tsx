import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel, nextUpcomingWeek } from "../lib/ui";

const EDGE_THRESHOLD = 1.5;
const VALUE_THRESHOLD = 1.0;

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
  week: string;
  date: string;
  neutral_site?: string;
  home_team: string;
  away_team: string;
  model_spread_book?: string;
  market_spread_book?: string;
  model_spread_baseline?: string;
  market_adjustment?: string;
  expected_market_spread_book?: string;
  edge_points_book?: string;
  value_points_book?: string;
  model_confidence?: string;
  qualified_edge_flag?: string;
  kickoff_utc?: string;
  start_date?: string;
};

const confidenceBucket = (conf: number | null) => {
  if (!Number.isFinite(conf)) return "low";
  if (conf >= 0.75) return "high";
  if (conf >= 0.55) return "medium";
  return "low";
};

export default function BetsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [wk, setWk] = useState<number | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const preds = (await loadTable("upa_predictions")) as PredRow[];
        setRows(preds);
        const next = nextUpcomingWeek(preds as any);
        setWk(next);
      } catch {
        setRows([]);
        setWk(null);
      }
    })();
  }, []);

  const weekOptions = useMemo(() => {
    const weeks = Array.from(new Set(rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x)))).sort((a, b) => a - b);
    return weeks as number[];
  }, [rows]);

  const cards = useMemo(() => {
    if (!rows.length) return [] as any[];
    const selected = rows.filter((r) => (wk ? Number(r.week) === wk : true));

    return selected
      .map((r) => {
        const model = toNum(r.model_spread_book);
        const market = toNum(r.market_spread_book);
        const baseline = toNum((r as any).model_spread_baseline);
        const adjustment = toNum((r as any).market_adjustment);
        const edge = toNum(r.edge_points_book);
        const value = toNum(r.value_points_book);
        const confidence = toNum((r as any).model_confidence);
        const anchored = model;
        const delta = Number.isFinite(model) && Number.isFinite(market) ? model - market : NaN;
        const play = Number.isFinite(edge) && Number.isFinite(value) && Math.abs(edge) >= EDGE_THRESHOLD && Math.abs(value) >= VALUE_THRESHOLD;

        return {
          ...r,
          _model: model,
          _market: market,
          _anchored: anchored,
          _baseline: baseline,
          _adjustment: adjustment,
          _edge: edge,
          _value: value,
          _delta: delta,
          _confidence: confidence,
          _confidenceBucket: confidenceBucket(confidence),
          _play: play,
          _source: providerLabel((r as any).market_spread_source),
        };
      })
      .filter((r) => r._play)
      .sort((a, b) => {
        const confDiff = (b._confidence ?? 0) - (a._confidence ?? 0);
        if (Math.abs(confDiff) > 1e-6) return confDiff;
        const edgeDiff = Math.abs(b._edge ?? 0) - Math.abs(a._edge ?? 0);
        if (Math.abs(edgeDiff) > 1e-6) return edgeDiff;
        return Math.abs(b._value ?? 0) - Math.abs(a._value ?? 0);
      });
  }, [rows, wk]);

  return (
    <section className="page">
      <div className="header">
        <div>
          <h1>Recommended Bets</h1>
          <p className="sub">Confidence-weighted edges pulled from the latest predictions</p>
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
        <Badge tone="muted">Plays: {cards.length}</Badge>
      </div>

      <div className="note">Showing plays where |edge| ≥ 1.5 and |value| ≥ 1.0. Negative edges point to the home side.</div>

      {!cards.length && (
        <div className="note">No recommended edges for this week. Check back after markets update.</div>
      )}

      <div className="pred-grid">
        {cards.map((card) => {
          const key = `${card.week}-${card.home_team}-${card.away_team}`;
          const isOpen = expanded === key;
          const toggle = () => setExpanded((prev) => (prev === key ? null : key));

          return (
            <div key={key} className={`pred-card pred-card--bets pred-card--confidence-${card._confidenceBucket}`}>
              <div className="pred-card__header">
                <div>
                  <div className="pred-card__week">Week {card.week}</div>
                  <div className="pred-card__date">{card.date || "TBD"}</div>
                </div>
                <div className="pred-card__header-meta">
                  <Badge tone="muted">{card._source}</Badge>
                </div>
              </div>

              <div className="pred-card__teams">
                <div className="pred-card__team pred-card__team--away">
                  <div className="pred-card__team-role">Away</div>
                  <TeamLabel home={false} team={card.away_team} neutral={false} showTags={false} />
                  <div className="pred-card__score">{fmtNum(undefined)}</div>
                </div>
                <div className="pred-card__match-info">
                  <div className="pred-card__kick">{card.date || ""}</div>
                  <div className="pred-card__scoreline">{card._delta ? `Delta: ${fmtNum(card._delta)}` : "Spread delta"}</div>
                  <div className="pred-card__live-tag">{card.neutral_site === "1" || card.neutral_site === "true" ? "Neutral" : ""}</div>
                </div>
                <div className="pred-card__team pred-card__team--home">
                  <div className="pred-card__team-role">Home</div>
                  <TeamLabel home={true} team={card.home_team} neutral={card.neutral_site === "1" || card.neutral_site === "true"} showTags={false} />
                  <div className="pred-card__score">{fmtNum(undefined)}</div>
                </div>
              </div>

              <div className="pred-card__summary">
                <div className="metric"><span className="label">Market</span><span>{fmtNum(card._market)}</span></div>
                <div className="metric"><span className="label">Anchored</span><span>{fmtNum(card._anchored)}</span></div>
                <div className="metric"><span className="label">Baseline</span><span>{fmtNum(card._baseline)}</span></div>
                <div className="metric"><span className="label">Adj Δ</span><span>{fmtNum(card._adjustment)}</span></div>
                <div className="metric"><span className="label">Edge</span><span>{fmtNum(card._edge)}</span></div>
                <div className="metric"><span className="label">Value</span><span>{fmtNum(card._value)}</span></div>
                <div className="metric"><span className="label">Confidence</span><span>{card._confidence ? `${Math.round(card._confidence * 100)}%` : "—"}</span></div>
                <div className="metric metric--pick"><span className="label">Suggested Side</span><span>{card._edge > 0 ? `${card.away_team} (away)` : `${card.home_team} (home)`}</span></div>
              </div>

              <button className="pred-card__toggle" onClick={toggle}>
                {isOpen ? "Hide details" : "More details"}
              </button>

              <div className={`pred-card__details${isOpen ? " open" : ""}`}>
                <div className="pred-card__notes">
                  <div className="pred-card__notes-title">Why we like it</div>
                  <div className="pred-card__notes-grid">
                    <div className="pred-card__notes-row"><span>Market vs Model</span><span>{fmtNum(card._delta)}</span></div>
                    <div className="pred-card__notes-row"><span>Anchored Spread</span><span>{fmtNum(card._anchored)}</span></div>
                    <div className="pred-card__notes-row"><span>Baseline Spread</span><span>{fmtNum(card._baseline)}</span></div>
                    <div className="pred-card__notes-row"><span>Adj Δ</span><span>{fmtNum(card._adjustment)}</span></div>
                    <div className="pred-card__notes-row"><span>Confidence bucket</span><span>{card._confidenceBucket}</span></div>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
