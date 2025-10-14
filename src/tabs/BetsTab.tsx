import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel, nextUpcomingWeek } from "../lib/ui";
import {
  BETS_EDGE_MIN,
  BETS_VALUE_MIN,
  CONFIDENCE_MIN,
  HIGH_CONFIDENCE_MIN,
  HIGH_CONF_VALUE_MIN,
  LARGE_SPREAD_ABS_MIN,
  LARGE_SPREAD_CONF_MIN,
} from "./constants";

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
  played?: string | number;
  model_result?: string;
};

type ViewMode = "cards" | "list";

type BetRow = PredRow & {
  _model: number | null;
  _market: number | null;
  _baseline: number | null;
  _adjustment: number | null;
  _anchored: number | null;
  _edge: number | null;
  _value: number | null;
  _delta: number | null;
  _confidence: number | null;
  _confidenceBucket: string;
  _qualified: boolean;
  _spreadAbs: number | null;
  _source: string;
  _sourceKey: string;
  _pick: string;
  _dateIso: string;
  _spreadBucket: string | null;
  _result: string | null;
};

const confidenceBucket = (conf: number | null) => {
  if (!Number.isFinite(conf)) return "low";
  if (conf >= 0.75) return "high";
  if (conf >= 0.55) return "medium";
  return "low";
};

const SOURCE_OPTIONS = [
  { key: "fanduel", label: "FanDuel" },
  { key: "cfbd", label: "CFBD" },
  { key: "model", label: "Model" },
  { key: "unknown", label: "Unknown" },
];

const spreadBucketFor = (spreadAbs: number | null): string | null => {
  if (!Number.isFinite(spreadAbs)) return null;
  const val = spreadAbs as number;
  if (val < 3) return "<3";
  if (val < 5) return "3-5";
  if (val < 10) return "5-10";
  if (val < 15) return "10-15";
  if (val < 20) return "15-20";
  return "20+";
};

const SPREAD_BUCKETS = [
  { key: "<3", label: "<3", min: 0, max: 3 },
  { key: "3-5", label: "3-5", min: 3, max: 5 },
  { key: "5-10", label: "5-10", min: 5, max: 10 },
  { key: "10-15", label: "10-15", min: 10, max: 15 },
  { key: "15-20", label: "15-20", min: 15, max: 20 },
  { key: "20+", label: "20+", min: 20, max: Number.POSITIVE_INFINITY },
];

export default function BetsTab() {
  const [viewMode, setViewMode] = useState<ViewMode>("cards");
  const [rows, setRows] = useState<PredRow[]>([]);
  const [selectedWeeks, setSelectedWeeks] = useState<number[]>([]);
  const [onlyQualified, setOnlyQualified] = useState<boolean>(false);
  const [highConfidenceOnly, setHighConfidenceOnly] = useState<boolean>(false);
  const [largeSpreadOnly, setLargeSpreadOnly] = useState<boolean>(false);
  const [mostLikelyOnly, setMostLikelyOnly] = useState<boolean>(false);
  const [edgeMin, setEdgeMin] = useState<number>(BETS_EDGE_MIN);
  const [valueMin, setValueMin] = useState<number>(BETS_VALUE_MIN);
  const [confidenceMin, setConfidenceMin] = useState<number>(CONFIDENCE_MIN);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [selectedBands, setSelectedBands] = useState<string[]>(() => SPREAD_BUCKETS.map((b) => b.key));
  const [sourceFilters, setSourceFilters] = useState<Record<string, boolean>>(
    () =>
      SOURCE_OPTIONS.reduce<Record<string, boolean>>((acc, opt) => {
        acc[opt.key] = true;
        return acc;
      }, {})
  );
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const preds = (await loadTable("upa_predictions")) as PredRow[];
        setRows(preds);
      } catch {
        setRows([]);
      }
    })();
  }, []);

  const upcomingWeek = useMemo(() => nextUpcomingWeek(rows as any), [rows]);

  useEffect(() => {
    if (!rows.length) return;
    if (!selectedWeeks.length && upcomingWeek) {
      setSelectedWeeks([upcomingWeek]);
    }
  }, [rows, selectedWeeks.length, upcomingWeek]);

  const weekOptions = useMemo(() => {
    const weeks = Array.from(new Set(rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x)))).sort((a, b) => a - b);
    return weeks as number[];
  }, [rows]);

  const shapedRows = useMemo<BetRow[]>(() => {
    if (!rows.length) return [];
    return rows.map((r) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const baseline = toNum((r as any).model_spread_baseline);
      const adjustment = toNum((r as any).market_adjustment);
      const edge = toNum(r.edge_points_book);
      const value = toNum(r.value_points_book);
      const confidence = toNum((r as any).model_confidence);
      const delta = Number.isFinite(model) && Number.isFinite(market) ? model - market : null;
      const anchored = model;
      const rawSource = (r as any).market_spread_source ?? "";
      const sourceKeyRaw = rawSource ? rawSource.toString().trim().toLowerCase() : "";
      const sourceKey = sourceKeyRaw || "unknown";
      const normalizedSource = providerLabel(rawSource);
      const qualified = Number(r.qualified_edge_flag) === 1;
      const spreadAbs = Number.isFinite(market)
        ? Math.abs(market as number)
        : Number.isFinite(model)
        ? Math.abs(model as number)
        : Number.isFinite(baseline)
        ? Math.abs(baseline as number)
        : null;
      const pick =
        Number.isFinite(edge) && (edge as number) !== 0
          ? edge! > 0
            ? `${r.away_team} (away)`
            : `${r.home_team} (home)`
          : "—";
      const dateIso = (() => {
        const raw = (r.kickoff_utc || r.start_date || r.date || "").toString();
        if (!raw) return "";
        if (raw.length >= 10) return raw.slice(0, 10);
        return raw;
      })();
      const modelResult = (r.model_result || "").toString().trim().toUpperCase();
      const playedFlag = Number(r.played);
      const spreadBucket = spreadBucketFor(spreadAbs);
      return {
        ...r,
        _model: Number.isFinite(model) ? model : null,
        _market: Number.isFinite(market) ? market : null,
        _baseline: Number.isFinite(baseline) ? baseline : null,
        _adjustment: Number.isFinite(adjustment) ? adjustment : null,
        _anchored: Number.isFinite(anchored) ? anchored : null,
        _edge: Number.isFinite(edge) ? edge : null,
        _value: Number.isFinite(value) ? value : null,
        _delta: Number.isFinite(delta) ? delta : null,
        _confidence: Number.isFinite(confidence) ? confidence : null,
        _confidenceBucket: confidenceBucket(confidence),
        _qualified: qualified,
        _spreadAbs: spreadAbs,
        _source: normalizedSource,
        _sourceKey: sourceKey,
        _pick: pick,
        _dateIso: dateIso,
        _spreadBucket: spreadBucket,
        _result: playedFlag === 1 ? modelResult : null,
      };
    });
  }, [rows]);

  const bucketInsights = useMemo(() => {
    const counts = new Map<
      string,
      {
        wins: number;
        total: number;
      }
    >();
    shapedRows.forEach((row) => {
      if (!row._spreadBucket || !row._result) return;
      if (!["CORRECT", "INCORRECT"].includes(row._result)) return;
      const entry = counts.get(row._spreadBucket) ?? { wins: 0, total: 0 };
      entry.total += 1;
      if (row._result === "CORRECT") entry.wins += 1;
      counts.set(row._spreadBucket, entry);
    });
    let bestRate = 0;
    let bestBuckets: string[] = [];
    counts.forEach((entry, bucket) => {
      if (!entry.total) return;
      const rate = entry.wins / entry.total;
      if (rate > bestRate + 1e-6) {
        bestRate = rate;
        bestBuckets = [bucket];
      } else if (Math.abs(rate - bestRate) <= 1e-6) {
        bestBuckets = [...bestBuckets, bucket];
      }
    });
    return {
      counts,
      bestBuckets,
      bestRate,
    };
  }, [shapedRows]);

  const bandStats = useMemo(() => {
    const active = selectedBands.length ? selectedBands : SPREAD_BUCKETS.map((b) => b.key);
    let wins = 0;
    let total = 0;
    active.forEach((bucket) => {
      const entry = bucketInsights.counts.get(bucket);
      if (!entry) return;
      wins += entry.wins;
      total += entry.total;
    });
    return {
      label: active.length === SPREAD_BUCKETS.length ? "All spreads" : active.join(", "),
      wins,
      total,
      pct: total > 0 ? (wins / total) * 100 : null,
    };
  }, [selectedBands, bucketInsights.counts]);

  const filteredRows = useMemo(() => {
    if (!shapedRows.length) return [];
    const weekSet = new Set(selectedWeeks);
    const activeSources = SOURCE_OPTIONS.filter((opt) => sourceFilters[opt.key]).map((opt) => opt.key);
    const bestBuckets = bucketInsights.bestBuckets;
    const activeBandKeys = selectedBands.length ? new Set(selectedBands) : new Set(SPREAD_BUCKETS.map((b) => b.key));

    return shapedRows
      .filter((row) => {
        if (weekSet.size && !weekSet.has(Number(row.week))) return false;
        if (onlyQualified && !row._qualified) return false;
        if (highConfidenceOnly) {
          if (!(Number(row._confidence ?? 0) >= HIGH_CONFIDENCE_MIN && Math.abs(row._value ?? 0) >= HIGH_CONF_VALUE_MIN)) {
            return false;
          }
        }
        if (largeSpreadOnly) {
          if (
            !(
              Number.isFinite(row._spreadAbs) &&
              (row._spreadAbs as number) >= LARGE_SPREAD_ABS_MIN &&
              Number(row._confidence ?? 0) >= LARGE_SPREAD_CONF_MIN
            )
          ) {
            return false;
          }
        }
        if (!(Number.isFinite(row._edge) && Math.abs(row._edge as number) >= edgeMin)) return false;
        if (!(Number.isFinite(row._value) && Math.abs(row._value as number) >= valueMin)) return false;
        if (confidenceMin > 0 && !(Number.isFinite(row._confidence) && (row._confidence as number) >= confidenceMin)) {
          return false;
        }
        if (startDate && (!row._dateIso || row._dateIso < startDate)) return false;
        if (endDate && (!row._dateIso || row._dateIso > endDate)) return false;
        if (activeSources.length && !activeSources.includes(row._sourceKey || "unknown")) return false;
        if (activeBandKeys.size && row._spreadBucket && !activeBandKeys.has(row._spreadBucket)) return false;
        if (mostLikelyOnly) {
          if (!row._spreadBucket || !bestBuckets.includes(row._spreadBucket)) return false;
        }
        return true;
      })
      .sort((a, b) => {
        const confDiff = (b._confidence ?? 0) - (a._confidence ?? 0);
        if (Math.abs(confDiff) > 1e-6) return confDiff;
        const edgeDiff = Math.abs(b._edge ?? 0) - Math.abs(a._edge ?? 0);
        if (Math.abs(edgeDiff) > 1e-6) return edgeDiff;
        return Math.abs(b._value ?? 0) - Math.abs(a._value ?? 0);
      });
  }, [
    shapedRows,
    selectedWeeks,
    onlyQualified,
    highConfidenceOnly,
    largeSpreadOnly,
    edgeMin,
    valueMin,
    confidenceMin,
    startDate,
    endDate,
    sourceFilters,
    selectedBands,
    mostLikelyOnly,
    bucketInsights.bestBuckets,
  ]);

  const handleWeekToggle = (week: number) => {
    setSelectedWeeks((prev) => {
      if (prev.includes(week)) {
        return prev.filter((w) => w !== week);
      }
      return [...prev, week].sort((a, b) => a - b);
    });
  };

  const handleSourceToggle = (key: string) => {
    setSourceFilters((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleBandToggle = (key: string) => {
    setSelectedBands((prev) => {
      if (prev.includes(key)) {
        return prev.filter((k) => k !== key);
      }
      const added = [...prev, key];
      return SPREAD_BUCKETS.map((b) => b.key).filter((k) => added.includes(k));
    });
  };

  const handleSelectAllBands = () => {
    setSelectedBands(SPREAD_BUCKETS.map((b) => b.key));
  };

  const handleEdgeMin = (value: string) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return;
    setEdgeMin(Math.max(0, num));
  };

  const handleValueMin = (value: string) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return;
    setValueMin(Math.max(0, num));
  };

  const handleConfidenceMin = (value: string) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return;
    setConfidenceMin(Math.min(Math.max(0, num), 1));
  };

  return (
    <section className="page">
      <div className="header">
        <div>
          <h1>Recommended Bets</h1>
          <p className="sub">Confidence-weighted edges pulled from the latest predictions</p>
        </div>
        <div className="view-toggle">
          <button
            type="button"
            className={viewMode === "cards" ? "view-toggle__btn active" : "view-toggle__btn"}
            onClick={() => setViewMode("cards")}
          >
            Card view
          </button>
          <button
            type="button"
            className={viewMode === "list" ? "view-toggle__btn active" : "view-toggle__btn"}
            onClick={() => setViewMode("list")}
          >
            List view
          </button>
        </div>
      </div>

      <div className="controls controls--bets">
        <div className="control-block">
          <div className="control-label">Weeks</div>
          <div className="control-options">
            {weekOptions.map((w) => (
              <label key={w} className="control-chip">
                <input
                  type="checkbox"
                  checked={selectedWeeks.includes(w)}
                  onChange={() => handleWeekToggle(w)}
                />
                <span>W{w}</span>
              </label>
            ))}
            <button type="button" className="control-chip action" onClick={() => setSelectedWeeks([])}>
              All
            </button>
          </div>
        </div>

        <div className="control-block">
          <div className="control-label">Filters</div>
          <div className="control-options">
            <label className="control-chip">
              <input
                type="checkbox"
                checked={onlyQualified}
                onChange={(e) => setOnlyQualified(e.target.checked)}
              />
              <span>Qualified</span>
            </label>
            <label className="control-chip">
              <input
                type="checkbox"
                checked={highConfidenceOnly}
                onChange={(e) => setHighConfidenceOnly(e.target.checked)}
              />
              <span>High confidence</span>
            </label>
            <label className="control-chip">
              <input
                type="checkbox"
                checked={largeSpreadOnly}
                onChange={(e) => setLargeSpreadOnly(e.target.checked)}
              />
              <span>Large spread</span>
            </label>
            <label className="control-chip">
              <input
                type="checkbox"
                checked={mostLikelyOnly}
                onChange={(e) => setMostLikelyOnly(e.target.checked)}
              />
              <span>Most likely to hit</span>
            </label>
          </div>
        </div>

        <div className="control-block">
          <div className="control-label">Thresholds</div>
          <div className="control-options tight">
            <label>
              Edge ≥
              <input
                type="number"
                step="0.1"
                value={edgeMin}
                onChange={(e) => handleEdgeMin(e.target.value)}
              />
            </label>
            <label>
              Value ≥
              <input
                type="number"
                step="0.1"
                value={valueMin}
                onChange={(e) => handleValueMin(e.target.value)}
              />
            </label>
            <label>
              Confidence ≥
              <input
                type="number"
                step="0.01"
                value={confidenceMin}
                onChange={(e) => handleConfidenceMin(e.target.value)}
              />
            </label>
          </div>
        </div>

        <div className="control-block">
          <div className="control-label">Market source</div>
          <div className="control-options">
            {SOURCE_OPTIONS.map(({ key, label }) => (
              <label key={key} className="control-chip">
                <input
                  type="checkbox"
                  checked={sourceFilters[key]}
                  onChange={() => handleSourceToggle(key)}
                />
                <span>{label}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="control-block">
          <div className="control-label">Kickoff window</div>
          <div className="control-options tight">
            <label>
              From
              <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} />
            </label>
            <label>
              To
              <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} />
            </label>
          </div>
        </div>

        <div className="control-block">
          <div className="control-label">Spread band</div>
          <div className="control-options">
            {SPREAD_BUCKETS.map((band) => (
              <label key={band.key} className="control-chip">
                <input
                  type="checkbox"
                  checked={selectedBands.includes(band.key) || selectedBands.length === 0}
                  onChange={() => handleBandToggle(band.key)}
                />
                <span>{band.label}</span>
              </label>
            ))}
            <button type="button" className="control-chip action" onClick={handleSelectAllBands}>
              All
            </button>
          </div>
        </div>

        <Badge tone="muted">Plays: {filteredRows.length}</Badge>
      </div>

      <div className="note">
        Default filters require |edge| ≥ {fmtNum(BETS_EDGE_MIN)} and |value| ≥ {fmtNum(BETS_VALUE_MIN)}. Adjust thresholds or toggles above to widen or narrow the slate.
      </div>
      {bucketInsights.bestBuckets.length > 0 && (
        <div className="note">
          Highest historical hit rate: {bucketInsights.bestBuckets.join(", ")} spreads (~
          {Math.round(bucketInsights.bestRate * 1000) / 10}%). “Most likely to hit” filters to these bucket(s).
        </div>
      )}
      <div className="note">
        Historical win rate for {bandStats.label}:{" "}
        {bandStats.total
          ? `${Math.round((bandStats.pct ?? 0) * 10) / 10}% (${bandStats.wins}/${bandStats.total})`
          : "—"}
      </div>

      {!filteredRows.length && (
        <div className="note">No recommended edges currently match these filters. Loosen thresholds or refresh after markets update.</div>
      )}

      {viewMode === "list" && filteredRows.length > 0 && (
        <div className="table-wrap bets-table">
          <table className="tbl">
            <thead>
              <tr>
                <th>Week</th>
                <th>Kickoff</th>
                <th>Away</th>
                <th>Home</th>
                <th>Market</th>
                <th>Model</th>
                <th>Edge</th>
                <th>Value</th>
                <th>Confidence</th>
                <th>Source</th>
                <th>Qualified</th>
                <th>Pick</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row, idx) => {
                const confPct = Number.isFinite(row._confidence) ? Math.round((row._confidence as number) * 100) : null;
                return (
                  <tr key={`${row.week}-${row.home_team}-${row.away_team}`} className={idx % 2 === 1 ? "alt" : undefined}>
                    <td>{row.week}</td>
                    <td>{row.date || row._dateIso || "TBD"}</td>
                    <td>{row.away_team}</td>
                    <td>{row.home_team}</td>
                    <td>{fmtNum(row._market)}</td>
                    <td>{fmtNum(row._model)}</td>
                    <td className={Number(row._edge ?? 0) >= 0 ? "pos" : "neg"}>{fmtNum(row._edge)}</td>
                    <td className={Number(row._value ?? 0) >= 0 ? "pos" : "neg"}>{fmtNum(row._value)}</td>
                    <td>{confPct !== null ? `${confPct}%` : "—"}</td>
                    <td>{row._source}</td>
                    <td>{row._qualified ? "Yes" : "No"}</td>
                    <td>{row._pick}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {viewMode === "cards" && (
        <div className="pred-grid">
          {filteredRows.map((card) => {
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
                    <div className="pred-card__scoreline">{Number.isFinite(card._delta) ? `Delta: ${fmtNum(card._delta)}` : "Spread delta"}</div>
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
                  <div className="metric"><span className="label">Confidence</span><span>{Number.isFinite(card._confidence) ? `${Math.round((card._confidence as number) * 100)}%` : "—"}</span></div>
                  <div className="metric metric--pick"><span className="label">Suggested Side</span><span>{card._pick}</span></div>
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
      )}
    </section>
  );
}
