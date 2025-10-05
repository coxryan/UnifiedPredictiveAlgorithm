import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum, playedBool } from "../lib/csv";
import { Badge, TeamLabel, downloadDataset, nextUpcomingWeek } from "../lib/ui";
import { EDGE_MIN, VALUE_MIN } from "./constants";

// -----------------------------
// Types
// -----------------------------
type PredRow = {
 week: string; date: string; neutral_site?: string;
 home_team: string; away_team: string; played?: any;
 model_spread_book?: string; market_spread_book?: string;
 expected_market_spread_book?: string;
 edge_points_book?: string; value_points_book?: string;
 qualified_edge_flag?: string;
 // optional ranks (if present)
 home_rank?: string; away_rank?: string;
 home_ap_rank?: string; away_ap_rank?: string;
 home_coaches_rank?: string; away_coaches_rank?: string;
 // optional scores (if present for settled games)
 home_points?: string; away_points?: string;
 // additional optional fields for schedule/game info
 game_id?: string;
 kickoff_utc?: string;
 start_date?: string;
};

 type StakeMode = "flat" | "prop" | "kelly";
 type Bucket = "morning" | "afternoon" | "night";

 // Probability of cover from edge (simple bounded mapping)
 function probFromEdge(edgePts: number) {
  const x = Math.min(20, Math.max(0, Math.abs(edgePts)));
  return 0.50 + (x / 20) * 0.25; // 50%..75%
 }

 function kellyFraction(p: number, oddsAmerican: number) {
  let b: number;
  if (oddsAmerican > 0) b = oddsAmerican / 100; else b = 100 / Math.abs(oddsAmerican);
  const q = 1 - p;
  const f = (b * p - q) / b;
  return Math.max(0, Math.min(0.25, f)); // cap 25%
 }

 function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { side: "—", edge };
  return edge > 0 ? { side: `${away} (away)`, edge } : { side: `${home} (home)`, edge };
 }

 // Compute EV (expected ROI per $1) from edge via prob mapping and odds
 function evFromEdge(edgePts: number, oddsAmerican: number) {
  let b = oddsAmerican > 0 ? oddsAmerican / 100 : 100 / Math.abs(oddsAmerican); // net win per $1
  const p = probFromEdge(edgePts);
  return b * p - (1 - p); // expected ROI per $1 staked
 }

// Extract kickoff hour (Pacific Time) from a date string; robustly handles date-only and timezones
function kickoffLabelET(dateStr?: string): string {
  if (!dateStr) return '';
  const s = dateStr.toString().trim();
  // If we only have a date (no time), do not fabricate a time (avoids 8:00 PM ET issue).
  // Example: "2025-09-06" should show blank / TBA rather than 8:00 PM (UTC→ET artifact).
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return '';
  // If an explicit clock is present, prefer that clock as ET even if no timezone is present.
  const clock = s.match(/(\d{1,2}):(\d{2})/);
  // If the string carries an explicit timezone (Z or ±HH:MM), format in ET.
  const hasTZ = /Z|[+-]\d{2}:?\d{2}$/.test(s);
  if (hasTZ) {
    const d = new Date(s);
    if (!isNaN(d.getTime())) {
      try {
        const t = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit' }).format(d);
        return `${t} ET`;
      } catch {}
    }
    // Fall back to clock if parse failed
    return clock ? `${clock[1]}:${clock[2]} ET` : '';
  }
  // If we have a clock but no timezone, just echo the clock (assume ET for display).
  if (clock) return `${clock[1]}:${clock[2]} ET`;
  // Otherwise, try a last-resort parse (may succeed for full ISO without TZ on some platforms).
  const d = new Date(s);
  if (!isNaN(d.getTime())) {
    try {
      const t = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hour: 'numeric', minute: '2-digit' }).format(d);
      return `${t} ET`;
    } catch {}
  }
  return '';
}

function kickoffHourET(dateStr?: string): number | null {
  if (!dateStr) return null;
  const s = dateStr.toString().trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return null; // date-only → unknown
  const clock = s.match(/(\d{1,2}):(\d{2})/);
  const hasTZ = /Z|[+-]\d{2}:?\d{2}$/.test(s);
  if (hasTZ) {
    const d = new Date(s);
    if (!isNaN(d.getTime())) {
      try {
        const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit', hour12: true }).formatToParts(d);
        const hh = parts.find(p=>p.type==='hour')?.value ?? '00';
        return Number(hh);
      } catch {}
    }
    return clock ? Math.max(0, Math.min(23, Number(clock[1]))) : null;
  }
  if (clock) return Math.max(0, Math.min(23, Number(clock[1])));
  const d = new Date(s);
  if (isNaN(d.getTime())) return null;
  try {
    const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/Los_Angeles', hour: '2-digit', minute: '2-digit', hour12: true }).formatToParts(d);
    const hh = parts.find(p=>p.type==='hour')?.value ?? '00';
    return Number(hh);
  } catch { return null; }
}

function bucketOfHour(h: number | null): Bucket {
  if (h === null || !Number.isFinite(h)) return "afternoon"; // default bucket if unknown
  if (h < 13) return "morning";    // 0:00–12:59 → Morning slate
  if (h < 18) return "afternoon";  // 13:00–17:59 → Afternoon slate
  return "night";                  // 18:00+ → Night slate
}

export default function BetsTab() {
  const [rows, setRows] = useState<PredRow[]>([]);
  const [wk, setWk] = useState<number | null>(null);
  const [kickKeyMap, setKickKeyMap] = useState<Record<string, string>>({});
  const [kickIdMap, setKickIdMap] = useState<Record<string, string>>({});
  const keyOf = (w: any, a: any, h: any) => `${Number(w) || 0}|${(a ?? "").toString().trim()}|${(h ?? "").toString().trim()}`;

  // Controls
  const [bankroll, setBankroll] = useState<string>("1000");
  const [mode, setMode] = useState<StakeMode>("prop");
  const [odds, setOdds] = useState<string>("-110");
  const [topN, setTopN] = useState<string>("10");
  const [requireQualified, setRequireQualified] = useState<boolean>(true);
  const [requirePositiveEV, setRequirePositiveEV] = useState<boolean>(true);
  const [rollForward, setRollForward] = useState<boolean>(true);

  useEffect(() => {
   (async () => {
    try {
      const r = (await loadTable("upa_predictions")) as PredRow[];
      setRows(r);
      const nextWk = nextUpcomingWeek(r as any);
      setWk(nextWk);
      try {
        const sched = (await loadTable("cfb_schedule")) as any[];
        const byKey: Record<string, string> = {};
        const byId: Record<string, string> = {};
        const norm = (s: any) => (s ?? "").toString().trim();
        const key = (w: any, a: any, h: any) => `${Number(w) || 0}|${norm(a)}|${norm(h)}`;
        // Helper: combine date and time if needed
        function combineDateTime(dateStr: string, timeStr: string): string {
          if (!dateStr) return '';
          if (!timeStr) return dateStr;
          // If dateStr already has a clock, don't append timeStr
          if (/(\d{1,2}):(\d{2})/.test(dateStr)) return dateStr;
          return `${dateStr} ${timeStr}`;
        }
        for (const r of sched || []) {
          let dateStr = (r.kickoff_utc ?? r.start_date ?? r.datetime ?? r.date ?? '').toString().trim();
          if (!dateStr) continue;
          let timeStr = '';
          if (!/(\d{1,2}):(\d{2})/.test(dateStr)) {
            timeStr =
              (r.kickoff_et ?? r.kickoff_time ?? r.start_time ?? r.time ?? r.kick_time ?? '').toString().trim();
          }
          const dt = timeStr ? combineDateTime(dateStr, timeStr) : dateStr;
          byKey[key(r.week, r.away_team, r.home_team)] = dt;
          if (r.game_id != null && r.game_id !== "") byId[String(r.game_id)] = dt;
        }
        setKickKeyMap(byKey);
        setKickIdMap(byId);
      } catch {}
    } catch {
      setRows([]);
      setWk(null);
    }
   })();
  }, []);

  // available week list for dropdown
  const weekOptions = useMemo(() => {
    const w = Array.from(
      new Set(
        rows.map((r) => Number(r.week)).filter((x) => Number.isFinite(x))
      )
    ).sort((a, b) => a - b);
    return w as number[];
  }, [rows]);

  const nextWk = useMemo(() => nextUpcomingWeek(rows as any), [rows]);

  const withComputed = useMemo(() => {
    return rows.map((r) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const expected = toNum(r.expected_market_spread_book);
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
      const neutral = r.neutral_site === "1" || r.neutral_site === "true";
      const ev = Number.isFinite(edge) ? evFromEdge(edge, Number(odds) || -110) : NaN;
      // Build schedule datetime string as in PredictionsTab
      let schedDt: string | undefined =
        (r as any).kickoff_utc ||
        (r as any).start_date ||
        (r as any).dateTime ||
        (r.game_id ? kickIdMap[String(r.game_id)] : undefined) ||
        kickKeyMap[keyOf(r.week, r.away_team, r.home_team)];
      // If not present, fallback to r.date
      if (!schedDt) schedDt = r.date;
      const kickLabel = kickoffLabelET(schedDt);
      const hour = kickoffHourET(schedDt);
      const bucket = bucketOfHour(hour);
      return {
        ...r,
        _model: model,
        _market: market,
        _edge: edge,
        _value: value,
        _pick: pick.side,
        _neutral: neutral,
        _ev: ev,
        _hour: hour,
        _bucket: bucket,
        _kick: kickLabel,
      } as any;
    });
  }, [rows, odds, kickKeyMap, kickIdMap]);

  // Base candidate set for selected week
  const candidatesRaw = useMemo(() => {
    if (!wk) return [] as any[];
    const isFutureWeek = nextWk && wk >= nextWk; // for current/upcoming, hide already played
    return withComputed.filter((r: any) => {
      const sameWeek = Number(r.week) === wk;
      if (!sameWeek) return false;
      if (!isFutureWeek) return true; // past weeks: include played
      return !playedBool(r.played);
    });
  }, [withComputed, wk, nextWk]);

  // Strong-filtered list with outlier removal + thresholds + optional positive EV
  const strongFiltered = useMemo(() => {
    if (!candidatesRaw.length) return [] as any[];

    const edgeAbs = candidatesRaw
      .map((r: any) => Math.abs(r._edge))
      .filter((v: number) => Number.isFinite(v)) as number[];

    const mean = edgeAbs.length ? edgeAbs.reduce((a, b) => a + b, 0) / edgeAbs.length : 0;
    const variance = edgeAbs.length ? edgeAbs.reduce((a, b) => a + (b - mean) ** 2, 0) / edgeAbs.length : 0;
    const sd = Math.sqrt(variance);

    const isOutlier = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      if (!Number.isFinite(e) || !Number.isFinite(v)) return true;
      if (v > 40) return true; // unrealistically large value → likely data issue
      if (sd > 0 && e > mean + 2.5 * sd) return true; // z-score outlier
      return false;
    };

    const meetsThresholds = (r: any) => {
      const e = Math.abs(r._edge);
      const v = Math.abs(r._value);
      const q = r.qualified_edge_flag === "1";
      const evOk = !requirePositiveEV || (Number.isFinite(r._ev) && r._ev > 0);
      if (requireQualified && !q) return false;
      if (!Number.isFinite(e) || !Number.isFinite(v)) return false;
      if (e < EDGE_MIN) return false;
      if (v < VALUE_MIN) return false;
      if (!evOk) return false;
      return true;
    };

    const kept = candidatesRaw.filter((r: any) => meetsThresholds(r) && !isOutlier(r));

    kept.sort((a: any, b: any) => {
      const qa = a.qualified_edge_flag === "1" ? 1 : 0;
      const qb = b.qualified_edge_flag === "1" ? 1 : 0;
      if (qa !== qb) return qb - qa;
      const ea = Math.abs(a._edge);
      const eb = Math.abs(b._edge);
      if (ea !== eb) return eb - ea;
      const va = Math.abs(a._value);
      const vb = Math.abs(b._value);
      if (va !== vb) return vb - va;
      // earlier kickoff first (so morning, then afternoon, then night naturally if hours present)
      const ha = (a._hour ?? 24);
      const hb = (b._hour ?? 24);
      if (ha !== hb) return ha - hb;
      return (a.date || "").localeCompare(b.date || "");
    });

    const limit = Math.max(1, Number(topN) || 10);
    return kept.slice(0, limit);
  }, [candidatesRaw, requireQualified, requirePositiveEV, topN]);

  // Split into buckets
  const byBucket = useMemo(() => {
    const m: any[] = [], a: any[] = [], n: any[] = [];
    for (const r of strongFiltered) {
      if (r._bucket === "morning") m.push(r);
      else if (r._bucket === "afternoon") a.push(r);
      else n.push(r);
    }
    return { morning: m, afternoon: a, night: n } as Record<Bucket, any[]>;
  }, [strongFiltered]);

  // Helper stake allocator for a group
  function stakeGroup(list: any[], bankroll: number, oddsAm: number, mode: StakeMode) {
    if (!list.length || bankroll <= 0) return list.map((r) => ({ ...r, _stake: 0, _result: resultForRow(r) }));
    const edges = list.map((r: any) => Math.abs(r._edge));
    const maxEdge = Math.max(...edges.map((e: number) => (Number.isFinite(e) ? e : 0)));
    const weights = list.map((r: any, i: number) => {
      const e = edges[i];
      if (!Number.isFinite(e) || e <= 0) return 0;
      if (mode === "flat") return 1;
      if (mode === "prop") return e / (maxEdge || 1);
      const p = probFromEdge(r._edge);
      return kellyFraction(p, oddsAm);
    });
    const sumW = weights.reduce((a, b) => a + b, 0) || 1;
    return list.map((r: any, i: number) => {
      const stake = bankroll * (weights[i] / sumW);
      return { ...r, _stake: Math.round(stake * 100) / 100, _result: resultForRow(r) };
    });
  }

  // Helper to compute W/L/P for a row when scores are present
  function resultForRow(r: any): "W" | "L" | "P" | "" {
    const hp = toNum(r.home_points);
    const ap = toNum(r.away_points);
    if (!Number.isFinite(hp) || !Number.isFinite(ap) || !Number.isFinite(r._market)) return "";
    // book-style: negative = home favorite. Home covers if (home - away + market) > 0
    const adj = (hp - ap) + r._market;
    const coverHome = adj > 0 ? 1 : adj < 0 ? -1 : 0; // 1 cover, -1 no cover, 0 push
    const pickHome = typeof r._pick === "string" && r._pick.includes("(home)");
    const pickAway = typeof r._pick === "string" && r._pick.includes("(away)");
    if (!pickHome && !pickAway) return "";
    if (coverHome === 0) return "P";
    return pickHome ? (coverHome > 0 ? "W" : "L") : (coverHome < 0 ? "W" : "L");
  }

  // Stakes per bucket (with optional roll-forward)
  const stakedBuckets = useMemo(() => {
    const bk = Math.max(0, Number(bankroll) || 0);
    const oddsAm = Number(odds) || -110;

    if (!rollForward) {
      // allocate on the union, then split by bucket for display
      const union = [...byBucket.morning, ...byBucket.afternoon, ...byBucket.night];
      const unionStaked = stakeGroup(union, bk, oddsAm, mode);
      const pick = (arr: any[]) => arr.map((r) => unionStaked.find((u) => u === r) || r);
      return {
        morning: pick(byBucket.morning),
        afternoon: pick(byBucket.afternoon),
        night: pick(byBucket.night),
      } as Record<Bucket, any[]>;
    }

    // Roll-forward: spend morning first; leftover moves to afternoon; leftover to night
    let remaining = bk;
    const m = stakeGroup(byBucket.morning, remaining, oddsAm, mode);
    const spentM = m.reduce((s, r: any) => s + (r._stake || 0), 0);
    remaining = Math.max(0, remaining - spentM);

    const a = stakeGroup(byBucket.afternoon, remaining, oddsAm, mode);
    const spentA = a.reduce((s, r: any) => s + (r._stake || 0), 0);
    remaining = Math.max(0, remaining - spentA);

    const n = stakeGroup(byBucket.night, remaining, oddsAm, mode);

    return { morning: m, afternoon: a, night: n } as Record<Bucket, any[]>;
  }, [byBucket, bankroll, mode, odds, rollForward]);

  const totals = useMemo(() => {
    const sum = (arr: any[]) => arr.reduce((s, r: any) => s + (r._stake || 0), 0);
    return {
      morning: sum(stakedBuckets.morning),
      afternoon: sum(stakedBuckets.afternoon),
      night: sum(stakedBuckets.night),
      all: sum(stakedBuckets.morning) + sum(stakedBuckets.afternoon) + sum(stakedBuckets.night),
    };
  }, [stakedBuckets]);

  function Section({ title, data }: { title: string; data: any[] }) {
    return (
      <section className="subcard">
        <div className="subcard-title">{title} — {data.length} plays <Badge tone="muted">Stake: ${fmtNum(data.reduce((s, r:any)=> s + (r._stake||0), 0))}</Badge></div>
        {!data.length ? (
          <div className="note">No strong, data-quality-safe plays in this slate.</div>
        ) : (
          <div className="table-wrap">
            <table className="tbl wide">
              <thead>
                <tr>
                  <th>Wk</th>
                  <th>Date</th>
                  <th>Kick (ET)</th>
                  <th colSpan={2}>Matchup</th>
                  <th>Model (H)</th>
                  <th>Market (H)</th>
                  <th>Edge</th>
                  <th>Value</th>
                  <th>EV</th>
                  <th>Qualified</th>
                  <th>Recommended Side</th>
                  <th>Stake</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r: any, i: number) => (
                  <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i % 2 ? "alt" : undefined}>
                    <td>{r.week}</td>
                    <td>{r.date}</td>
                    <td>{r._kick}</td>
                    <td style={{ textAlign: "right" }}>
                      <TeamLabel home={false} team={r.away_team} neutral={false} />
                    </td>
                    <td style={{ textAlign: "left" }}>
                      <TeamLabel home={true} team={r.home_team} neutral={r._neutral} />
                    </td>
                    <td>{fmtNum(r._model)}</td>
                    <td>{fmtNum(r._market)}</td>
                    <td className={Number.isFinite(r._edge) ? (r._edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._edge)}</td>
                    <td className={Number.isFinite(r._value) ? (r._value > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._value)}</td>
                    <td className={Number.isFinite(r._ev) ? (r._ev > 0 ? "pos" : "neg") : undefined}>{fmtNum((r._ev as number) * 100, { maximumFractionDigits: 1 })}%</td>
                    <td>{r.qualified_edge_flag === "1" ? "✓" : "—"}</td>
                    <td>{r._pick}</td>
                    <td>${fmtNum(r._stake, { maximumFractionDigits: 0 })}</td>
                    <td>{r._result || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    );
  }

  const unionStaked = useMemo(() => [
    ...stakedBuckets.morning,
    ...stakedBuckets.afternoon,
    ...stakedBuckets.night,
  ], [stakedBuckets]);

  return (
    <section className="card">
      <div className="card-title">Recommended Bets (Strong Only) — 2025</div>
      <div className="controls" style={{ flexWrap: "wrap", gap: 8 }}>
        {/* Week selection */}
        <label>Week
          <select
            className="input"
            value={wk ?? ""}
            onChange={(e) => setWk(e.target.value ? Number(e.target.value) : null)}
          >
            {(weekOptions.length ? weekOptions : [wk ?? 1]).map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </label>
        {wk && nextWk && wk === nextWk ? (
          <Badge tone="muted">Upcoming</Badge>
        ) : wk ? (
          <Badge tone="muted">Historical</Badge>
        ) : null}

        <label>Bankroll ($)
          <input className="input" type="number" min="0" step="1" value={bankroll} onChange={(e) => setBankroll(e.target.value)} />
        </label>

        <label>Stake Mode
          <select className="input" value={mode} onChange={(e) => setMode(e.target.value as StakeMode)}>
            <option value="flat">Flat (equal)</option>
            <option value="prop">Proportional (by edge)</option>
            <option value="kelly">Kelly-Lite (cap 25%)</option>
          </select>
        </label>

        <label>Odds (American)
          <input className="input" type="number" step="5" value={odds} onChange={(e) => setOdds(e.target.value)} />
        </label>

        <label>Show Top N
          <input className="input" type="number" min="1" step="1" value={topN} onChange={(e) => setTopN(e.target.value)} />
        </label>

        <label className="chk">
          <input type="checkbox" checked={requireQualified} onChange={(e) => setRequireQualified(e.target.checked)} />
          Require Qualified ✓ (|Edge| ≥ {EDGE_MIN}, |Value| ≥ {VALUE_MIN}, side agreement)
        </label>

        <label className="chk">
          <input type="checkbox" checked={requirePositiveEV} onChange={(e) => setRequirePositiveEV(e.target.checked)} />
          Require positive EV (based on edge & odds)
        </label>

        <label className="chk">
          <input type="checkbox" checked={rollForward} onChange={(e)=> setRollForward(e.target.checked)} />
          Roll bankroll forward (Morning → Afternoon → Night)
        </label>

        <Badge tone="muted">Total Stake: ${fmtNum(totals.all)}</Badge>
        <Badge tone="muted">Morning: ${fmtNum(totals.morning)}</Badge>
        <Badge tone="muted">Afternoon: ${fmtNum(totals.afternoon)}</Badge>
        <Badge tone="muted">Night: ${fmtNum(totals.night)}</Badge>

        <button
          className="btn"
          disabled={!unionStaked.length}
          onClick={() =>
            downloadDataset(
              `week_${wk ?? "NA"}_bets.data`,
              unionStaked.map((r: any) => ({
                week: r.week,
                date: r.date,
                slate: r._bucket,
                matchup: `${r.away_team} @ ${r.home_team}`,
                recommended_side: r._pick,
                model_line_home: r._model,
                market_line_home: r._market,
                edge_points: r._edge,
                value_points: r._value,
                ev_roi: r._ev,
                qualified: r.qualified_edge_flag === "1" ? "YES" : "NO",
                stake: r._stake,
                result: r._result || "",
                odds: Number(odds) || -110,
              }))
            )
          }
        >
          Download betslip
        </button>
      </div>

      {/* Sections */}
      <Section title="Morning Slate" data={stakedBuckets.morning} />
      <Section title="Afternoon Slate" data={stakedBuckets.afternoon} />
      <Section title="Night Slate" data={stakedBuckets.night} />

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads shown (negative = home favorite). <b>Value side</b> is computed from
        edge = model − market (home perspective). Positive EV is calculated from a bounded probability
        mapping of edge to cover-probability and the American odds. Games without explicit kickoff
        time default to the Afternoon slate for display.
      </div>
    </section>
  );
}
