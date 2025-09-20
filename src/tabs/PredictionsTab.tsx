import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum } from "../lib/csv";
import { Badge, TeamLabel, nextUpcomingWeek } from "../lib/ui";

// Numeric normalization helpers
const isNum = (v: any) => Number.isFinite(toNum(v));
const num = (v: any) => toNum(v);

type LiveRow = {
  state?: string;
  away_school?: string; home_school?: string;
  away_points?: string | number; home_points?: string | number;
};

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
};

function valueSide(modelHome: number, marketHome: number, home: string, away: string) {
  const edge = modelHome - marketHome; // >0 => market too heavy on HOME => value = AWAY
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

  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' }>({ key: 'week', dir: 'asc' });

  useEffect(() => {
    (async () => {
      try {
        const r = (await loadCsv("data/upa_predictions.csv")) as PredRow[];
        const norm = (x: PredRow): PredRow => ({
          ...x,
          // Coerce numbers so 0 stays 0 and strings/NaN don't get treated as falsy
          model_spread_book: String(num(x.model_spread_book)),
          market_spread_book: String(num(x.market_spread_book)),
          expected_market_spread_book: String(num(x.expected_market_spread_book)),
          edge_points_book: String(num(x.edge_points_book)),
          value_points_book: String(num(x.value_points_book)),
        });
        const normalized = r.map(norm);
        setRows(normalized);

        // Revert: default to upcoming week (original behavior). If unavailable, fall back to earliest week present.
        const nextWk = nextUpcomingWeek(normalized as any);
        if (nextWk) {
          setWk(nextWk);
        } else {
          const w = Array.from(new Set(normalized.map(x => Number(x.week)).filter(x => Number.isFinite(x)))).sort((a,b)=>a-b);
          setWk(w.length ? w[0] : null);
        }
      } catch {
        setRows([]);
        setWk(null);
      }

      try {
        const l = (await loadCsv("data/live_scores.csv")) as LiveRow[];
        setLiveRows(l || []);
      } catch {
        setLiveRows([]);
      }

      try {
        const sched = (await loadCsv("data/cfb_schedule.csv")) as any[];
        const byKey: Record<string, string> = {};
        const byId: Record<string, string> = {};
        const norm = (s: any) => (s ?? '').toString().trim();
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
          // Preferred order for date and time fields
          let dateStr = (r.kickoff_utc ?? r.start_date ?? r.datetime ?? r.date ?? '').toString().trim();
          if (!dateStr) continue;
          // If dateStr does not contain a clock, look for a time-only field
          let timeStr = '';
          if (!/(\d{1,2}):(\d{2})/.test(dateStr)) {
            timeStr =
              (r.kickoff_et ?? r.kickoff_time ?? r.start_time ?? r.time ?? r.kick_time ?? '').toString().trim();
          }
          const dt = timeStr ? combineDateTime(dateStr, timeStr) : dateStr;
          byKey[key(r.week, r.away_team, r.home_team)] = dt;
          if (r.game_id != null && r.game_id !== '') byId[String(r.game_id)] = dt;
        }
        setKickKeyMap(byKey);
        setKickIdMap(byId);
      } catch {
        setKickKeyMap({});
        setKickIdMap({});
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

  // map of live scores keyed by "away|home" using normalized school names from live CSV
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

  const keyOf = (w:any,a:any,h:any)=> `${Number(w)||0}|${String(a||'').trim()}|${String(h||'').trim()}`;

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
          const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true }).formatToParts(d);
          const hh = parts.find(p=>p.type==='hour')?.value ?? '00';
          const mm = parts.find(p=>p.type==='minute')?.value ?? '00';
          const day = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', month: '2-digit', day: '2-digit' }).format(d);
          return Number(hh);
        } catch {}
      }
      return clock ? Math.max(0, Math.min(23, Number(clock[1]))) : null;
    }
    if (clock) return Math.max(0, Math.min(23, Number(clock[1])));
    const d = new Date(s);
    if (isNaN(d.getTime())) return null;
    try {
      const parts = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: true }).formatToParts(d);
      const hh = parts.find(p=>p.type==='hour')?.value ?? '00';
      return Number(hh);
    } catch { return null; }
  }

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
          const t = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).format(d);
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
        const t = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit' }).format(d);
        return `${t} ET`;
      } catch {}
    }
    return '';
  }

  const toneStyleRow = (tone: 'win'|'loss'|'none') => (
    tone === 'win' ? { backgroundColor: '#16a34a22' } : tone === 'loss' ? { backgroundColor: '#dc262622' } : undefined
  );

  const sortIndicator = (k:string)=> sort.key===k ? (sort.dir==='asc'?' ▲':' ▼') : '';

  const tableRows = useMemo(() => {
    const filtered = rows.filter((r) => (wk ? Number(r.week) === wk : true));
    const f2 = onlyQualified ? filtered.filter((r:any)=> r.qualified_edge_flag === '1') : filtered;
    return f2.map((r:any) => {
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const expected = toNum(r.expected_market_spread_book);
      const edge = Number.isFinite(toNum(r.edge_points_book)) ? toNum(r.edge_points_book)
        : (Number.isFinite(model) && Number.isFinite(market) ? model - market : NaN);
      const value = Number.isFinite(toNum(r.value_points_book)) ? toNum(r.value_points_book)
        : (Number.isFinite(market) && Number.isFinite(expected) ? market - expected : NaN);

      const pick = valueSide(model, market, r.home_team, r.away_team);

      // kickoff resolution
      const schedDt = (r as any).kickoff_utc || (r as any).start_date || kickIdMap[String((r as any).game_id || '')] || kickKeyMap[keyOf(r.week, r.away_team, r.home_team)] || r.date;
      const kickLabel = kickoffLabelET(schedDt);

      // live override: use live scores whenever available (in or post)
      const key = `${(r.away_team||'').toString().trim()}|${(r.home_team||'').toString().trim()}`;
      const live = liveMap.get(key);
      let hp = toNum(r.home_points);
      let ap = toNum(r.away_points);
      if (live && (live.state === 'in' || live.state === 'post')) {
        if (Number.isFinite(live.hp)) hp = live.hp as number;
        if (Number.isFinite(live.ap)) ap = live.ap as number;
      }

      const score = (Number.isFinite(hp) && Number.isFinite(ap)) ? `${fmtNum(ap,{maximumFractionDigits:0})} @ ${fmtNum(hp,{maximumFractionDigits:0})}` : '—';
      const finalDiff = (Number.isFinite(hp) && Number.isFinite(ap)) ? (hp - ap) : NaN; // home minus away

      // correctness tone if game is final (post)
      let tone: 'win'|'loss'|'none' = 'none';
      if (live && live.state === 'post' && Number.isFinite(finalDiff) && Number.isFinite(market)) {
        const homeCover = (finalDiff + market) > 0; // book-style cover test
        const pickHome = (pick.side || '').includes('(home)');
        const pickAway = (pick.side || '').includes('(away)');
        const correct = pickHome ? homeCover : pickAway ? !homeCover : null;
        if (correct === true) tone = 'win';
        else if (correct === false) tone = 'loss';
      }

      return { ...r,
        _kick: kickLabel,
        _model: model, _market: market, _expected: expected,
        _edge: edge, _value: value, _pick: pick.side,
        _score: score, _finalDiff: finalDiff,
        _tone: tone,
      };
    });
  }, [rows, wk, onlyQualified, liveMap, kickKeyMap, kickIdMap]);

  const sortedRows = useMemo(() => {
    const arr = [...tableRows];
    const dir = sort.dir === 'asc' ? 1 : -1;
    const get = (r:any) => {
      switch (sort.key) {
        case 'week': return Number(r.week) || 0;
        case 'date': return new Date(r.date || '').getTime() || 0;
        case 'kick': return r._kick || '';
        case 'away': return (r.away_team||'').toString();
        case 'home': return (r.home_team||'').toString();
        case 'score': return Number.isFinite(r._finalDiff) ? r._finalDiff : -1e9;
        case 'model': return Number(r._model) || 0;
        case 'market': return Number(r._market) || 0;
        case 'expected': return Number(r._expected) || 0;
        case 'edge': return Number(r._edge) || 0;
        case 'value': return Number(r._value) || 0;
        case 'qual': return r.qualified_edge_flag === '1' ? 1 : 0;
        default: return 0;
      }
    };
    arr.sort((a,b)=> (get(a) > get(b) ? 1 : get(a) < get(b) ? -1 : 0) * dir);
    return arr;
  }, [tableRows, sort]);

  const requestSort = (key:string) => {
    setSort((s)=> s.key === key ? { key, dir: s.dir === 'asc' ? 'desc' : 'asc' } : { key, dir: 'asc' });
  };

  return (
    <section className="card">
      <div className="card-title">Predictions — 2025</div>
      <div className="controls">
        <label>Week
          <select className="input" value={wk ?? ""} onChange={(e)=> setWk(e.target.value ? Number(e.target.value) : null)}>
            {(weekOptions.length ? weekOptions : [wk ?? 1]).map((w) => (
              <option key={w} value={w}>{w}</option>
            ))}
          </select>
        </label>
        <label className="chk">
          <input type="checkbox" checked={onlyQualified} onChange={(e)=> setOnlyQualified(e.target.checked)} />
          Exclude non-qualified (show only ✓)
        </label>
        <Badge tone="muted">Rows: {tableRows.length}</Badge>
      </div>

      {!rows.length && (
        <div className="note" style={{marginBottom:8}}>No predictions found. Expecting <code>data/upa_predictions.csv</code>.</div>
      )}

      <div className="table-wrap">
        <table className="tbl wide">
          <thead>
            <tr>
              <th onClick={()=>requestSort('week')}>Week{sortIndicator('week')}</th>
              <th onClick={()=>requestSort('date')}>Date{sortIndicator('date')}</th>
              <th onClick={()=>requestSort('kick')}>Kick (ET){sortIndicator('kick')}</th>
              <th colSpan={2}>Matchup</th>
              <th onClick={()=>requestSort('score')}>Score (A @ H) / Final (H){sortIndicator('score')}</th>
              <th onClick={()=>requestSort('model')}>Model (H){sortIndicator('model')}</th>
              <th onClick={()=>requestSort('market')}>Market (H){sortIndicator('market')}</th>
              <th onClick={()=>requestSort('expected')}>Expected (H){sortIndicator('expected')}</th>
              <th onClick={()=>requestSort('edge')}>Edge{sortIndicator('edge')}</th>
              <th onClick={()=>requestSort('value')}>Value{sortIndicator('value')}</th>
              <th onClick={()=>requestSort('qual')}>Qualified{sortIndicator('qual')}</th>
            </tr>
          </thead>
          <tbody>
            {sortedRows.map((r:any, i:number)=> (
              <tr key={`${r.week}-${r.date}-${r.home_team}-${r.away_team}-${i}`} className={i%2?"alt":undefined} style={toneStyleRow(r._tone)}>
                <td>{r.week}</td>
                <td>{r.date}</td>
                <td>{r._kick}</td>
                <td style={{ textAlign: "right" }}>
                  <TeamLabel home={false} team={r.away_team} neutral={false} />
                  {Number.isFinite(Number(r.away_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.away_rank)}</span>
                  ) : null}
                </td>
                <td style={{ textAlign: "left" }}>
                  <TeamLabel home={true} team={r.home_team} neutral={r.neutral_site === "1" || r.neutral_site === "true"} />
                  {Number.isFinite(Number(r.home_rank)) ? (
                    <span style={{ marginLeft: 6, opacity: 0.7, fontSize: "0.85em" }}>#{Number(r.home_rank)}</span>
                  ) : null}
                </td>
                <td>
                  <div>{r._score}</div>
                  <div style={{opacity:0.75}}>ΔH: {fmtNum(r._finalDiff)}</div>
                </td>
                <td>{fmtNum(r._model)}</td>
                <td>{fmtNum(r._market)}</td>
                <td>{fmtNum(r._expected)}</td>
                <td className={Number.isFinite(r._edge) ? (r._edge > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._edge)}</td>
                <td className={Number.isFinite(r._value) ? (r._value > 0 ? "pos" : "neg") : undefined}>{fmtNum(r._value)}</td>
                <td>{r.qualified_edge_flag === "1" ? "✓" : "—"}</td>
              </tr>
            ))}
            {!sortedRows.length && (
              <tr><td colSpan={12} style={{textAlign:"center", padding:12}}>No rows to display.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="note" style={{ marginTop: 8 }}>
        Book-style spreads shown (negative = home favorite). <b>Edge</b> = model − market (home perspective). <b>Value</b> = market − expected.
        Games marked with a score are complete or live; upcoming games show “—”.
      </div>
    </section>
  );
}