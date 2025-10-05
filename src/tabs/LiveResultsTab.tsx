import { useEffect, useMemo, useState } from "react";
import { loadTable, fmtNum, toNum } from "../lib/csv";
import { Badge } from "../lib/ui";

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

export default function LiveResultsTab() {
  const [live, setLive] = useState<LiveRow[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [fbs, setFbs] = useState<Set<string>>(new Set());

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
    const map = new Map<string, { model:number, market:number }>();
    for (const r of preds) {
      const a = (r.away_team||"").toString().trim();
      const h = (r.home_team||"").toString().trim();
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      const src = providerLabel((r as any).market_spread_source);
      if (a && h) map.set(key(a,h), { model, market, source: src });
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
        const pick = Number.isFinite(model) && Number.isFinite(market)
          ? pickFromModelMarket(model, market, hSchool, aSchool).side
          : "—";
        const hp = typeof r.home_points === 'string' ? Number(r.home_points) : (r.home_points as number);
        const ap = typeof r.away_points === 'string' ? Number(r.away_points) : (r.away_points as number);
        const tone = coverStatus(Number(hp), Number(ap), Number(market), pick);
        return { ...r, _model:model, _market:market, _pick:pick, _tone:tone, _marketSource: source } as any;
      });
  }, [live, predByMatch, fbs]);

  // High-contrast tone styles for full-card background (color-blind friendly)
  const toneStyle = (tone: string): React.CSSProperties => {
    switch (tone) {
      case "ok":
        // Bright green with white text
        return { backgroundColor: "#16a34a", color: "#ffffff" };
      case "bad":
        // Bright red with white text
        return { backgroundColor: "#dc2626", color: "#ffffff" };
      default:
        return {};
    }
  };

  return (
    <section className="card">
      <div className="card-title">
        Live Results <Badge tone="muted">Updated {lastUpdatedLabel}</Badge>
      </div>
      {!live.length && (
        <div className="note">No live rows found. Expecting dataset <code>live_scores</code>.</div>
      )}
      <div className="grid">
        {cards.map((g:any, i:number)=> (
          <div
            key={`${g.event_id || g.date || i}`}
            className="live-card"
            style={toneStyle(g._tone)}
            aria-label={g._tone === "ok" ? "On track" : g._tone === "bad" ? "Behind" : "Mixed/Unknown"}
          >
            <div className="row teams">
              <div className="side">
                <div className="lbl">AWAY</div>
                <div className="name">{g.away_school || g.away_team}</div>
                <div className="score">{fmtNum(g.away_points, { maximumFractionDigits: 0 })}</div>
              </div>
              <div className="side">
                <div className="lbl">HOME</div>
                <div className="name">{g.home_school || g.home_team}</div>
                <div className="score">{fmtNum(g.home_points, { maximumFractionDigits: 0 })}</div>
              </div>
            </div>
            <div className="row meta">
              <div>{g.detail || "—"}</div>
              <div>{g.venue || ""}</div>
            </div>
            <div className="row picks">
              <div>Model (H): {fmtNum(g._model)}</div>
              <div>
                Market (H): {fmtNum(g._market)}
                {g._marketSource && g._marketSource !== "—" ? ` (${g._marketSource})` : ""}
              </div>
              <div>Pick: {g._pick || "—"}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
