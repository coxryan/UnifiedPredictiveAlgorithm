import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum } from "../lib/csv";
import { Badge } from "../lib/ui";

type LiveRow = {
  event_id?: string;
  date?: string;
  state?: string;   // "pre" | "in" | "post"
  detail?: string;  // "10:44 - 4th" | "Final"
  clock?: string;
  period?: string | number;
  venue?: string;
  home_team: string; away_team: string;
  home_points?: string; away_points?: string;
};

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string;
  model_spread_book?: string; market_spread_book?: string;
  edge_points_book?: string;
};

function valueSide(modelHome: number, marketHome: number) {
  const edge = modelHome - marketHome;
  if (!Number.isFinite(edge) || Math.abs(edge) < 1e-9) return { pickHome: null, edge };
  // edge>0 ⇒ market too heavy on home ⇒ value = AWAY
  return edge > 0 ? { pickHome: false, edge } : { pickHome: true, edge };
}

function parseClock(detail?: string): { elapsedMin: number | null; remMin: number | null } {
  // "10:44 - 4th"
  if (!detail) return { elapsedMin: null, remMin: null };
  const m = detail.match(/(\d{1,2}):(\d{2})\s*-\s*(\d+)(?:st|nd|rd|th)/i);
  if (!m) return { elapsedMin: null, remMin: null };
  const min = Number(m[1]); const sec = Number(m[2]); const per = Number(m[3]);
  const remInPeriod = min + sec/60;
  const elapsed = (per - 1) * 15 + (15 - remInPeriod);
  const remaining = Math.max(0, 60 - elapsed);
  return { elapsedMin: Math.max(0, Math.min(60, elapsed)), remMin: remaining };
}

export default function LiveResultsTab() {
  const [live, setLive] = useState<LiveRow[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);

  useEffect(() => {
    (async () => {
      try { setLive((await loadCsv("data/live_scores.csv")) as any[] as LiveRow[]); } catch { setLive([]); }
      try { setPreds((await loadCsv("data/upa_predictions.csv")) as any[] as PredRow[]); } catch { setPreds([]); }
    })();
  }, []);

  const predMap = useMemo(() => {
    const m = new Map<string, PredRow>();
    const k = (h: string, a: string) => `${(h||"").trim()}||${(a||"").trim()}`.toLowerCase();
    for (const r of preds) m.set(k(r.home_team, r.away_team), r);
    return { get: (h: string, a: string) => m.get(k(h, a)) };
  }, [preds]);

  const rows = useMemo(() => {
    const out = (live || []).map((g: LiveRow) => {
      const pr = predMap.get(g.home_team, g.away_team);
      const model = toNum(pr?.model_spread_book);
      const market = toNum(pr?.market_spread_book);
      const pick = valueSide(model, market);
      const hp = toNum(g.home_points);
      const ap = toNum(g.away_points);
      const nowCoverHome = Number.isFinite(hp) && Number.isFinite(ap) && Number.isFinite(market)
        ? (hp - ap + market) > 0
        : null;
      const { elapsedMin } = parseClock(g.detail);
      let projectedCover: boolean | null = null;
      if (elapsedMin !== null && elapsedMin >= 10 && Number.isFinite(hp) && Number.isFinite(ap) && Number.isFinite(market)) {
        const paceFinalMargin = (hp - ap) * (60 / Math.max(1, elapsedMin)); // naive pace projection
        projectedCover = (paceFinalMargin + market) > 0;
      } else if (nowCoverHome !== null) {
        projectedCover = nowCoverHome;
      }
      let tone: "ok" | "bad" | "warn" | "muted" = "muted";
      if (g.state === "in" && projectedCover !== null && pick.pickHome !== null) {
        const onTrack = pick.pickHome ? projectedCover : !projectedCover;
        const coverDiffNow = Number.isFinite(hp) && Number.isFinite(ap) && Number.isFinite(market) ? Math.abs((hp - ap + market)) : Infinity;
        if (onTrack && coverDiffNow >= 3) tone = "ok";
        else if (onTrack) tone = "warn";
        else tone = "bad";
      } else if (g.state === "post" && nowCoverHome !== null && pick.pickHome !== null) {
        tone = (pick.pickHome ? nowCoverHome : !nowCoverHome) ? "ok" : "bad";
      }
      return { ...g, _model: model, _market: market, _pickHome: pick.pickHome, _tone: tone };
    });

    const order = { "in": 0, "pre": 1, "post": 2 } as any;
    out.sort((a:any,b:any)=> (order[a.state||"post"] - order[b.state||"post"]) || (a.venue||"").localeCompare(b.venue||""));
    return out;
  }, [live, predMap]);

  const liveCount = rows.filter(r => r.state === "in").length;

  return (
    <section className="card">
      <div className="card-title">Live Results <Badge tone="muted">{liveCount} live</Badge></div>
      {!rows.length && <div className="note">No live feed found. Expecting <code>data/live_scores.csv</code> (updated every 5 minutes).</div>}
      <div className="grid">
        {rows.map((g:any, i:number)=>(
          <div key={`${g.event_id||g.home_team}-${i}`} className={`live-card ${g._tone}`}>
            <div className="row teams">
              <div className="side away">
                <div className="lbl">AWAY</div>
                <div className="name">{g.away_team}</div>
                <div className="score">{fmtNum(g.away_points)}</div>
              </div>
              <div className="side home">
                <div className="lbl">HOME</div>
                <div className="name">{g.home_team}</div>
                <div className="score">{fmtNum(g.home_points)}</div>
              </div>
            </div>
            <div className="row meta">
              <div>{g.detail || "—"}</div>
              <div>{g.venue || ""}</div>
            </div>
            <div className="row picks">
              <div>Model (H): {fmtNum(g._model)}</div>
              <div>Market (H): {fmtNum(g._market)}</div>
              <div>Pick: {g._pickHome===null ? "—" : (g._pickHome ? "HOME" : "AWAY")}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="note" style={{marginTop:8}}>
        Cards turn <span className="pos">green</span> when our pick is projected to cover, <span className="neg">red</span> when off-track, amber when marginal. Projection is a simple pace model (good enough for at-a-glance).
      </div>
    </section>
  );
}