import { useEffect, useMemo, useState } from "react";
import { loadCsv, fmtNum, toNum } from "../lib/csv";

// Join ESPN live scores with our predictions by (away_school, home_school)
// and filter to FBS-vs-FBS using the current season team inputs file.

type LiveRow = {
  date?: string; state?: string; detail?: string; clock?: string; period?: string;
  venue?: string; event_id?: string;
  away_team?: string; home_team?: string;           // ESPN display (e.g., "Pittsburgh Panthers")
  away_school?: string; home_school?: string;       // normalized school (e.g., "Pittsburgh")
  away_points?: string | number; home_points?: string | number;
};

type PredRow = {
  week: string; date: string; neutral_site?: string;
  home_team: string; away_team: string; played?: any;
  model_spread_book?: string; market_spread_book?: string;
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
      try { setLive((await loadCsv("data/live_scores.csv")) as LiveRow[]); } catch { setLive([]); }
      try { setPreds((await loadCsv("data/upa_predictions.csv")) as PredRow[]); } catch { setPreds([]); }
      try {
        const t = (await loadCsv("data/upa_team_inputs_datadriven_v0.csv")) as TeamInput[];
        setFbs(new Set((t || []).map((x:any)=> (x.team||"").toString().trim())));
      } catch { setFbs(new Set()); }
    })();
  }, []);

  const predByMatch = useMemo(() => {
    const key = (a:string,h:string)=> `${a}|${h}`;
    const map = new Map<string, { model:number, market:number }>();
    for (const r of preds) {
      const a = (r.away_team||"").toString().trim();
      const h = (r.home_team||"").toString().trim();
      const model = toNum(r.model_spread_book);
      const market = toNum(r.market_spread_book);
      if (a && h) map.set(key(a,h), { model, market });
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
        const pick = Number.isFinite(model) && Number.isFinite(market)
          ? pickFromModelMarket(model, market, hSchool, aSchool).side
          : "—";
        const hp = typeof r.home_points === 'string' ? Number(r.home_points) : (r.home_points as number);
        const ap = typeof r.away_points === 'string' ? Number(r.away_points) : (r.away_points as number);
        const tone = coverStatus(Number(hp), Number(ap), Number(market), pick);
        return { ...r, _model:model, _market:market, _pick:pick, _tone:tone } as any;
      });
  }, [live, predByMatch, fbs]);

  return (
    <section className="card">
      <div className="card-title">Live Results (auto-refreshed via GitHub Actions)</div>
      {!live.length && (
        <div className="note">No live rows found. Expecting <code>data/live_scores.csv</code>.</div>
      )}
      <div className="grid">
        {cards.map((g:any, i:number)=> (
          <div key={`${g.event_id||g.date||i}`} className={`live-card ${g._tone}`}>
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
              <div>Market (H): {fmtNum(g._market)}</div>
              <div>Pick: {g._pick || "—"}</div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}