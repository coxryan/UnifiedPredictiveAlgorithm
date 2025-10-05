
import { useEffect, useMemo, useState } from "react"
import { loadTable, toNum } from "../lib/csv"

type GameRow = {
  week?: string
  date?: string
  home_team: string
  away_team: string
  home_conf?: string
  away_conf?: string
  neutral_site?: string
  model_spread_home?: string
  market_spread_home?: string
  edge_points?: string
}

export default function GamesTable() {
  const [rows, setRows] = useState<GameRow[]>([])
  const [week, setWeek] = useState("All")
  const [q, setQ] = useState("")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    (async () => {
      try {
        const obs = (await loadTable("upa_predictions")) as GameRow[]
        setRows(obs)
        setError(null)
      } catch (e: any) {
        setError(e?.message || "Failed to load predictions")
      }
    })()
  }, [])

  const weeks = useMemo(() => {
    const s = new Set<string>()
    rows.forEach(r => r.week && s.add(String(r.week)))
    const arr = Array.from(s).sort((a,b)=>Number(a)-Number(b))
    return ["All", ...arr]
  }, [rows])

  const filtered = useMemo(() => {
    return rows.filter(r => {
      const str = `${r.home_team} ${r.away_team}`.toLowerCase()
      const matchesQ = !q || str.includes(q.toLowerCase())
      const matchesW = week === "All" || String(r.week) === week
      return matchesQ && matchesW
    })
  }, [rows, q, week])

  return (
    <div>
      <div style={{display:"flex", gap:12, marginBottom:12}}>
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search teams…" style={{padding:8,border:"1px solid #e5e7eb",borderRadius:8,width:240}} />
        <select value={week} onChange={e=>setWeek(e.target.value)} style={{padding:8,border:"1px solid #e5e7eb",borderRadius:8}}>
          {weeks.map(w => <option key={w} value={w}>{w}</option>)}
        </select>
      </div>
      {error && <div style={{color:"#dc2626", marginBottom:8}}>{error}</div>}
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%", borderCollapse:"collapse"}}>
          <thead>
            <tr>
              {["Week","Date","Away","Home","Neutral","Model (H)","Market (H)","Edge"].map(h => (
                <th key={h} style={{textAlign:"left",borderBottom:"1px solid #e5e7eb",padding:"8px 6px", background:"#f9fafb"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, idx) => {
              const model = toNum(r.model_spread_home)
              const market = toNum(r.market_spread_home)
              const edge = toNum(r.edge_points)
              const hot = Number.isFinite(edge) && Math.abs(edge) >= 3.0
              return (
                <tr key={idx} style={{background: hot ? "#fffbeb" : undefined}}>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.week || ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.date || ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.away_team}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.home_team}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{String(r.neutral_site||"0")==="1" ? "Y" : ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{Number.isFinite(model) ? model.toFixed(1) : "—"}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{Number.isFinite(market) ? market.toFixed(1) : "—"}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9",fontWeight:hot?600:400}}>{Number.isFinite(edge) ? edge.toFixed(1) : "—"}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
