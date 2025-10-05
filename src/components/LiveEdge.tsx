
import { useEffect, useState } from "react"
import { loadTable, toNum } from "../lib/csv"

type EdgeRow = {
  week?: string
  date?: string
  home_team: string
  away_team: string
  edge_points?: string
}

export default function LiveEdge() {
  const [rows, setRows] = useState<EdgeRow[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    (async () => {
      try {
        const obs = (await loadTable("live_edge_report")) as EdgeRow[]
        obs.sort((a,b)=> Math.abs(toNum(b.edge_points)) - Math.abs(toNum(a.edge_points)))
        setRows(obs.slice(0, 50))
        setError(null)
      } catch (e: any) {
        setError(e?.message || "Failed to load live edge")
      }
    })()
  }, [])

  return (
    <div>
      {error && <div style={{color:"#dc2626", marginBottom:8}}>{error}</div>}
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%", borderCollapse:"collapse"}}>
          <thead>
            <tr>
              {["Week","Date","Away @ Home","Edge"].map(h => (
                <th key={h} style={{textAlign:"left",borderBottom:"1px solid #e5e7eb",padding:"8px 6px", background:"#f9fafb"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const e = toNum(r.edge_points)
              return (
                <tr key={i}>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.week || ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.date || ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.away_team} @ {r.home_team}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9",fontWeight:600}}>{Number.isFinite(e) ? e.toFixed(1) : "—"}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
