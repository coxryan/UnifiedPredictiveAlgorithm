
import { useEffect, useMemo, useState } from "react"
import { loadTable, toNum } from "../lib/csv"

type TeamRow = {
  team: string
  conference?: string
  wrps_percent_0_100?: string
  talent_score_0_100?: string
  portal_net_0_100?: string
  prev_season_sos_rank_1_133?: string
}

export default function TeamsTable() {
  const [rows, setRows] = useState<TeamRow[]>([])
  const [q, setQ] = useState("")
  const [conf, setConf] = useState("All")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    (async () => {
      try {
        const obs = (await loadTable("upa_team_inputs_datadriven_v0")) as TeamRow[]
        setRows(obs)
        setError(null)
      } catch (e: any) {
        setError(e?.message || "Failed to load team inputs")
      }
    })()
  }, [])

  const conferences = useMemo(() => {
    const s = new Set<string>()
    rows.forEach(r => r.conference && s.add(r.conference))
    return ["All", ...Array.from(s).sort()]
  }, [rows])

  const filtered = useMemo(() => {
    return rows.filter(r => {
      const m1 = !q || (r.team || "").toLowerCase().includes(q.toLowerCase())
      const m2 = conf === "All" || r.conference === conf
      return m1 && m2
    })
  }, [rows, q, conf])

  return (
    <div>
      <div style={{display:"flex", gap:12, marginBottom:12}}>
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Search team…" style={{padding:8,border:"1px solid #e5e7eb",borderRadius:8,width:240}} />
        <select value={conf} onChange={e=>setConf(e.target.value)} style={{padding:8,border:"1px solid #e5e7eb",borderRadius:8}}>
          {conferences.map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
      {error && <div style={{color:"#dc2626", marginBottom:8}}>{error}</div>}
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%", borderCollapse:"collapse"}}>
          <thead>
            <tr>
              {["Team","Conf","WRPS","Talent","Portal","Prev SOS rk"].map(h => (
                <th key={h} style={{textAlign:"left",borderBottom:"1px solid #e5e7eb",padding:"8px 6px", background:"#f9fafb"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(r => {
              const wrps = toNum(r.wrps_percent_0_100)
              const tal = toNum(r.talent_score_0_100)
              const por = toNum(r.portal_net_0_100)
              const sosrk = r.prev_season_sos_rank_1_133 || ""
              return (
                <tr key={r.team}>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.team}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{r.conference || ""}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{Number.isFinite(wrps) ? wrps.toFixed(1) : "—"}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{Number.isFinite(tal) ? tal.toFixed(1) : "—"}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{Number.isFinite(por) ? por.toFixed(1) : "—"}</td>
                  <td style={{padding:"8px 6px",borderBottom:"1px solid #f1f5f9"}}>{sosrk}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
