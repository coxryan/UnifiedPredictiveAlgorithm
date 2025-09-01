
import StatusPanel from './components/StatusPanel'
import TeamsTable from './components/TeamsTable'
import GamesTable from './components/GamesTable'
import LiveEdge from './components/LiveEdge'
import { useState } from 'react'

export default function App() {
  const [tab, setTab] = useState<'status'|'teams'|'preds'|'edge'>('status')
  const TabBtn = ({
    id, label
  }: {id:'status'|'teams'|'preds'|'edge'; label:string}) => (
    <button onClick={()=>setTab(id)}
      style={{padding:"8px 12px", borderRadius:8, border:"1px solid #e5e7eb",
        background: tab===id ? "#111827" : "#fff",
        color: tab===id ? "#fff" : "#111827"}}>{label}</button>
  )

  return (
    <div className="container">
      <div className="title">UPA-F Dashboard</div>
      <div className="subtitle">Automated all-FBS data collection + live status + teams + predictions</div>

      <div style={{display:"flex", gap:8, marginBottom:16}}>
        <TabBtn id="status" label="Status" />
        <TabBtn id="teams" label="Teams" />
        <TabBtn id="preds" label="Predictions" />
        <TabBtn id="edge" label="Live Edge" />
      </div>

      {tab === 'status' && (
        <div className="grid">
          <div className="card"><StatusPanel /></div>
          <div className="card">
            <h3 style={{marginTop:0}}>Data files</h3>
            <ul>
              <li><a className="link" href="data/upa_team_inputs_datadriven_v0.csv" target="_blank">team inputs CSV ↗</a></li>
              <li><a className="link" href="data/upa_predictions.csv" target="_blank">predictions CSV ↗</a></li>
              <li><a className="link" href="data/live_edge_report.csv" target="_blank">live edge CSV ↗</a></li>
              <li><a className="link" href="data/cfb_schedule.csv" target="_blank">schedule CSV ↗</a></li>
              <li><a className="link" href="data/status.json" target="_blank">status.json ↗</a> <span className="badge">dashboard polls this</span></li>
            </ul>
            <p style={{color:"#4b5563"}}>Pages auto-redeploys when the collector commits new data.</p>
          </div>
        </div>
      )}

      {tab === 'teams' && (
        <div className="card"><TeamsTable /></div>
      )}

      {tab === 'preds' && (
        <div className="card"><GamesTable /></div>
      )}

      {tab === 'edge' && (
        <div className="card"><LiveEdge /></div>
      )}
    </div>
  )
}
