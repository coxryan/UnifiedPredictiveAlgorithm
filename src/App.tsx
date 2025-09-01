
import StatusPanel from './components/StatusPanel'

export default function App() {
  return (
    <div className="container">
      <div className="title">UPA-F Dashboard</div>
      <div className="subtitle">Automated all-FBS data collection + live status</div>

      <div className="grid">
        <div className="card">
          <StatusPanel />
        </div>
        <div className="card">
          <h3 style={{marginTop:0}}>Data files</h3>
          <ul>
            <li><a className="link" href="data/upa_team_inputs_datadriven_v0.csv" target="_blank">team inputs CSV ↗</a></li>
            <li><a className="link" href="data/status.json" target="_blank">status.json ↗</a> <span className="badge">dashboard polls this</span></li>
          </ul>
          <p style={{color:"#4b5563"}}>Pages auto-redeploys when the collector commits new data.</p>
        </div>
      </div>
    </div>
  )
}
