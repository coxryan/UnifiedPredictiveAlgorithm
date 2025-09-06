import { useState } from "react";
import "./styles.css";

import StatusTab from "./tabs/StatusTab";
import PredictionsTab from "./tabs/PredictionsTab";
import TeamTab from "./tabs/TeamTab";
import BetsTab from "./tabs/BetsTab";
import BacktestTab from "./tabs/BacktestTab";
import LiveResultsTab from "./tabs/LiveResultsTab";
import HelpTab from "./tabs/HelpTab";

type Tab = "status" | "team" | "preds" | "live" | "bets" | "backtest" | "help";

export default function App() {
  const [tab, setTab] = useState<Tab>("preds");
  return (
    <div className="page">
      <header className="header">
        <h1>UPA-F Dashboard</h1>
        <p className="sub">2025 live model + 2024 backtest (separate), value, results & recommended bets</p>
      </header>
      <nav className="tabs">
        <button className={tab==="status"?"active":""} onClick={()=>setTab("status")}>Status</button>
        <button className={tab==="team"?"active":""} onClick={()=>setTab("team")}>Team (Schedule)</button>
        <button className={tab==="preds"?"active":""} onClick={()=>setTab("preds")}>Predictions</button>
        <button className={tab==="live"?"active":""} onClick={()=>setTab("live")}>Live Results</button>
        <button className={tab==="bets"?"active":""} onClick={()=>setTab("bets")}>Recommended Bets</button>
        <button className={tab==="backtest"?"active":""} onClick={()=>setTab("backtest")}>Backtest (2024)</button>
        <button className={tab==="help"?"active":""} onClick={()=>setTab("help")}>Help</button>
      </nav>

      {tab==="status" && <StatusTab/>}
      {tab==="team" && <TeamTab/>}
      {tab==="preds" && <PredictionsTab/>}
      {tab==="live" && <LiveResultsTab/>}
      {tab==="bets" && <BetsTab/>}
      {tab==="backtest" && <BacktestTab/>}
      {tab==="help" && <HelpTab/>}

      <footer className="footer">© {new Date().getFullYear()} UPA-F</footer>
    </div>
  );
}