
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type StatusPayload = {
  generated_at_utc?: string;
  next_run_eta_utc?: string;
  teams?: number;
  fields?: string[];
};

function fmt(dt?: string) {
  if (!dt) return "—";
  const d = new Date(dt);
  return d.toLocaleString();
}
function diffHHMMSS(ms: number) {
  if (ms < 0) return "00:00:00";
  const s = Math.floor(ms / 1000);
  const hh = Math.floor(s / 3600).toString().padStart(2, "0");
  const mm = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
  const ss = Math.floor(s % 60).toString().padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}
function minutesSince(iso?: string) {
  if (!iso) return Infinity;
  const ms = Date.now() - new Date(iso).getTime();
  return ms / 60000;
}
function HealthLight({ minutes }: { minutes: number }) {
  let label = "unknown", color = "#d1d5db";
  if (isFinite(minutes)) {
    if (minutes <= 35) { label = "healthy"; color = "#22c55e"; }
    else if (minutes <= 90) { label = "stale"; color = "#facc15"; }
    else { label = "late"; color = "#ef4444"; }
  }
  return (
    <div style={{display:"flex",alignItems:"center",gap:8}}>
      <span style={{display:"inline-block",width:10,height:10,borderRadius:999,background:color}} />
      <span style={{fontSize:12,color:"#4b5563"}}>{label}</span>
    </div>
  );
}

export default function StatusPanel() {
  const [status, setStatus] = useState<StatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [csvLastMod, setCsvLastMod] = useState<string | null>(null);
  const [now, setNow] = useState<number>(Date.now());
  const timerRef = useRef<number | null>(null);
  const pollRef  = useRef<number | null>(null);

  const STATUS_URL = "data/status.json";
  const CSV_URL = "data/upa_team_inputs_datadriven_v0.csv";
  const WORKFLOW_URL = "https://github.com/coxryan/UnifiedPredictiveAlgorithm/actions/workflows/collect-all.yml";

  useEffect(() => {
    timerRef.current = window.setInterval(() => setNow(Date.now()), 1000);
    return () => { if (timerRef.current) window.clearInterval(timerRef.current); };
  }, []);

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch(STATUS_URL, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const json = (await r.json()) as StatusPayload;
      setStatus(json);
      setError(null);
    } catch (e: any) {
      setError(`Status unavailable (${e?.message || "fetch error"})`);
    }
  }, []);

  const fetchCsvLastModified = useCallback(async () => {
    try {
      const head = await fetch(CSV_URL, { method: "HEAD", cache: "no-store" });
      let lm = head.headers.get("Last-Modified");
      if (!lm) {
        const res = await fetch(CSV_URL, { cache: "no-store" });
        lm = res.headers.get("Last-Modified");
      }
      if (lm) setCsvLastMod(new Date(lm).toLocaleString());
    } catch (_) {}
  }, []);

  useEffect(() => {
    fetchStatus();
    pollRef.current = window.setInterval(fetchStatus, 15000);
    const csvTimer = window.setInterval(fetchCsvLastModified, 60000);
    fetchCsvLastModified();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
      window.clearInterval(csvTimer);
    };
  }, [fetchStatus, fetchCsvLastModified]);

  const etaMs = useMemo(() => {
    if (!status?.next_run_eta_utc) return null;
    return new Date(status.next_run_eta_utc).getTime() - now;
  }, [status?.next_run_eta_utc, now]);

  const mins = minutesSince(status?.generated_at_utc);

  return (
    <div>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:12}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <h3 style={{margin:0}}>Collector Status</h3>
          <HealthLight minutes={mins} />
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span className="badge">polls every 15s</span>
          <a className="badge" href={WORKFLOW_URL} target="_blank" rel="noreferrer">Run now ↗</a>
        </div>
      </div>

      {error && <div style={{marginTop:8, color:"#dc2626", fontSize:14}}>{error}</div>}

      <div style={{display:"grid",gap:12,gridTemplateColumns:"repeat(5, minmax(0, 1fr))",marginTop:12}}>
        <div className="mono cardlet">
          <div style={{fontSize:12,color:"#6b7280"}}>Last updated</div>
          <div>{fmt(status?.generated_at_utc)}</div>
        </div>
        <div className="mono cardlet">
          <div style={{fontSize:12,color:"#6b7280"}}>Next run ETA</div>
          <div>{fmt(status?.next_run_eta_utc)}</div>
        </div>
        <div className="mono cardlet">
          <div style={{fontSize:12,color:"#6b7280"}}>Countdown</div>
          <div>{etaMs == null ? "—" : diffHHMMSS(etaMs)}</div>
        </div>
        <div className="mono cardlet">
          <div style={{fontSize:12,color:"#6b7280"}}>Teams</div>
          <div>{status?.teams ?? "—"}</div>
        </div>
        <div className="mono cardlet">
          <div style={{fontSize:12,color:"#6b7280"}}>CSV last modified</div>
          <div>{csvLastMod ?? "—"}</div>
        </div>
      </div>

      {status?.fields && (
        <details style={{marginTop:12}}>
          <summary style={{cursor:"pointer", fontSize:14, color:"#4b5563"}}>Columns</summary>
          <div style={{marginTop:6, fontSize:12, color:"#374151"}}>
            {status.fields.join(", ")}
          </div>
        </details>
      )}
    </div>
  );
}
