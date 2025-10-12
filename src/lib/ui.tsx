import React from "react";

export function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default"|"pos"|"neg"|"muted" }) {
  const style: any = {};
  if (tone === "pos") { style.background = "rgba(19,209,142,.12)"; style.borderColor = "#0fbf83"; }
  if (tone === "neg") { style.background = "rgba(255,107,107,.12)"; style.borderColor = "#ff6b6b"; }
  if (tone === "muted") { style.opacity = 0.85; }
  return <span className="badge" style={style}>{children}</span>;
}

export function TeamLabel({ home, team, neutral, showTags = true }: { home: boolean; team: string; neutral: boolean; showTags?: boolean }) {
  const words = (team || "—").toString().trim().split(/\s+/).filter(Boolean);
  const displayWords = words.length ? words : ["—"];
  return (
    <div className="team-label">
      <span className={`team-label__name${home ? " team-label__name--home" : ""}`}>
        {displayWords.map((word, idx) => (
          <span key={`${word}-${idx}`}>{word}</span>
        ))}
      </span>
      {showTags && (
        <span className="team-label__tags">
          <Badge tone="muted">{home ? "HOME" : "AWAY"}</Badge>
          {neutral && <Badge tone="muted">NEUTRAL</Badge>}
        </span>
      )}
    </div>
  );
}

export function downloadDataset(filename: string, rows: Record<string, any>[]) {
  if (!rows.length) return;
  const cols = Object.keys(rows[0]);
  const body = rows.map(r => cols.map(c => (r[c] ?? "")).join(",")).join("\n");
  const csv = cols.join(",") + "\n" + body + "\n";
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

export function nextUpcomingWeek(rows: {week: string, date?: string, played?: any}[]): number | null {
  if (!rows.length) return null;
  const today = new Date();
  const withDates = rows
    .map(r => ({ r, d: r.date ? new Date(r.date + "T00:00:00Z") : null }))
    .filter((x): x is {r:any,d:Date} => x.d !== null);
  const future = withDates.filter(x => x.d.getTime() >= today.getTime());
  if (future.length) {
    const wk = Math.min(...future.map(x => Number(x.r.week)));
    return isFinite(wk) ? wk : null;
  }
  const unplayed = rows.filter(r => !(r.played === true || r.played === "True" || r.played === "true"));
  if (unplayed.length) {
    const wk = Math.min(...unplayed.map(r => Number(r.week)));
    return isFinite(wk) ? wk : null;
  }
  const anyWk = Math.min(...rows.map(r => Number(r.week)));
  return isFinite(anyWk) ? anyWk : null;
}

export function scoreText(a?: string, h?: string) {
  if (a==null || h==null || a==="" || h==="") return "—";
  return `${a} @ ${h}`;
}
