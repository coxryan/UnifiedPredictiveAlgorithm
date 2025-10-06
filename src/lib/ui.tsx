import React from "react";

export function Badge({ children, tone = "default" }: { children: React.ReactNode; tone?: "default"|"pos"|"neg"|"muted" }) {
  const style: any = {};
  if (tone === "pos") { style.background = "rgba(19,209,142,.12)"; style.borderColor = "#0fbf83"; }
  if (tone === "neg") { style.background = "rgba(255,107,107,.12)"; style.borderColor = "#ff6b6b"; }
  if (tone === "muted") { style.opacity = 0.85; }
  return <span className="badge" style={style}>{children}</span>;
}

const hashColor = (team: string): { from: string; to: string } => {
  let hash = 0;
  const input = (team || "").toUpperCase();
  for (let i = 0; i < input.length; i++) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  const hueA = Math.abs(hash) % 360;
  const hueB = (hueA + 30) % 360;
  return {
    from: `hsl(${hueA}, 70%, 48%)`,
    to: `hsl(${hueB}, 65%, 40%)`,
  };
};

export function TeamLabel({ home, team, neutral, showTags = true }: { home: boolean; team: string; neutral: boolean; showTags?: boolean }) {
  const initials = (team || "")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("" || "?");
  const colors = hashColor(team || "Team");

  return (
    <div className="team-label">
      <div className="team-label__avatar" style={{ background: `linear-gradient(135deg, ${colors.from}, ${colors.to})` }}>
        <span>{initials || "?"}</span>
      </div>
      <div className="team-label__meta">
        <div className="team-label__name">{team || "—"}</div>
        {showTags && (
          <div className="team-label__tags">
            <Badge tone="muted">{home ? "HOME" : "AWAY"}</Badge>
            {neutral && <Badge tone="muted">NEUTRAL</Badge>}
          </div>
        )}
      </div>
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
