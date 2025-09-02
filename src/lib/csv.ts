export async function loadText(path: string) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to fetch ${path}: ${r.status}`);
  return r.text();
}

export async function loadCsv(path: string) {
  const txt = await loadText(path);
  const lines = txt.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const cols = lines[0].split(",").map((c) => c.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const o: Record<string, string> = {};
    cols.forEach((c, i) => (o[c] = (cells[i] ?? "").trim()));
    return o;
  });
}

export function toNum(v: any) {
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

export function fmtNum(v: any, opts: Intl.NumberFormatOptions = {}) {
  if (v === null || v === undefined || v === "" || v === "NaN") return "—";
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { maximumFractionDigits: 1, ...opts });
}

export function fmtPct01(n: number | string | undefined) {
  if (n === undefined || n === null || n === "" || n === "NaN") return "—";
  const x = Number(n);
  if (!Number.isFinite(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}

export function playedBool(v: any) {
  return v === true || v === "True" || v === "true";
}