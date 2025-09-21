import { loadTableFromPath, loadJsonBlob } from "./db";

export async function loadCsv(path: string) {
  return loadTableFromPath(path);
}

export async function loadJson(path: string) {
  const payload = await loadJsonBlob(path);
  return payload ?? {};
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
