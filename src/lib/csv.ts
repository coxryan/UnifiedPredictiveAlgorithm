import { loadTableFromPath, loadJsonBlob } from "./db";

export async function loadTable(name: string) {
  return loadTableFromPath(name);
}

export async function loadJson(key: string) {
  const payload = await loadJsonBlob(key);
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
  if (v === true || v === "True" || v === "true") return true;
  if (v === false || v === "False" || v === "false") return false;
  const n = Number(v);
  return Number.isFinite(n) ? n !== 0 : false;
}
