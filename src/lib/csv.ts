
export async function fetchCSV(path: string): Promise<string[][]> {
  const res = await fetch(path, { cache: "no-store" })
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)
  const text = await res.text()
  const rows: string[][] = []
  let cur: string[] = []
  let cell = ""
  let inQuotes = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    const nxt = text[i+1]
    if (ch === '"') {
      if (inQuotes && nxt === '"') { cell += '"'; i++; }
      else inQuotes = !inQuotes
    } else if (ch === "," && !inQuotes) {
      cur.push(cell); cell = ""
    } else if (ch === "\n" && !inQuotes) {
      cur.push(cell); rows.push(cur); cur = []; cell = ""
    } else {
      cell += ch
    }
  }
  if (cell.length || cur.length) { cur.push(cell); rows.push(cur) }
  if (rows.length && rows[rows.length-1].every(x => x.trim() === "")) rows.pop()
  return rows
}

export function toObjects(rows: string[][]): Record<string,string>[] {
  if (!rows.length) return []
  const [hdr, ...data] = rows
  return data.map(r => {
    const o: Record<string,string> = {}
    for (let i=0;i<hdr.length;i++) o[hdr[i]] = (r[i] ?? "").trim()
    return o
  })
}

export function num(x: any, fallback = NaN): number {
  const n = typeof x === "number" ? x : parseFloat(String(x))
  return Number.isFinite(n) ? n : fallback
}
