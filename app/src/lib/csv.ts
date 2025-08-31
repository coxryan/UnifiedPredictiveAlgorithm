import Papa from 'papaparse'
export type Row = Record<string, string | number | boolean | null>
export async function loadCsv(path: string): Promise<Row[]> {
  const res = await fetch(path, { cache: 'no-store' })
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`)
  const text = await res.text()
  const parsed = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true })
  if (parsed.errors?.length) console.warn('CSV parse errors', parsed.errors.slice(0,3))
  return (parsed.data as any[]) as Row[]
}
