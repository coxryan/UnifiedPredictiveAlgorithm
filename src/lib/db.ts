import initSqlJs, { Database } from "sql.js";
import wasmUrl from "sql.js/dist/sql-wasm.wasm?url";

let dbPromise: Promise<Database> | null = null;

function normalizeTableName(path: string): string {
  let p = path.replace(/\\/g, "/").trim();
  if (p.startsWith("data/")) p = p.slice(5);
  if (p.endsWith(".csv") || p.endsWith(".json")) {
    p = p.slice(0, -4);
  }
  p = p.replace(/\/+/g, "/");
  p = p.replace(/\//g, "__").replace(/-/g, "_").replace(/\./g, "_");
  return p || "dataset";
}

async function getDatabase(): Promise<Database> {
  if (!dbPromise) {
    dbPromise = (async () => {
      const SQL = await initSqlJs({ locateFile: () => wasmUrl });
      const resp = await fetch("data/upa_data.sqlite", { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`Failed to fetch SQLite DB: ${resp.status}`);
      }
      const buf = new Uint8Array(await resp.arrayBuffer());
      return new SQL.Database(buf);
    })();
  }
  return dbPromise;
}

export async function loadTableFromPath(path: string): Promise<Record<string, any>[]> {
  try {
    const db = await getDatabase();
    const table = normalizeTableName(path);
    const stmt = db.prepare(`SELECT * FROM "${table}"`);
    const rows: Record<string, any>[] = [];
    while (stmt.step()) {
      rows.push(stmt.getAsObject());
    }
    stmt.free();
    return rows;
  } catch (err) {
    console.warn(`loadTableFromPath(${path}) failed:`, err);
    return [];
  }
}

export async function loadJsonBlob(path: string): Promise<any> {
  try {
    const db = await getDatabase();
    const stmt = db.prepare("SELECT payload FROM data_json WHERE path = ?");
    stmt.bind([path]);
    let payload: any = null;
    if (stmt.step()) {
      const row = stmt.getAsObject() as { payload?: string };
      if (row && row.payload) {
        payload = JSON.parse(row.payload);
      }
    }
    stmt.free();
    return payload;
  } catch (err) {
    console.warn(`loadJsonBlob(${path}) failed:`, err);
    return null;
  }
}

export function resetDatabaseCache(): void {
  dbPromise = null;
}

export { normalizeTableName };
