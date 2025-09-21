from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Optional

import pandas as pd

def _env_path(name: str, default: str) -> str:
    p = os.environ.get(name, default)
    p = os.path.abspath(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
DATA_DIR = os.environ.get("DATA_DIR", "data").strip() or "data"
DATA_DB_PATH = _env_path("DATA_DB_PATH", os.path.join(DATA_DIR, "upa_data.sqlite"))
CACHE_DB_PATH = _env_path("CACHE_DB_PATH", os.path.join(DATA_DIR, "upa_cache.sqlite"))

_DATA_LOCK = threading.Lock()
_CACHE_LOCK = threading.Lock()


@contextmanager
def _connect(path: str, lock: threading.Lock):
    with lock:
        conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        finally:
            conn.close()


def get_data_connection():
    return _connect(DATA_DB_PATH, _DATA_LOCK)


def path_to_table_name(path: str) -> str:
    path = path.replace("\\", "/").strip()
    if path.startswith("data/"):
        path = path[5:]
    if path.endswith(".csv") or path.endswith(".json"):
        path = path.rsplit(".", 1)[0]
    path = path.strip("/")
    path = path.replace("/", "__").replace("-", "_").replace(".", "_")
    if not path:
        path = "dataset"
    return path


def _ensure_artifact_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_artifacts (
            path TEXT PRIMARY KEY,
            table_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )


def write_table_from_path(df: pd.DataFrame, path: str, *, if_exists: str = "replace") -> str:
    table = path_to_table_name(path)
    with get_data_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
        _ensure_artifact_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO data_artifacts(path, table_name, kind, updated_at) VALUES(?,?,?,?)",
            (path, table, "table", time.time()),
        )
    return table


def read_table_from_path(path: str) -> pd.DataFrame:
    table = path_to_table_name(path)
    with get_data_connection() as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


def write_named_table(df: pd.DataFrame, table: str, *, if_exists: str = "replace", kind: str = "table") -> None:
    with get_data_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
        _ensure_artifact_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO data_artifacts(path, table_name, kind, updated_at) VALUES(?,?,?,?)",
            (f"table:{table}", table, kind, time.time()),
        )


def read_named_table(table: str) -> pd.DataFrame:
    with get_data_connection() as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


def delete_rows(table: str, column: Optional[str] = None, value: Any = None) -> None:
    with get_data_connection() as conn:
        try:
            if column is None:
                conn.execute(f"DELETE FROM {table}")
            else:
                conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (value,))
        except sqlite3.OperationalError:
            # table may not exist yet; ignore
            pass


def write_json_blob(path: str, payload: Any) -> None:
    with get_data_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS data_json (path TEXT PRIMARY KEY, payload TEXT, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO data_json(path, payload, updated_at) VALUES(?,?,?)",
            (path, json.dumps(payload, sort_keys=True), time.time()),
        )
        _ensure_artifact_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO data_artifacts(path, table_name, kind, updated_at) VALUES(?,?,?,?)",
            (path, path_to_table_name(path), "json", time.time()),
        )


def read_json_blob(path: str) -> Optional[Any]:
    with get_data_connection() as conn:
        cursor = conn.execute("SELECT payload FROM data_json WHERE path = ?", (path,))
        row = cursor.fetchone()
        if not row:
            return None
        return json.loads(row["payload"])


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS api_cache (
            cache_key TEXT PRIMARY KEY,
            payload TEXT,
            expires_at REAL
        )
        """
    )


def write_cache_entry(key: str, payload: Any, ttl_seconds: int, *, db_path: Optional[str] = None) -> None:
    expires = time.time() + ttl_seconds
    path = db_path or CACHE_DB_PATH
    with _connect(path, _CACHE_LOCK) as conn:
        _ensure_cache_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO api_cache(cache_key, payload, expires_at) VALUES(?,?,?)",
            (key, json.dumps(payload, sort_keys=True), expires),
        )


def read_cache_entry(key: str, *, db_path: Optional[str] = None) -> Optional[Any]:
    now = time.time()
    path = db_path or CACHE_DB_PATH
    with _connect(path, _CACHE_LOCK) as conn:
        _ensure_cache_table(conn)
        row = conn.execute(
            "SELECT payload, expires_at FROM api_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] is not None and row["expires_at"] < now:
            conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (key,))
            return None
        return json.loads(row["payload"])


def purge_cache_entry(key: str, *, db_path: Optional[str] = None) -> None:
    path = db_path or CACHE_DB_PATH
    with _connect(path, _CACHE_LOCK) as conn:
        _ensure_cache_table(conn)
        conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (key,))
