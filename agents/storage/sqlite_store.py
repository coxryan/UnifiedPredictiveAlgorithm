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


def _normalize_name(name: str) -> str:
    norm = name.strip().lower().replace(" ", "_")
    norm = norm.replace("/", "__").replace("-", "_")
    if not norm:
        norm = "dataset"
    return norm


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


def write_dataset(df: pd.DataFrame, name: str, *, if_exists: str = "replace", kind: str = "table") -> str:
    table = _normalize_name(name)
    with get_data_connection() as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)
        _ensure_artifact_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO data_artifacts(path, table_name, kind, updated_at) VALUES(?,?,?,?)",
            (name, table, kind, time.time()),
        )
    return table


def read_dataset(name: str) -> pd.DataFrame:
    table = _normalize_name(name)
    with get_data_connection() as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


def delete_rows(name: str, column: Optional[str] = None, value: Any = None) -> None:
    table = _normalize_name(name)
    with get_data_connection() as conn:
        try:
            if column is None:
                conn.execute(f"DELETE FROM {table}")
            else:
                conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (value,))
        except sqlite3.OperationalError:
            pass


def write_json_blob(name: str, payload: Any) -> None:
    with get_data_connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS data_json (path TEXT PRIMARY KEY, payload TEXT, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO data_json(path, payload, updated_at) VALUES(?,?,?)",
            (name, json.dumps(payload, sort_keys=True), time.time()),
        )
        _ensure_artifact_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO data_artifacts(path, table_name, kind, updated_at) VALUES(?,?,?,?)",
            (name, _normalize_name(name), "json", time.time()),
        )


def read_json_blob(name: str) -> Optional[Any]:
    with get_data_connection() as conn:
        cursor = conn.execute("SELECT payload FROM data_json WHERE path = ?", (name,))
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
