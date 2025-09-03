# agents/lib/cache.py
import os, time, pickle, hashlib, json
from typing import Any, Dict, Callable

class ApiCache:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _key_to_path(self, namespace: str, key: Dict[str, Any]):
        key_str = json.dumps(key, sort_keys=True, separators=(",", ":"))
        h = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
        ns_dir = os.path.join(self.root, namespace)
        os.makedirs(ns_dir, exist_ok=True)
        return os.path.join(ns_dir, f"{h}.pkl")

    def get(self, namespace: str, key: Dict[str, Any], ttl_seconds: int):
        p = self._key_to_path(namespace, key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
            ts = payload.get("_ts", 0)
            if ttl_seconds > 0 and (time.time() - ts) > ttl_seconds:
                return None
            return payload.get("data", None)
        except Exception:
            return None

    def set(self, namespace: str, key: Dict[str, Any], data: Any):
        p = self._key_to_path(namespace, key)
        try:
            with open(p, "wb") as f:
                pickle.dump({"_ts": time.time(), "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

def cached_call(cache: ApiCache, namespace: str, key: Dict[str, Any], ttl_seconds: int, fetch_fn: Callable[[], Any]):
    v = cache.get(namespace, key, ttl_seconds)
    if v is not None:
        return v, True
    data = fetch_fn()
    cache.set(namespace, key, data)
    return data, False