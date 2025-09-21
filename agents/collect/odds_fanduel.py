from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from .cache import ApiCache
from .config import ODDS_API_KEY, CACHE_ONLY, DATA_DIR, _dbg
from .status import _upsert_status_market_source
from agents.storage.sqlite_store import read_json_blob, write_json_blob


def _odds_api_fetch_fanduel(year: int, weeks: List[int], cache: ApiCache) -> List[Dict[str, Any]]:
    """
    Return list of {home_name, away_name, point_home_book, commence_time} using The Odds API.
    We cache one “slate” per UTC day; TTL controlled by ODDS_CACHE_TTL_DAYS in ApiCache instance.
    """
    if not ODDS_API_KEY:
        _dbg("odds_api_fetch_fanduel: missing ODDS_API_KEY")
        try:
            _upsert_status_market_source(
                market_used="none",
                market_requested="fanduel",
                fallback_reason="missing ODDS_API_KEY",
                extra={"market_error": "missing_odds_api_key"}
            )
        except Exception:
            pass
        return []

    day_key = pd.Timestamp.utcnow().strftime("%Y%m%d")
    key = f"oddsapi:fanduel:daily:{day_key}"
    ok, cached = cache.get(key)
    _dbg(f"odds_api_fetch_fanduel: cache_ok={ok} cached_items={len(cached) if ok and cached is not None else 0} key={key}")
    if ok and cached is not None:
        return list(cached)

    if CACHE_ONLY:
        cache.set(key, [])
        _dbg("odds_api_fetch_fanduel: CACHE_ONLY=1 and cache miss → empty cached")
        return []

    out: List[Dict[str, Any]] = []
    sport = "americanfootball_ncaaf"
    base = "https://api.the-odds-api.com/v4/sports/{sport}/odds"

    try:
        url = base.format(sport=sport)
        agg: List[Dict[str, Any]] = []
        for page in range(1, 6):
            params = {
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "spreads",
                "bookmakers": "fanduel",
                "oddsFormat": "american",
                "dateFormat": "iso",
                "page": page,
            }
            r = requests.get(url, params=params, timeout=25)
            if r.status_code == 404:
                _dbg(f"odds_api_fetch_fanduel: page {page} -> 404 (stop)")
                break
            r.raise_for_status()
            data = r.json() or []
            _dbg(f"odds_api_fetch_fanduel: page={page} items={len(data)}")
            if not data:
                break
            agg.extend(data)

        rows: List[Dict[str, Any]] = []
        for game in agg:
            bks = game.get("bookmakers") or []
            if not bks:
                continue
            mk = None
            for bk in bks:
                markets = bk.get("markets") or []
                mk = next((m for m in markets if m.get("key") == "spreads"), None)
                if mk:
                    break
            if not mk:
                continue
            outs = mk.get("outcomes") or []
            g_home = game.get("home_team")
            g_away = game.get("away_team")
            out_home = next((o for o in outs if o.get("name") == g_home), None)
            if out_home is None:
                continue
            try:
                point_home_book = float(out_home.get("point")) if out_home.get("point") is not None else None
            except Exception:
                point_home_book = None
            if point_home_book is None:
                continue
            rows.append(
                {
                    "home_name": g_home,
                    "away_name": g_away,
                    "point_home_book": point_home_book,
                    "commence_time": game.get("commence_time"),
                }
            )

        _dbg(f"odds_api_fetch_fanduel: total_items={len(agg)} usable_rows={len(rows)} (saving to cache key={key})")
        cache.set(key, rows)
        return rows
    except Exception as e:
        print(f"[warn] odds api fetch failed: {e}")
        cache.set(key, [])
        return []


# ------------- Name resolution (FanDuel → schedule) -------------------------
import difflib


def _date_from_iso(s: Any) -> Optional[str]:
    if s is None:
        return None
    try:
        if isinstance(s, str) and len(s) >= 10:
            return s[:10]
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return str(dt.date())
    except Exception:
        return None
    return None


def _best_fuzzy_match(q_name: str, candidates: Iterable[str], normalize_fn) -> Tuple[Optional[str], float, str]:
    try:
        qn = normalize_fn(q_name or "")
        cand_norm = [(c, normalize_fn(c)) for c in candidates]
        best_c, best_s = None, 0.0
        for raw, cn in cand_norm:
            s = difflib.SequenceMatcher(None, qn, cn).ratio()
            if s > best_s:
                best_s, best_c = s, raw
        return best_c, float(best_s), qn
    except Exception:
        return None, 0.0, str(q_name or "")


def _resolve_names_to_schedule(schedule_df: pd.DataFrame, name: str) -> Optional[str]:
    """Map Odds API/FanDuel team name to schedule school name using robust normalization."""
    if not name:
        return None

    MASCOT_WORDS = {
        "bulldogs","wildcats","tigers","aggressors","agies","aggies","gators","longhorns","buckeyes","nittany","lions","nittany lions",
        "yellow","jackets","yellow jackets","demon","deacons","demon deacons","crimson","tide","crimson tide","redhawks","red hawks",
        "chippewas","huskies","zips","warhawks","cardinals","terrapins","razorbacks","trojans","bruins","gophers","badgers","cornhuskers",
        "rebels","utes","bearcats","cowboys","mountaineers","hurricanes","seminoles","sooners","volunteers","commodores",
        "panthers","wolfpack","falcons","eagles","golden eagles","golden","golden flashes","flashes","blazers","tar","heels","tar heels",
        "skyhawks","gamecocks","blue devils","blue","blue hens","scarlet knights","knights","rainbow warriors","warriors","rainbows",
        "rainbow","broncos","lancers","gaels","lions","rams","owls","spartans","tigers","tide","pirates","raiders","mean green",
        "anteaters","jaguars","trojans","minutemen","red wolves","hokies","uconn huskies","bulls","thundering herd","mustangs","cavaliers",
        "paladins","mocs","moccasins","mocsins","thunderbirds","mountaineers","phoenix","blue raiders","jayhawks","illini","aztecs",
        "redbirds","salukis","lumberjacks","cowgirls","cowboys","bears","mavericks","rivers","catamounts","governors","bengals",
        "buccaneers","runnin","runnin bulldogs","runnin' bulldogs","runnin-bulldogs","lobos","vandals","owls","golden hurricane",
        "scarlet","scarlet knights",
        # appended for FanDuel variants
        "midshipmen","dukes","bearkats","roadrunners","cardinal","cougars","knights"
    }

    def strip_diacritics(s: str) -> str:
        try:
            s2 = s.replace("ʻ", "'").replace("’", "'").replace("`", "'")
            return s2.encode("ascii", "ignore").decode("ascii", "ignore")
        except Exception:
            return s

    STOP_WORDS = {"university", "univ", "the", "of", "men's", "womens", "women's", "college", "st", "st.", "and", "at", "amp", "amp;"}

    def drop_mascots(tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        toks = tokens[:]
        i = 0
        out = []
        while i < len(toks):
            if i + 1 < len(toks) and f"{toks[i]} {toks[i+1]}" in MASCOT_WORDS and len(toks) > 2:
                i += 2
                continue
            if toks[i] in MASCOT_WORDS and len(toks) > 1:
                i += 1
                continue
            out.append(toks[i])
            i += 1
        return out if out else tokens

    def clean(s: str) -> str:
        s = strip_diacritics(s or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s.endswith(" university"):
            s = s[:-11]
        toks = [t for t in s.split() if t and t not in STOP_WORDS]
        toks = drop_mascots(toks)
        return " ".join(toks)

    def no_paren(s: str) -> str:
        import re
        return re.sub(r"\([^\)]*\)", " ", s or "").strip()

    def acronym_from(s: str) -> Optional[str]:
        if not s:
            return None
        u = s.upper()
        if u in {"LSU","TCU","UCLA","USC","UAB","UTEP","UTSA","BYU","SMU","FIU","FAU","UTEP","UNLV","UCF","UIC"}:
            return u
        return None

    # FanDuel -> schedule alias map
    alias_map = {
        "air force falcons": "air force",
        "akron zips": "akron",
        "alabama crimson tide": "alabama",
        "appalachian state mountaineers": "appalachian state",
        "arizona wildcats": "arizona",
        "arizona state sun devils": "arizona state",
        "arkansas razorbacks": "arkansas",
        "arkansas state red wolves": "arkansas state",
        "army black knights": "army",
        "auburn tigers": "auburn",
        "ball state cardinals": "ball state",
        "baylor bears": "baylor",
        "boise state broncos": "boise state",
        "boston college eagles": "boston college",
        "bowling green falcons": "bowling green",
        "buffalo bulls": "buffalo",
        "byu cougars": "brigham young",
        "california golden bears": "california",
        "central michigan chippewas": "central michigan",
        "charlotte 49ers": "charlotte",
        "cincinnati bearcats": "cincinnati",
        "clemson tigers": "clemson",
        "coastal carolina chanticleers": "coastal carolina",
        "colorado buffaloes": "colorado",
        "colorado state rams": "colorado state",
        "duke blue devils": "duke",
        "east carolina pirates": "east carolina",
        "eastern michigan eagles": "eastern michigan",
        "florida gators": "florida",
        "florida atlantic owls": "florida atlantic",
        "florida international panthers": "florida international",
        "fresno state bulldogs": "fresno state",
        "georgia bulldogs": "georgia",
        "georgia southern eagles": "georgia southern",
        "georgia state panthers": "georgia state",
        "georgia tech yellow jackets": "georgia tech",
        "hawaii rainbow warriors": "hawai'i",
        "houston cougars": "houston",
        "illinois fighting illini": "illinois",
        "indiana hoosiers": "indiana",
        "iowa hawkeyes": "iowa",
        "iowa state cyclones": "iowa state",
        "james madison dukes": "james madison",
        "kansas jayhawks": "kansas",
        "kansas state wildcats": "kansas state",
        "kent state golden flashes": "kent state",
        "kentucky wildcats": "kentucky",
        "liberty flames": "liberty",
        "louisiana tech bulldogs": "louisiana tech",
        "louisiana lafayette ragin cajuns": "louisiana",
        "louisiana monroe warhawks": "louisiana-monroe",
        "louisville cardinals": "louisville",
        "lsu tigers": "louisiana state",
        "marshall thundering herd": "marshall",
        "maryland terrapins": "maryland",
        "memphis tigers": "memphis",
        "miami hurricanes": "miami",
        "miami (oh) redhawks": "miami (oh)",
        "michigan wolverines": "michigan",
        "michigan state spartans": "michigan state",
        "middle tennessee blue raiders": "middle tennessee",
        "minnesota golden gophers": "minnesota",
        "mississippi state bulldogs": "mississippi state",
        "mississippi rebels": "mississippi",
        "missouri tigers": "missouri",
        "navy midshipmen": "navy",
        "nebraska cornhuskers": "nebraska",
        "nevada wolf pack": "nevada",
        "new mexico lobos": "new mexico",
        "new mexico state aggies": "new mexico state",
        "north carolina tar heels": "north carolina",
        "nc state wolfpack": "nc state",
        "north texas mean green": "north texas",
        "northern illinois huskies": "northern illinois",
        "northwestern wildcats": "northwestern",
        "notre dame fighting irish": "notre dame",
        "ohio bobcats": "ohio",
        "ohio state buckeyes": "ohio state",
        "oklahoma sooners": "oklahoma",
        "oklahoma state cowboys": "oklahoma state",
        "old dominion monarchs": "old dominion",
        "oregon ducks": "oregon",
        "oregon state beavers": "oregon state",
        "penn state nittany lions": "penn state",
        "pittsburgh panthers": "pittsburgh",
        "purdue boilermakers": "purdue",
        "rice owls": "rice",
        "rutgers scarlet knights": "rutgers",
        "san diego state aztecs": "san diego state",
        "san jose state spartans": "san jose state",
        "smu mustangs": "southern methodist",
        "south alabama jaguars": "south alabama",
        "south carolina gamecocks": "south carolina",
        "south florida bulls": "south florida",
        "southern miss golden eagles": "southern mississippi",
        "stanford cardinal": "stanford",
        "syracuse orange": "syracuse",
        "tcu horned frogs": "texas christian",
        "temple owls": "temple",
        "tennessee volunteers": "tennessee",
        "texas longhorns": "texas",
        "texas a and m": "texas a&m",
        "texas a m": "texas a&m",
        "texas a&m aggies": "texas a&m",
        "texas state bobcats": "texas state",
        "texas tech red raiders": "texas tech",
        "toledo rockets": "toledo",
        "troy trojans": "troy",
        "tulane green wave": "tulane",
        "tulsa golden hurricane": "tulsa",
        "uab blazers": "uab",
        "ucf knights": "central florida",
        "ucla bruins": "ucla",
        "unlv rebels": "nevada las vegas",
        "usc trojans": "southern california",
        "utah utes": "utah",
        "utah state aggies": "utah state",
        "utsa roadrunners": "texas san antonio",
        "virginia cavaliers": "virginia",
        "virginia tech hokies": "virginia tech",
        "wake forest demon deacons": "wake forest",
        "washington huskies": "washington",
        "washington state cougars": "washington state",
        "western kentucky hilltoppers": "western kentucky",
        "western michigan broncos": "western michigan",
        "wisconsin badgers": "wisconsin",
        "wyoming cowboys": "wyoming",
    }
    alias_map.update({
        "arizona wildcats":"arizona","arkansas razorbacks":"arkansas","arkansas state red wolves":"arkansas state",
        "byu cougars":"brigham young","boston college eagles":"boston college","california golden bears":"california",
        "charlotte 49ers":"charlotte","coastal carolina chanticleers":"coastal carolina","colorado state rams":"colorado state",
        "east carolina pirates":"east carolina","eastern michigan eagles":"eastern michigan","houston cougars":"houston",
        "iowa state cyclones":"iowa state","kansas state wildcats":"kansas state","miami hurricanes":"miami",
        "miami (oh) redhawks":"miami (oh)","mississippi state bulldogs":"mississippi state","nc state wolfpack":"nc state",
        "navy midshipmen":"navy","nevada wolf pack":"nevada","north carolina tar heels":"north carolina",
        "oklahoma state cowboys":"oklahoma state","old dominion monarchs":"old dominion","san diego st aztecs":"san diego state",
        "san jose st spartans":"san jose state","south carolina gamecocks":"south carolina","stanford cardinal":"stanford",
        "uab blazers":"uab","ucf knights":"central florida","unlv rebels":"nevada las vegas","usc trojans":"southern california",
        "utah utes":"utah","utep miners":"texas el paso","utsa roadrunners":"texas san antonio","washington huskies":"washington",
        "western michigan broncos":"western michigan","appalachian state mountaineers":"appalachian state","virginia cavaliers":"virginia",
        "texas tech red raiders":"texas tech","baylor bears":"baylor","purdue boilermakers":"purdue","air force falcons":"air force",
        "wyoming cowboys":"wyoming","georgia bulldogs":"georgia","georgia tech yellow jackets":"georgia tech",
        "boise state broncos":"boise state","washington state cougars":"washington state"
    })

    # Allow external overrides
    try:
        for _pth in (os.path.join(DATA_DIR, "team_aliases.json"), "agents/team_aliases.json", "data/team_aliases.json"):
            if os.path.exists(_pth):
                with open(_pth, "r") as _f:
                    _extra = json.load(_f) or {}
                _norm_extra = {clean(str(k)): clean(str(v)) for k, v in _extra.items() if isinstance(k, str) and isinstance(v, str)}
                alias_map.update(_norm_extra)
                break
    except Exception:
        pass

    def alias(s: str) -> str:
        cs = clean(s)
        return alias_map.get(cs, cs)

    if "home_team" not in schedule_df.columns or "away_team" not in schedule_df.columns:
        return None

    schools = set(str(x).strip() for x in schedule_df["home_team"].dropna().unique()) | set(
        str(x).strip() for x in schedule_df["away_team"].dropna().unique()
    )

    norm_map: Dict[str, str] = {}
    acro_map: Dict[str, str] = {}
    token_index: List[Tuple[str, set]] = []

    for sch in schools:
        can = alias(sch)
        norm_map[can] = sch
        ac = acronym_from(sch)
        if ac:
            acro_map[ac] = sch
        token_index.append((sch, set(can.split())))

    q_raw = name
    q_norm = alias(q_raw)

    if q_norm in norm_map:
        return norm_map[q_norm]

    q_np = no_paren(q_norm)
    if q_np in norm_map:
        return norm_map[q_np]

    q_alias = alias(q_np)
    if q_alias in norm_map:
        return norm_map[q_alias]

    q_acro = acronym_from(q_raw)
    if q_acro and q_acro in acro_map:
        return acro_map[q_acro]

    q_tokens = set(q_alias.split())
    best_team, best_score = None, 0.0
    for sch, toks in token_index:
        if not toks:
            continue
        inter = len(q_tokens & toks)
        if inter == 0:
            continue
        union = len(q_tokens | toks)
        jacc = inter / float(union) if union else 0.0
        contain = 1.0 if (q_tokens.issubset(toks) or toks.issubset(q_tokens)) else 0.0
        score = jacc + 0.25 * contain
        if score > best_score:
            best_score, best_team = score, sch

    if best_team and best_score >= 0.40:
        return best_team
    return None


def _resolve_names_to_schedule_with_details(schedule_df: pd.DataFrame, name: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Detailed resolver: (resolved_school, detail_dict)."""
    detail: Dict[str, Any] = {"q_raw": name, "q_norm": None, "q_no_paren": None, "q_alias": None, "best_team": None, "best_score": None}
    if not name:
        return None, detail

    # Keep logic in sync with _resolve_names_to_schedule
    MASCOT_WORDS = {
        "bulldogs","wildcats","tigers","aggressors","agies","aggies","gators","longhorns","buckeyes","nittany","lions","nittany lions",
        "yellow","jackets","yellow jackets","demon","deacons","demon deacons","crimson","tide","crimson tide","redhawks","red hawks",
        "chippewas","huskies","zips","warhawks","cardinals","terrapins","razorbacks","trojans","bruins","gophers","badgers","cornhuskers",
        "rebels","utes","bearcats","cowboys","mountaineers","hurricanes","seminoles","sooners","volunteers","commodores",
        "panthers","wolfpack","falcons","eagles","golden eagles","golden","golden flashes","flashes","blazers","tar","heels","tar heels",
        "skyhawks","gamecocks","blue devils","blue","blue hens","scarlet knights","knights","rainbow warriors","warriors","rainbows",
        "rainbow","broncos","lancers","gaels","lions","rams","owls","spartans","tigers","tide","pirates","raiders","mean green",
        "anteaters","jaguars","trojans","minutemen","red wolves","hokies","uconn huskies","bulls","thundering herd","mustangs","cavaliers",
        "paladins","mocs","moccasins","mocsins","thunderbirds","mountaineers","phoenix","blue raiders","jayhawks","illini","aztecs",
        "redbirds","salukis","lumberjacks","cowgirls","cowboys","bears","mavericks","rivers","catamounts","governors","bengals",
        "buccaneers","runnin","runnin bulldogs","runnin' bulldogs","runnin-bulldogs","lobos","vandals","owls","golden hurricane",
        "scarlet","scarlet knights",
        # appended for FanDuel variants
        "midshipmen","dukes","bearkats","roadrunners","cardinal","cougars","knights"
    }

    def strip_diacritics(s: str) -> str:
        try:
            s2 = s.replace("ʻ", "'").replace("’", "'").replace("`", "'")
            return s2.encode("ascii", "ignore").decode("ascii", "ignore")
        except Exception:
            return s

    STOP_WORDS = {"university", "univ", "the", "of", "men's", "womens", "women's", "college", "st", "st.", "and", "at", "amp", "amp;"}

    def drop_mascots(tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens
        toks = tokens[:]
        i = 0
        out = []
        while i < len(toks):
            if i + 1 < len(toks) and f"{toks[i]} {toks[i+1]}" in MASCOT_WORDS and len(toks) > 2:
                i += 2
                continue
            if toks[i] in MASCOT_WORDS and len(toks) > 1:
                i += 1
                continue
            out.append(toks[i])
            i += 1
        return out if out else tokens

    def clean(s: str) -> str:
        s = strip_diacritics(s or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if s.endswith(" university"):
            s = s[:-11]
        toks = [t for t in s.split() if t and t not in STOP_WORDS]
        toks = drop_mascots(toks)
        return " ".join(toks)

    def no_paren(s: str) -> str:
        import re
        return re.sub(r"\([^\)]*\)", " ", s or "").strip()

    def acronym_from(s: str) -> Optional[str]:
        if not s:
            return None
        u = s.upper()
        if u in {"LSU","TCU","UCLA","USC","UAB","UTEP","UTSA","BYU","SMU","FIU","FAU","UTEP","UNLV","UCF","UIC"}:
            return u
        return None

    # FanDuel -> schedule alias map
    alias_map: Dict[str, str] = {
        "air force falcons": "air force",
        "akron zips": "akron",
        "alabama crimson tide": "alabama",
        "appalachian state mountaineers": "appalachian state",
        "arizona wildcats": "arizona",
        "arizona state sun devils": "arizona state",
        "arkansas razorbacks": "arkansas",
        "arkansas state red wolves": "arkansas state",
        "army black knights": "army",
        "auburn tigers": "auburn",
        "ball state cardinals": "ball state",
        "baylor bears": "baylor",
        "boise state broncos": "boise state",
        "boston college eagles": "boston college",
        "bowling green falcons": "bowling green",
        "buffalo bulls": "buffalo",
        "byu cougars": "brigham young",
        "california golden bears": "california",
        "central michigan chippewas": "central michigan",
        "charlotte 49ers": "charlotte",
        "cincinnati bearcats": "cincinnati",
        "clemson tigers": "clemson",
        "coastal carolina chanticleers": "coastal carolina",
        "colorado buffaloes": "colorado",
        "colorado state rams": "colorado state",
        "duke blue devils": "duke",
        "east carolina pirates": "east carolina",
        "eastern michigan eagles": "eastern michigan",
        "florida gators": "florida",
        "florida atlantic owls": "florida atlantic",
        "florida international panthers": "florida international",
        "fresno state bulldogs": "fresno state",
        "georgia bulldogs": "georgia",
        "georgia southern eagles": "georgia southern",
        "georgia state panthers": "georgia state",
        "georgia tech yellow jackets": "georgia tech",
        "hawaii rainbow warriors": "hawai'i",
        "houston cougars": "houston",
        "illinois fighting illini": "illinois",
        "indiana hoosiers": "indiana",
        "iowa hawkeyes": "iowa",
        "iowa state cyclones": "iowa state",
        "james madison dukes": "james madison",
        "kansas jayhawks": "kansas",
        "kansas state wildcats": "kansas state",
        "kent state golden flashes": "kent state",
        "kentucky wildcats": "kentucky",
        "liberty flames": "liberty",
        "louisiana tech bulldogs": "louisiana tech",
        "louisiana lafayette ragin cajuns": "louisiana",
        "louisiana monroe warhawks": "louisiana-monroe",
        "louisville cardinals": "louisville",
        "lsu tigers": "louisiana state",
        "marshall thundering herd": "marshall",
        "maryland terrapins": "maryland",
        "memphis tigers": "memphis",
        "miami hurricanes": "miami",
        "miami (oh) redhawks": "miami (oh)",
        "michigan wolverines": "michigan",
        "michigan state spartans": "michigan state",
        "middle tennessee blue raiders": "middle tennessee",
        "minnesota golden gophers": "minnesota",
        "mississippi state bulldogs": "mississippi state",
        "mississippi rebels": "mississippi",
        "missouri tigers": "missouri",
        "navy midshipmen": "navy",
        "nebraska cornhuskers": "nebraska",
        "nevada wolf pack": "nevada",
        "new mexico lobos": "new mexico",
        "new mexico state aggies": "new mexico state",
        "north carolina tar heels": "north carolina",
        "nc state wolfpack": "nc state",
        "north texas mean green": "north texas",
        "northern illinois huskies": "northern illinois",
        "northwestern wildcats": "northwestern",
        "notre dame fighting irish": "notre dame",
        "ohio bobcats": "ohio",
        "ohio state buckeyes": "ohio state",
        "oklahoma sooners": "oklahoma",
        "oklahoma state cowboys": "oklahoma state",
        "old dominion monarchs": "old dominion",
        "oregon ducks": "oregon",
        "oregon state beavers": "oregon state",
        "penn state nittany lions": "penn state",
        "pittsburgh panthers": "pittsburgh",
        "purdue boilermakers": "purdue",
        "rice owls": "rice",
        "rutgers scarlet knights": "rutgers",
        "san diego state aztecs": "san diego state",
        "san jose state spartans": "san jose state",
        "smu mustangs": "southern methodist",
        "south alabama jaguars": "south alabama",
        "south carolina gamecocks": "south carolina",
        "south florida bulls": "south florida",
        "southern miss golden eagles": "southern mississippi",
        "stanford cardinal": "stanford",
        "syracuse orange": "syracuse",
        "tcu horned frogs": "texas christian",
        "temple owls": "temple",
        "tennessee volunteers": "tennessee",
        "texas longhorns": "texas",
        "texas a and m": "texas a&m",
        "texas a m": "texas a&m",
        "texas a&m aggies": "texas a&m",
        "texas state bobcats": "texas state",
        "texas tech red raiders": "texas tech",
        "toledo rockets": "toledo",
        "troy trojans": "troy",
        "tulane green wave": "tulane",
        "tulsa golden hurricane": "tulsa",
        "uab blazers": "uab",
        "ucf knights": "central florida",
        "ucla bruins": "ucla",
        "unlv rebels": "nevada las vegas",
        "usc trojans": "southern california",
        "utah utes": "utah",
        "utah state aggies": "utah state",
        "utsa roadrunners": "texas san antonio",
        "virginia cavaliers": "virginia",
        "virginia tech hokies": "virginia tech",
        "wake forest demon deacons": "wake forest",
        "washington huskies": "washington",
        "washington state cougars": "washington state",
        "western kentucky hilltoppers": "western kentucky",
        "western michigan broncos": "western michigan",
        "wisconsin badgers": "wisconsin",
        "wyoming cowboys": "wyoming",
    }
    alias_map.update({
        "arizona wildcats":"arizona","arkansas razorbacks":"arkansas","arkansas state red wolves":"arkansas state",
        "byu cougars":"brigham young","boston college eagles":"boston college","california golden bears":"california",
        "charlotte 49ers":"charlotte","coastal carolina chanticleers":"coastal carolina","colorado state rams":"colorado state",
        "east carolina pirates":"east carolina","eastern michigan eagles":"eastern michigan","houston cougars":"houston",
        "iowa state cyclones":"iowa state","kansas state wildcats":"kansas state","miami hurricanes":"miami",
        "miami (oh) redhawks":"miami (oh)","mississippi state bulldogs":"mississippi state","nc state wolfpack":"nc state",
        "navy midshipmen":"navy","nevada wolf pack":"nevada","north carolina tar heels":"north carolina",
        "oklahoma state cowboys":"oklahoma state","old dominion monarchs":"old dominion","san diego st aztecs":"san diego state",
        "san jose st spartans":"san jose state","south carolina gamecocks":"south carolina","stanford cardinal":"stanford",
        "uab blazers":"uab","ucf knights":"central florida","unlv rebels":"nevada las vegas","usc trojans":"southern california",
        "utah utes":"utah","utep miners":"texas el paso","utsa roadrunners":"texas san antonio","washington huskies":"washington",
        "western michigan broncos":"western michigan","appalachian state mountaineers":"appalachian state","virginia cavaliers":"virginia",
        "texas tech red raiders":"texas tech","baylor bears":"baylor","purdue boilermakers":"purdue","air force falcons":"air force",
        "wyoming cowboys":"wyoming","georgia bulldogs":"georgia","georgia tech yellow jackets":"georgia tech",
        "boise state broncos":"boise state","washington state cougars":"washington state"
    })

    # External alias overrides
    try:
        for _pth in (os.path.join(DATA_DIR, "team_aliases.json"), "agents/team_aliases.json", "data/team_aliases.json"):
            if os.path.exists(_pth):
                with open(_pth, "r") as _f:
                    _extra = json.load(_f) or {}
                _norm_extra = {clean(str(k)): clean(str(v)) for k, v in _extra.items() if isinstance(k, str) and isinstance(v, str)}
                alias_map.update(_norm_extra)
                break
    except Exception:
        pass

    def alias(s: str) -> str:
        cs = clean(s)
        return alias_map.get(cs, cs)

    if "home_team" not in schedule_df.columns or "away_team" not in schedule_df.columns:
        return None, detail

    schools = set(str(x).strip() for x in schedule_df["home_team"].dropna().unique()) | set(
        str(x).strip() for x in schedule_df["away_team"].dropna().unique()
    )

    norm_map: Dict[str, str] = {}
    acro_map: Dict[str, str] = {}
    token_index: List[Tuple[str, set]] = []

    for sch in schools:
        can = alias(sch)
        norm_map[can] = sch
        ac = acronym_from(sch)
        if ac:
            acro_map[ac] = sch
        token_index.append((sch, set(can.split())))

    q_raw = name
    q_norm = alias(q_raw)
    detail["q_norm"] = q_norm

    if q_norm in norm_map:
        return norm_map[q_norm], detail

    q_np = no_paren(q_norm)
    detail["q_no_paren"] = q_np
    if q_np in norm_map:
        return norm_map[q_np], detail

    q_alias = alias(q_np)
    detail["q_alias"] = q_alias
    if q_alias in norm_map:
        return norm_map[q_alias], detail

    q_acro = acronym_from(q_raw)
    if q_acro and q_acro in acro_map:
        return acro_map[q_acro], detail

    q_tokens = set(q_alias.split())
    best_team, best_score = None, 0.0
    for sch, toks in token_index:
        if not toks:
            continue
        inter = len(q_tokens & toks)
        if inter == 0:
            continue
        union = len(q_tokens | toks)
        jacc = inter / float(union) if union else 0.0
        contain = 1.0 if (q_tokens.issubset(toks) or toks.issubset(q_tokens)) else 0.0
        score = jacc + 0.25 * contain
        if score > best_score:
            best_score, best_team = score, sch

    if best_team and best_score >= 0.40:
        detail.update({"best_team": best_team, "best_score": best_score})
        return best_team, detail

    detail.update({"best_team": best_team, "best_score": best_score})
    return None, detail


def _autofix_aliases_from_unmatched(
    unmatched_json_path: str = os.path.join(DATA_DIR, "market_unmatched.json"),
    alias_json_path: str = os.path.join(DATA_DIR, "team_aliases.json"),
    min_score: float = 0.86,
) -> Dict[str, str]:
    """Auto-generate alias entries for FanDuel names that had strong fuzzy matches."""
    try:
        payload = read_json_blob(unmatched_json_path) or {}
        if not payload:
            return {}
        items = payload.get("unmatched", payload) or []

        alias_map: Dict[str, str] = read_json_blob(alias_json_path) or {}
        added: Dict[str, str] = {}

        def _norm_for_alias(s: str) -> str:
            s = (s or "").lower().strip()
            s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
            s = s.replace(" st.", " state").replace(" st ", " state ")
            import re
            s = re.sub(r"[^a-z0-9() ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        for u in items:
            fd_h = u.get("fd_home")
            h_best = u.get("home_fuzzy") or u.get("home_best")
            h_score = float(u.get("home_fuzzy_score") or u.get("home_best_score") or 0.0)
            h_res = u.get("home_resolved")

            fd_a = u.get("fd_away")
            a_best = u.get("away_fuzzy") or u.get("away_best")
            a_score = float(u.get("away_fuzzy_score") or u.get("away_best_score") or 0.0)
            a_res = u.get("away_resolved")

            target_h = h_res or (h_best if h_score >= min_score else None)
            target_a = a_res or (a_best if a_score >= min_score else None)

            if fd_h and target_h:
                k = _norm_for_alias(fd_h)
                v = _norm_for_alias(target_h)
                if k and v and k != v and alias_map.get(k) != v:
                    alias_map[k] = v
                    added[k] = v
            if fd_a and target_a:
                k = _norm_for_alias(fd_a)
                v = _norm_for_alias(target_a)
                if k and v and k != v and alias_map.get(k) != v:
                    alias_map[k] = v
                    added[k] = v

        if added:
            write_json_blob(alias_json_path, alias_map)
            _dbg(f"autofix_aliases_from_unmatched: added {len(added)} alias entries -> {alias_json_path}")
        else:
            _dbg("autofix_aliases_from_unmatched: no strong fuzzy candidates to add")
        return added
    except Exception as e:
        print(f"[warn] autofix aliases failed: {e}")
        return {}


def get_market_lines_fanduel_for_weeks(
    year: int, weeks: List[int], schedule_df: pd.DataFrame, cache: ApiCache
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw = _odds_api_fetch_fanduel(year, weeks, cache)
    raw_count = len(raw)
    _dbg(f"get_market_lines_fanduel_for_weeks: raw games from odds api={raw_count}")
    if not raw:
        stats = {"raw": raw_count, "mapped": 0, "unmatched": 0}
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "spread"]), stats

    # Pre-index schedule
    idx = {}
    for _, row in schedule_df.iterrows():
        a = str(row.get("away_team") or "").strip()
        h = str(row.get("home_team") or "").strip()
        if a and h:
            idx[(a, h)] = {"game_id": row.get("game_id"), "week": row.get("week")}

    # Schedule by date for constrained matching
    sched_by_date: Dict[str, Dict[str, Any]] = {}

    def _clean_local(x: str) -> str:
        s = (x or "").lower().strip()
        s = s.replace("&", " and ").replace("-", " ").replace("/", " ")
        s = s.replace(" st.", " state").replace(" st ", " state ")
        import re
        s = re.sub(r"[^a-z0-9() ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    for _, row in schedule_df.iterrows():
        d = str(row.get("date") or "").strip()
        if not d:
            continue
        h = str(row.get("home_team") or "").strip()
        a = str(row.get("away_team") or "").strip()
        ent = sched_by_date.setdefault(d, {"pairs": set(), "home_set": set(), "away_set": set(), "teams": set()})
        ent["pairs"].add((a, h))
        ent["home_set"].add(h)
        ent["away_set"].add(a)
        ent["teams"].update([h, a])

    out_rows: List[Dict[str, Any]] = []
    unmatched_details: List[Dict[str, Any]] = []

    for g in raw:
        h_raw = g.get("home_name")
        a_raw = g.get("away_name")

        h_name, h_dbg = _resolve_names_to_schedule_with_details(schedule_df, h_raw)
        a_name, a_dbg = _resolve_names_to_schedule_with_details(schedule_df, a_raw)

        if not h_name or not a_name:
            # Constrain to same date and fuzzy within that slate
            cdate = _date_from_iso(g.get("commence_time"))
            same_date = sched_by_date.get(cdate, {}) if cdate else {}
            cand_teams = same_date.get("teams") if same_date else None
            if not cand_teams:
                cand_teams = set(str(x).strip() for x in schedule_df.get("home_team", pd.Series([], dtype=str)).dropna()) | set(
                    str(x).strip() for x in schedule_df.get("away_team", pd.Series([], dtype=str)).dropna()
                )

            def _norm(s: str) -> str:
                return _clean_local(s)

            if not h_name:
                h_best, h_score, h_qn = _best_fuzzy_match(h_raw, cand_teams, _norm)
            else:
                h_best, h_score, h_qn = h_name, 1.0, _norm(h_raw)
            if not a_name:
                a_best, a_score, a_qn = _best_fuzzy_match(a_raw, cand_teams, _norm)
            else:
                a_best, a_score, a_qn = a_name, 1.0, _norm(a_raw)

            pair_ok = False
            if h_best and a_best:
                if cdate and same_date:
                    pair_ok = (a_best, h_best) in same_date.get("pairs", set())
                if not pair_ok:
                    # Allow if both teams appear in same slate teams on same date
                    pair_ok = (h_best in (same_date.get("home_set") or set())) and (a_best in (same_date.get("away_set") or set()))

            # Accept only if the (away, home) pair exists for that slate/date AND both fuzzy scores clear threshold
            accept = pair_ok and (min(float(h_score or 0.0), float(a_score or 0.0)) >= 0.78)
            h_name = h_name or (h_best if accept else None)
            a_name = a_name or (a_best if accept else None)

            unmatched_details.append(
                {
                    "fd_home": h_raw,
                    "fd_away": a_raw,
                    "cdate": cdate,
                    "home_resolved": h_name,
                    "away_resolved": a_name,
                    "home_fuzzy": h_best,
                    "away_fuzzy": a_best,
                    "home_fuzzy_score": h_score,
                    "away_fuzzy_score": a_score,
                    "home_qnorm": h_qn,
                    "away_qnorm": a_qn,
                }
            )

        if not h_name or not a_name:
            continue
        meta = idx.get((a_name, h_name))
        if not meta:
            continue
        out_rows.append(
            {
                "game_id": meta.get("game_id"),
                "week": meta.get("week"),
                "home_team": h_name,
                "away_team": a_name,
                "spread": g.get("point_home_book"),
            }
        )

    df = pd.DataFrame(out_rows)
    stats = {"raw": raw_count, "mapped": len(df), "unmatched": len(unmatched_details)}

    # Also record summary stats to status.json so the UI can reflect mapping health
    try:
        _upsert_status_market_source(
            market_used="fanduel" if len(df) >= 1 else "none",
            market_requested="fanduel",
            fallback_reason=None if len(df) >= 1 else "no_fanduel_matches",
            extra={"market_raw": raw_count, "market_mapped": len(df), "market_unmatched": len(unmatched_details)}
        )
    except Exception:
        pass

    # Write unmatched details for alias improvement workflow
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        write_json_blob(
            os.path.join(DATA_DIR, "market_unmatched.json"),
            {"year": year, "stats": stats, "unmatched": unmatched_details},
        )
    except Exception:
        pass

    # Opportunistically add strong fuzzy matches to team_aliases.json for next run
    try:
        _autofix_aliases_from_unmatched(
            unmatched_json_path=os.path.join(DATA_DIR, "market_unmatched.json"),
            alias_json_path=os.path.join(DATA_DIR, "team_aliases.json"),
            min_score=0.86,
        )
    except Exception:
        pass

    return df, stats


__all__ = [
    "_odds_api_fetch_fanduel",
    "_date_from_iso",
    "_best_fuzzy_match",
    "_resolve_names_to_schedule",
    "_resolve_names_to_schedule_with_details",
    "_autofix_aliases_from_unmatched",
    "get_market_lines_fanduel_for_weeks",
]
