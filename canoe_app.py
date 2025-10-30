# -*- coding: utf-8 -*-
"""
CANOE UI Script (Refactored w/ PSM)
By David Turnbull
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Any, Dict, Iterable, List, Set, Tuple
import flet as ft
import pandas as pd

# ------------------------------
# Paths & Constants
# ------------------------------
SQLITE_FOLDER = "input"
# MASTER_DB = "master.sqlite"
DATASETS_CSV = "input/datasets.csv"
SCHEMA_FILE = "input/schema.sql"
CONFIG_FILE = "input/canoe_config.json"

# New logging config
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "canoe_app.log")

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("canoe_app")
    logger.setLevel(logging.DEBUG)

    # Rotating file handler
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(module)s:%(lineno)d - %(message)s")
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S")
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Avoid duplicate logs if re-imported
    logger.propagate = False
    return logger

logger = setup_logging()
logger.debug("Logging initialized")

ALL_REGIONS = [
    "AB", "BC", "MB", "ON", "QC", "SK", "NB", "NS", "PEI", "NLLAB"
]

# ------------------------------
# Helpers & Matching Logic
# ------------------------------

# Low-level families for CM / GNZ by sector
LOW_PREFIXES: Dict[str, Dict[str, str]] = {
    "Ind": {"CM": "INDCM", "GNZ": "INDNZ"},
    "Res": {"CM": "RESCM", "GNZ": "RESNZ"},
    "Comm": {"CM": "COMCM", "GNZ": "COMNZ"},
    "Tran": {"CM": "TRPCM", "GNZ": "TRPNZ"},
}

# Optional generic fallbacks if region-specific is missing
GENERIC_LOW_BY_SECTOR: Dict[str, Dict[str, str]] = {
    "Ind": {"CM": "INDCM001", "GNZ": "INDNZ001"},
    "Res": {"CM": "RESCM001", "GNZ": "RESNZ001"},
    "Comm": {"CM": "COMCM001", "GNZ": "COMNZ001"},
    "Tran": {"CM": "TRPCM001", "GNZ": "TRPNZ001"},
}

HIGH_PREFIXES = {
    "Ind": "GENINDHR",
    "Res": "RESHR",
    "Comm": "COMHR",
    "Tran": "TRPHR",
}

GENERIC_BY_SECTOR = {
    "Elc": "ELCHR001",
    "Ind": "GENINDHR001",
    "Res": "RESHR001",
    "Comm": "COMHR001",
    "Tran": "TRPHR001",
}

# Canonicalized to NLLAB (was LAB). Include all known aliases in the list.
REGION_ALIASES = {
    "AB": ["AB"],
    "BC": ["BC"],
    "MB": ["MB"],
    "ON": ["ON"],
    "QC": ["QC"],
    "SK": ["SK"],
    "NB": ["NB"],
    "NS": ["NS"],
    "PEI": ["PEI", "PE"],
    "NLLAB": ["NLLAB", "NL", "LAB"],
    "USA": ["USA"],
}
KNOWN_REGION_CODES = {
    "AB","BC","MB","SK","ON","QC","NB","NS","PEI","PE","NLLAB","NL","LAB",
    "CAN","USA","ROW"
}

ALIAS_MAP = {
    "PE": "PEI",
    "NL": "NLLAB",
    "LAB": "NLLAB",
}

def canon_region(code: str) -> str:
    c = (code or "").upper()
    return ALIAS_MAP.get(c, c)

def collect_known_codes_from_db(conn: sqlite3.Connection) -> set[str]:
    """
    Augment KNOWN_REGION_CODES using distinct region values from any table that has a 'region' column,
    and using intertie tech tokens if present.
    """
    cur = conn.cursor()
    codes = set(KNOWN_REGION_CODES)
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
        for t in tables:
            try:
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t});")]
                if "region" in cols:
                    cur.execute(f"SELECT DISTINCT region FROM {t} WHERE region IS NOT NULL;")
                    for (r,) in cur.fetchall():
                        if r:
                            codes.add(canon_region(str(r)))
            except sqlite3.Error:
                pass

        # Try to learn codes from tech strings like EINTABBC / BINTABUSA
        try:
            cur.execute("SELECT DISTINCT tech FROM Technology WHERE tech LIKE 'EINT%' OR tech LIKE 'BINT%';")
            for (tech,) in cur.fetchall():
                rest = tech[4:]
                for k in sorted({*codes}, key=len, reverse=True):
                    if rest.startswith(k):
                        codes.add(canon_region(k))
                        second = rest[len(k):]
                        for k2 in sorted({*codes}, key=len, reverse=True):
                            if second.startswith(k2):
                                codes.add(canon_region(k2))
                                break
                        break
        except sqlite3.Error:
            pass
    except Exception:
        pass
    return {canon_region(c) for c in codes}
ALL_ALIAS_TOKENS = sorted(
    {tok for toks in REGION_ALIASES.values() for tok in toks}, key=len, reverse=True
)
CONFLICT_SUFFIXES = {
    t: [u for u in ALL_ALIAS_TOKENS if u != t and u.endswith(t)] for t in ALL_ALIAS_TOKENS
}
def split_intertie_tech(tech: str, codes: set[str]) -> tuple[str, str, str] | None:
    """
    Accepts EINT/BINT and ELCHREINT/ELCHRBINT.
    Returns (kind, origin, dest) with canonical region codes.
    """
    if not tech:
        return None
    t = tech.upper()

    # Allow ELCHR prefix (electricity wrapper)
    if t.startswith("ELCHR"):
        t = t[5:]

    if not (t.startswith("EINT") or t.startswith("BINT")):
        return None

    kind = t[:4]
    rest = t[4:]

    # longest-match split for variable-length codes (PEI, NLLAB)
    first = None
    for c in sorted(codes, key=len, reverse=True):
        if rest.startswith(c):
            first = c
            break
    if not first:
        return None

    second_raw = rest[len(first):]
    second = None
    for c in sorted(codes, key=len, reverse=True):
        if second_raw.startswith(c):
            second = c
            break
    if not second:
        return None

    return kind, canon_region(first), canon_region(second)

# ------------------------------
# Post-Aggregation Filter / Cleanup
# ------------------------------
def _trailing_region_token(data_id: str) -> str | None:
    """
    Return the trailing region token if the data_id ends with a region alias
    (optionally followed by a 3-digit code), preferring the longest alias first
    (e.g., NLLAB over LAB). Otherwise, None.
    """
    if not data_id:
        return None
    u = data_id.upper()
    # Longest-first to avoid matching AB in NLLAB
    for tok in sorted({t for fam in REGION_ALIASES.values() for t in fam}, key=len, reverse=True):
        if u.endswith(tok) or re.search(rf"{re.escape(tok)}\d{{3}}$", u):
            return tok
    return None


def _is_intertie_data_id(u: str) -> bool:
    u = u.upper()
    return u.startswith("ELCHREINT") or u.startswith("ELCHRBINT") or u.startswith("EINT") or u.startswith("BINT")


def prune_unselected_data_ids(conn: sqlite3.Connection, selected_regions: list[str]) -> None:
    """
    Remove rows from ANY table that has a 'data_id' column if the data_id encodes
    an unselected region. Interties are handled directionally (origin/dest rules).
    Must be run BEFORE pinning.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
    except sqlite3.Error:
        return

    # Build allowed alias token set from selected regions
    allowed_tokens: set[str] = set()
    for r in (selected_regions or []):
        for tok in REGION_ALIASES.get(canon_region(r), [canon_region(r)]):
            allowed_tokens.add(tok)

    # Codes for intertie parsing
    codes = collect_known_codes_from_db(conn)

    cur.execute("PRAGMA foreign_keys = OFF;")

    for t in tables:
        try:
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t});")]
            if "data_id" not in cols:
                continue

            # Collect candidate data_ids in this table
            cur.execute(f"SELECT DISTINCT data_id FROM {t} WHERE data_id IS NOT NULL;")
            ids = [row[0] for row in cur.fetchall() if row and row[0]]

            drop_ids: list[str] = []
            for did in ids:
                u = did.upper()

                # Intertie: decide by origin/dest
                if _is_intertie_data_id(u):
                    parsed = split_intertie_tech(u, codes)
                    if not parsed:
                        # If we can't parse, be conservative: drop if it clearly ends in a non-allowed region token
                        tok = _trailing_region_token(u)
                        if tok and tok not in allowed_tokens:
                            drop_ids.append(did)
                        continue

                    kind, origin, dest = parsed
                    origin, dest = canon_region(origin), canon_region(dest)

                    if kind == "EINT":
                        # keep only if BOTH selected
                        if not (origin in map(canon_region, selected_regions) and dest in map(canon_region, selected_regions)):
                            drop_ids.append(did)
                    else:  # BINT
                        # keep only if ORIGIN selected
                        if origin not in map(canon_region, selected_regions):
                            drop_ids.append(did)
                    continue

                # Non-intertie: check trailing region token; keep generics (no token)
                tok = _trailing_region_token(u)
                if tok is None:
                    # generic/no region suffix — keep
                    continue
                # keep only if token belongs to allowed regions
                if tok not in allowed_tokens:
                    drop_ids.append(did)

            if drop_ids:
                placeholders = ",".join("?" for _ in drop_ids)
                try:
                    cur.execute(f"DELETE FROM {t} WHERE data_id IN ({placeholders});", drop_ids)
                except sqlite3.Error:
                    pass

        except sqlite3.Error:
            continue

    conn.commit()
    try:
        cur.execute("PRAGMA foreign_keys = ON;")
    except sqlite3.Error:
        pass

def prune_unrequested_interties(conn: sqlite3.Connection, selected_regions: list[str]) -> None:
    """
    Remove intertie technologies (and their rows in related tables) that *originate* in
    unselected regions.
      - EINT/ELCHREINT: keep only if BOTH origin and dest are selected.
      - BINT/ELCHRBINT: keep only if ORIGIN is selected (dest can be unselected).
    """
    sel = {canon_region(r) for r in (selected_regions or [])}
    cur = conn.cursor()

    codes = collect_known_codes_from_db(conn)

    # Include ELCHR* as well as plain EINT/BINT
    try:
        cur.execute("""
            SELECT DISTINCT tech FROM Technology
            WHERE tech LIKE 'EINT%' OR tech LIKE 'BINT%'
               OR tech LIKE 'ELCHREINT%' OR tech LIKE 'ELCHRBINT%';
        """)
        techs = [t[0] for t in cur.fetchall()]
    except sqlite3.Error:
        techs = []

    drop_set: set[str] = set()

    for t in techs:
        parsed = split_intertie_tech(t, codes)
        if not parsed:
            continue
        kind, origin, dest = parsed

        if kind == "EINT":
            if not (origin in sel and dest in sel):
                drop_set.add(t)
        else:  # BINT
            if origin not in sel:
                drop_set.add(t)

    if not drop_set:
        return

    # Delete across all tables with a 'tech' column
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
    except sqlite3.Error:
        tables = []

    tech_tables = []
    for t in tables:
        try:
            cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t});")]
            if "tech" in cols:
                tech_tables.append(t)
        except sqlite3.Error:
            pass

    cur.execute("PRAGMA foreign_keys = OFF;")
    placeholders = ",".join("?" for _ in drop_set)
    params = list(drop_set)

    for t in tech_tables:
        try:
            cur.execute(f"DELETE FROM {t} WHERE tech IN ({placeholders});", params)
        except sqlite3.Error:
            pass

    try:
        cur.execute(f"DELETE FROM Technology WHERE tech IN ({placeholders});", params)
    except sqlite3.Error:
        pass

    conn.commit()


def filter_func(output_db: str, pinned_techs: Set[str] | None = None) -> None:
    """
    Cleanup with safeguards:
      - Caps each loop to 20 iterations
      - Breaks if a pass makes no progress
      - Uses CAST(vintage AS INTEGER) for type-safe deletes
    """
    pinned_techs = set(pinned_techs or [])
    MAX_ITERS_PASS1 = 20
    MAX_ITERS_PASS2 = 20

    try:
        conn = sqlite3.connect(output_db)
        curs = conn.cursor()
        try:
            curs.execute("PRAGMA foreign_keys = OFF;")
            curs.execute("PRAGMA temp_store = MEMORY;")
            curs.execute("PRAGMA synchronous = OFF;")
            curs.execute("PRAGMA journal_mode = MEMORY;")
        except sqlite3.Error:
            pass

        # ---- Pass 1: orphan region-tech pruning ----
        for it in range(1, MAX_ITERS_PASS1 + 1):
            bad_rt = curs.execute(
                """
                SELECT DISTINCT region, tech
                FROM Efficiency 
                WHERE output_comm NOT IN (SELECT name FROM Commodity WHERE flag = 'd')
                  AND (region, output_comm) NOT IN (SELECT region, input_comm FROM Efficiency)
                """
            ).fetchall()
            bad_rt = [rt for rt in bad_rt if rt[1] not in pinned_techs]
            if not bad_rt:
                break

            deleted_total = 0
            tables = [t[0] for t in curs.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            for table in tables:
                cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
                if 'region' in cols and 'tech' in cols:
                    for region, tech in bad_rt:
                        try:
                            curs.execute(f"DELETE FROM {table} WHERE region = ? AND tech = ?", (region, tech))
                            if curs.rowcount and curs.rowcount > 0:
                                deleted_total += curs.rowcount
                        except sqlite3.Error:
                            pass

            tech_remaining = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Efficiency').fetchall()}
            tech_before    = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Technology').fetchall()}
            tech_gone = (tech_before - tech_remaining) - pinned_techs
            if tech_gone:
                for table in tables:
                    cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
                    if 'tech' in cols:
                        for tech in tech_gone:
                            try:
                                curs.execute(f"DELETE FROM {table} WHERE tech = ?", (tech,))
                                if curs.rowcount and curs.rowcount > 0:
                                    deleted_total += curs.rowcount
                            except sqlite3.Error:
                                pass

            conn.commit()
            if deleted_total == 0:
                logger.warning("Pass1 made no progress in iter %d; stopping to avoid endless loop.", it)
                break

        # ---- Pass 2: timing pruning ----
        for it in range(1, MAX_ITERS_PASS2 + 1):
            time_all = [p[0] for p in curs.execute('SELECT period FROM TimePeriod').fetchall()]
            if not time_all:
                break
            time_all_int = []
            for x in time_all:
                try: time_all_int.append(int(x))
                except Exception: pass
            if not time_all_int:
                break

            lifetime_process: Dict[Tuple[str,str,int], int] = {}
            DEFAULT_LT = 40

            for r, t, v in curs.execute('SELECT region, tech, vintage FROM Efficiency').fetchall():
                try: vi = int(v)
                except Exception: continue
                lifetime_process[(r, t, vi)] = DEFAULT_LT

            for r, t, lt in curs.execute('SELECT region, tech, lifetime FROM LifetimeTech').fetchall():
                try: lti = int(lt)
                except Exception: continue
                for v in time_all_int:
                    lifetime_process[(r, t, v)] = lti

            for r, t, v, lp in curs.execute('SELECT region, tech, vintage, lifetime FROM LifetimeProcess').fetchall():
                try: vi = int(v); lpi = int(lp)
                except Exception: continue
                lifetime_process[(r, t, vi)] = lpi

            df_eff = pd.read_sql_query('SELECT * FROM Efficiency', conn)
            if df_eff.empty:
                break
            df_eff['vintage'] = pd.to_numeric(df_eff['vintage'], errors='coerce').fillna(0).astype(int)

            def snap5(y: int) -> int:
                try: y = int(y)
                except Exception: return 0
                return min(2050, (y // 5) * 5)

            df_eff['last_out'] = [
                snap5(v + int(lifetime_process.get((r, t, v), DEFAULT_LT)))
                for r, t, v in df_eff[['region','tech','vintage']].itertuples(index=False, name=None)
            ]
            demand_comms = {c[0] for c in curs.execute("SELECT name FROM Commodity WHERE flag = 'd'").fetchall()}
            df_nd = df_eff.loc[~df_eff['output_comm'].isin(demand_comms)].copy()
            if df_nd.empty:
                break

            df_last_in = (
                df_eff.groupby(['region','input_comm'], as_index=True)['last_out']
                      .max()
                      .rename('last_in')
            )
            df_nd = df_nd.merge(df_last_in, left_on=['region','output_comm'], right_index=True, how='left')
            df_nd['last_in'] = pd.to_numeric(df_nd['last_in'], errors='coerce').fillna(0).astype(int)

            df_remove = df_nd.loc[df_nd['last_in'] < df_nd['last_out']].copy()
            if pinned_techs:
                df_remove = df_remove[~df_remove['tech'].isin(pinned_techs)]
            if df_remove.empty:
                break

            deleted_total = 0
            for region, input_comm, tech, vintage, output_comm in df_remove[['region','input_comm','tech','vintage','output_comm']].itertuples(index=False, name=None):
                try:
                    curs.execute(
                        """
                        DELETE FROM Efficiency 
                        WHERE region = ? AND input_comm = ? AND tech = ?
                          AND CAST(vintage AS INTEGER) = ?
                          AND output_comm = ?
                        """, (region, input_comm, tech, int(vintage), output_comm)
                    )
                    if curs.rowcount and curs.rowcount > 0: deleted_total += curs.rowcount
                    for tbl in ("CostVariable", "CostFixed", "EmissionActivity"):
                        try:
                            curs.execute(
                                f"DELETE FROM {tbl} WHERE region = ? AND tech = ? AND CAST(vintage AS INTEGER) = ?",
                                (region, tech, int(vintage))
                            )
                            if curs.rowcount and curs.rowcount > 0: deleted_total += curs.rowcount
                        except sqlite3.Error:
                            pass
                except sqlite3.Error:
                    pass

            conn.commit()
            if deleted_total == 0:
                logger.warning("Pass2 made no progress in iter %d; stopping to avoid endless loop.", it)
                break

        try: curs.execute("PRAGMA foreign_keys = ON;")
        except sqlite3.Error: pass

        conn.commit(); conn.close()
    except Exception as e:
        logger.exception("filter_func failed for %s: %s", output_db, e)



# ------------------------------
# ID Matching helpers
# ------------------------------

def region_tokens(region: str) -> List[str]:
    """Return all alias tokens for a canonical region (or for a token)."""
    toks = REGION_ALIASES.get(region)
    if toks:
        return toks
    # If we were passed an alias token (e.g., 'NL'), find its family.
    for canon, family in REGION_ALIASES.items():
        if region == canon or region in family:
            return family
    return [region]


def _ends_with_region_token_exact(_id: str, tok: str) -> bool:
    if not _id.endswith(tok) and not re.search(rf"{re.escape(tok)}\d{{3}}$", _id):
        return False
    for longer in CONFLICT_SUFFIXES.get(tok, []):
        if re.search(rf"{re.escape(longer)}\d{{3}}$", _id):
            return False
    return True


def csv_match_prefix_region(csv_ids: Iterable[str], prefix: str, region: str) -> Set[str]:
    toks = region_tokens(region)
    out: Set[str] = set()
    for _id in csv_ids:
        if not _id.startswith(prefix):
            continue
        if not re.search(r"\d{3}$", _id):
            continue
        for tok in toks:
            if _ends_with_region_token_exact(_id, tok):
                out.add(_id)
                break
    return out


def _intertie_middle_two_regions(_id: str, prefix: str) -> str | None:
    m = re.match(rf"^{re.escape(prefix)}([A-Z]+)\d{{3}}$", _id)
    return m.group(1) if m else None


def csv_match_intertie(csv_ids: Iterable[str], prefix: str, r1: str, r2: str) -> Set[str]:
    r1t = region_tokens(r1)
    r2t = region_tokens(r2)
    out: Set[str] = set()
    for _id in csv_ids:
        if not _id.startswith(prefix):
            continue
        mid = _intertie_middle_two_regions(_id, prefix)
        if not mid:
            continue
        if any(mid == (a + b) or mid == (b + a) for a in r1t for b in r2t):
            out.add(_id)
    return out

# --- Robust tweak: tolerate both ELCHREINT/ELCHRBINT and plain EINT/BINT ----
def csv_match_intertie_any(csv_ids: Iterable[str], prefixes: Iterable[str], r1: str, r2: str) -> Set[str]:
    out: Set[str] = set()
    for p in prefixes:
        out |= csv_match_intertie(csv_ids, p, r1, r2)
    return out


# ------------------------------
# CSV Loader
# ------------------------------

def collect_db_data_ids(db_path: str) -> Set[str]:
    """
    Scan the input SQLite and return all distinct data_id values that exist
    (across every table that has a 'data_id' column).
    """
    ids: Set[str] = set()
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
        for t in tables:
            try:
                cols = [c[1] for c in cur.execute(f'PRAGMA table_info({t});')]
                if 'data_id' in cols:
                    cur.execute(f"SELECT DISTINCT data_id FROM {t}")
                    ids.update(r[0] for r in cur.fetchall() if r and r[0] is not None)
            except sqlite3.Error:
                continue
        conn.close()
    except Exception as e:
        logger.exception("collect_db_data_ids failed for %s: %s", db_path, e)
    return ids


def get_data_ids_from_csv(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        logger.warning("datasets.csv not found at %s", file_path)
        return []
    # Try pandas first
    try:
        df = pd.read_csv(file_path)
        if 'data_id' in df.columns:
            return [str(x).strip() for x in df['data_id'].dropna().astype(str)]
        if df.shape[1] == 1:
            col = df.columns[0]
            return [str(x).strip() for x in df[col].dropna().astype(str)]
    except Exception as e:
        logger.exception("Failed to read datasets CSV with pandas: %s", e)
    # Fallback CSV reader
    out: List[str] = []
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            first = True
            for row in reader:
                if not row:
                    continue
                val = row[0].strip()
                if first and val.lower() in ("data_id", "id"):
                    first = False
                    continue
                first = False
                if val:
                    out.append(val)
    except Exception as e:
        logger.exception("Fallback CSV reader failed for %s: %s", file_path, e)
    return out

# New: lightweight config loader (returns dict)
def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
                logger.debug("Loaded config from %s", CONFIG_FILE)
                return cfg
    except Exception as e:
        logger.exception("Failed to load config %s: %s", CONFIG_FILE, e)
    return {}

# ------------------------------
# Desired IDs Builder (CCS removed)
# ------------------------------



def build_desired_ids_from_matrix(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    csv_ids: Set[str],
    global_settings: Dict[str, Any],
    get_current_regions,
) -> Set[str]:
    """
    Use only ids that actually exist in the current DB (csv_ids).
    Interties are *directional*: endogenous keeps both ways iff both regions selected;
    boundary keeps only origin->dest when origin is selected.
    """
    # --- local helpers to avoid relying on globals ---
    ALIAS_MAP = {"PE": "PEI", "NL": "NLLAB", "LAB": "NLLAB"}
    def canon_region(code: str) -> str:
        return ALIAS_MAP.get((code or "").upper(), (code or "").upper())
    def region_alias_tokens(code: str) -> Set[str]:
        c = canon_region(code)
        if c == "PEI": return {"PEI", "PE"}
        if c == "NLLAB": return {"NLLAB", "NL", "LAB"}
        return {c}
    def csv_match_intertie_dir(ids: Iterable[str], prefix: str, origin: str, dest: str) -> Set[str]:
        ot, dt = region_alias_tokens(origin), region_alias_tokens(dest)
        out: Set[str] = set()
        for _id in ids:
            u = _id.upper()
            if not u.startswith(prefix): 
                continue
            for o in ot:
                for d in dt:
                    if u.startswith(prefix + o + d):
                        out.add(_id); break
                else:
                    continue
                break
        return out
    def csv_match_intertie_dir_any(ids: Iterable[str], prefixes: Iterable[str], origin: str, dest: str) -> Set[str]:
        out: Set[str] = set()
        for p in prefixes:
            out |= csv_match_intertie_dir(ids, p, origin, dest)
        return out

    # read global flags
    scenario = global_settings.get("scenario", "Current Measure")
    is_psm  = bool(global_settings.get("power_system_model", False))

    # which regions are actually active in the UI (row not all NA)
    try:
        current_regions = list(get_current_regions()) or []
    except Exception:
        current_regions = []

    selected_regions: Set[str] = set()
    for r in current_regions:
        row_active = any(
            (matrix.get((r, s)) and str(matrix[(r, s)].value).upper() != "NA")
            for s in {"Ind","Res","Comm","Elc","Tran"}
        )
        if row_active:
            selected_regions.add(r)

    desired: Set[str] = set()

    # electricity (High only; drop DEM/EINT/BINT here; they’re added explicitly below)
    for r in current_regions:
        dd = matrix.get((r, "Elc"))
        if dd and str(dd.value).strip().upper() == "HIGH":
            base = {x for x in csv_ids if x.startswith("ELCHR")}
            base -= {x for x in base if x.startswith("ELCHRDEM")}
            base -= {x for x in base if x.startswith("ELCHREINT")}
            base -= {x for x in base if x.startswith("ELCHRBINT")}
            desired |= {x for x in base if any(x.endswith(tok) or x.endswith(tok + "001") for tok in region_alias_tokens(r))}

    # non-electric sectors
    LOW_PREFIXES = {"Ind":{"CM":"INDCM","GNZ":"INDNZ"}, "Res":{"CM":"RESCM","GNZ":"RESNZ"},
                    "Comm":{"CM":"COMCM","GNZ":"COMNZ"}, "Tran":{"CM":"TRPCM","GNZ":"TRPNZ"}}
    HIGH_PREFIXES = {"Ind":"GENINDHR", "Res":"RESHR", "Comm":"COMHR", "Tran":"TRPHR"}
    GENERIC_LOW_BY_SECTOR = {"Ind":{"CM":"INDCM001","GNZ":"INDNZ001"},
                             "Res":{"CM":"RESCM001","GNZ":"RESNZ001"},
                             "Comm":{"CM":"COMCM001","GNZ":"COMNZ001"},
                             "Tran":{"CM":"TRPCM001","GNZ":"TRPNZ001"}}

    low_cm_seen  = {k: False for k in LOW_PREFIXES}
    low_gnz_seen = {k: False for k in LOW_PREFIXES}

    def match_prefix_region(ids: Iterable[str], prefix: str, r: str) -> Set[str]:
        toks = region_alias_tokens(r)
        out: Set[str] = set()
        for _id in ids:
            if not _id.startswith(prefix): 
                continue
            if not re.search(r"\d{3}$", _id): 
                continue
            if any(_id.endswith(tok) or re.search(rf"{re.escape(tok)}\d{{3}}$", _id) for tok in toks):
                out.add(_id)
        return out

    for r in current_regions:
        for sector, hr in HIGH_PREFIXES.items():
            dd = matrix.get((r, sector))
            if not dd: 
                continue
            val = (str(dd.value) or "").strip().upper()
            if val == "HIGH":
                desired |= match_prefix_region(csv_ids, hr, r)
            elif val == "LOW":
                if scenario == "Current Measure":
                    low_cm_seen[sector] = True
                    desired |= match_prefix_region(csv_ids, LOW_PREFIXES[sector]["CM"], r)
                elif scenario == "Global Net Zero":
                    low_gnz_seen[sector] = True
                    desired |= match_prefix_region(csv_ids, LOW_PREFIXES[sector]["GNZ"], r)

    # generic low-levels if any LOW was seen for that sector
    for sec in LOW_PREFIXES:
        if low_cm_seen[sec]:
            cm = GENERIC_LOW_BY_SECTOR[sec]["CM"]
            if cm in csv_ids: desired.add(cm)
        if low_gnz_seen[sec]:
            nz = GENERIC_LOW_BY_SECTOR[sec]["GNZ"]
            if nz in csv_ids: desired.add(nz)

    # DEM (power-system demand) only in PSM mode
    if is_psm:
        for r in selected_regions:
            desired |= match_prefix_region(csv_ids, "ELCHRDEM", r)

    # interties (directional)
    intertie_pairs = [
        ("AB", "BC"), ("AB", "SK"), ("SK", "MB"),
        ("MB", "ON"), ("ON", "QC"),
        ("NB", "NS"), ("NB", "QC"), ("NB", "PEI"),
        ("NS", "PEI"), ("NLLAB", "NS"), ("NLLAB", "QC"),
        ("BC", "USA"), ("AB", "USA"), ("SK", "USA"),
        ("MB", "USA"), ("ON", "USA"), ("QC", "USA"), ("NB", "USA"),
    ]
    def sel(code: str) -> bool:
        toks = region_alias_tokens(code)
        # selected if any alias token appears in a selected region’s aliases
        return any(t in {a for r in selected_regions for a in region_alias_tokens(r)} for t in toks)

    for a, b in intertie_pairs:
        sel_a, sel_b = sel(a), sel(b)
        if sel_a and sel_b:
            desired |= csv_match_intertie_dir_any(csv_ids, ("ELCHREINT", "EINT"), a, b)
            desired |= csv_match_intertie_dir_any(csv_ids, ("ELCHREINT", "EINT"), b, a)
        elif sel_a ^ sel_b:
            if sel_a:
                desired |= csv_match_intertie_dir_any(csv_ids, ("ELCHRBINT", "BINT"), a, b)
            else:
                desired |= csv_match_intertie_dir_any(csv_ids, ("ELCHRBINT", "BINT"), b, a)

    # AGRI/FUEL (skip AGRI when in power system mode)
    if not is_psm:
        desired |= {x for x in csv_ids if x == "AGRIHR001"}
        for r in selected_regions:
            desired |= match_prefix_region(csv_ids, "AGRIHR", r)
    for r in selected_regions:
        desired |= match_prefix_region(csv_ids, "FUELHR", r)
    if "FUELHR001" in csv_ids:
        desired.add("FUELHR001")

    # generics by sector (Elc generic only when relevant)
    GENERIC_BY_SECTOR = {"Elc":"ELCHR001", "Ind":"GENINDHR001", "Res":"RESHR001", "Comm":"COMHR001", "Tran":"TRPHR001"}
    def maybe_add_generic(sec: str):
        g = GENERIC_BY_SECTOR.get(sec)
        if g and g in csv_ids: desired.add(g)

    if is_psm:
        maybe_add_generic("Elc")
    else:
        if any((matrix.get((r, "Elc")) and str(matrix[(r,"Elc")].value).upper() != "NA") for r in current_regions):
            maybe_add_generic("Elc")
    for sec in ("Ind","Res","Comm","Tran"):
        if any((matrix.get((r, sec)) and str(matrix[(r,sec)].value).upper() != "NA") for r in current_regions):
            maybe_add_generic(sec)

    if is_psm:
        desired = {x for x in desired if not x.startswith("AGRIHR")}
    return desired



# ------------------------------
# Demand-led cleanup (region-aware)
# ------------------------------

def get_demand_lists_region_aware(
    output_db: str, 
    conn: sqlite3.Connection,
    *,
    power_system_model: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Identifies all related commodities and technologies starting from the 'Demand' table.
    RE-ADDED: In PSM mode, exclude R_ethos.
    """
    cursor = conn.cursor()

    # RE-ADDED: Logic for R_ethos
    remove: Set[str] = {'R_ethos'} if power_system_model else set()
    ethos_whitelist = ('E_ethos', 'F_ethos') if power_system_model else ('E_ethos', 'F_ethos', 'R_ethos')

    commodities: Set[str] = set()
    technologies: Set[str] = set()

    try:
        # Seed from Demand via Efficiency
        cursor.execute(
            """
            SELECT DISTINCT output_comm
            FROM Efficiency
            WHERE output_comm IN (SELECT commodity FROM Demand)
            """
        )
        commodities.update([row[0] for row in cursor.fetchall()])

        cursor.execute(
            """
            SELECT DISTINCT tech
            FROM Efficiency
            WHERE output_comm IN (SELECT commodity FROM Demand)
            """
        )
        technologies.update([row[0] for row in cursor.fetchall()])

        new_commodities = set(commodities)

        while new_commodities:
            id_str = "('" + "', '".join(new_commodities) + "')"

            cursor.execute(f"SELECT DISTINCT tech FROM Efficiency WHERE output_comm IN {id_str}")
            technologies.update({row[0] for row in cursor.fetchall()})

            cursor.execute(f"SELECT DISTINCT input_comm FROM Efficiency WHERE output_comm IN {id_str}")
            temp_commodities = {row[0] for row in cursor.fetchall()}

            temp_commodities -= remove

            newly_found_commodities = temp_commodities - commodities
            commodities.update(newly_found_commodities)
            new_commodities = newly_found_commodities

        # Always include emission commodities
        cursor.execute("SELECT DISTINCT emis_comm FROM EmissionActivity")
        commodities.update({row[0] for row in cursor.fetchall()})

        # Ensure ETHOS items present according to whitelist
        placeholders = ",".join("?" for _ in ethos_whitelist)
        cursor.execute(f"SELECT name FROM Commodity WHERE name IN ({placeholders})", ethos_whitelist)
        commodities.update({row[0] for row in cursor.fetchall()})

        # RE-ADDED: R_ethos removal
        if power_system_model and 'R_ethos' in commodities:
            commodities.discard('R_ethos')

    except sqlite3.Error as e:
        logger.exception("DB traversal failed for %s (power_system_model=%s): %s", output_db, power_system_model, e)
        return [], []

    com_list = sorted(commodities)
    tech_list = sorted(technologies)

    return com_list, tech_list


def infer_selected_regions_from_matrix(matrix: Dict[Tuple[str, str], ft.Dropdown]) -> Set[str]:
    """Regions that have any sector set to something other than 'NA'."""
    regions = {r for (r, _s) in matrix.keys()}
    picked: Set[str] = set()
    for r in regions:
        any_on = any(
            (dd := matrix.get((r, s))) is not None and str(dd.value).upper() != "NA"
            for s in ("Ind", "Res", "Comm", "Elc", "Tran")
        )
        if any_on:
            picked.add(r)
    return picked


def table_has_region_column(cursor: sqlite3.Cursor, table: str) -> bool:
    cursor.execute(f"PRAGMA table_info({table});")
    cols = [row[1] for row in cursor.fetchall()]
    return "region" in cols


def delete_rows_not_in_regions(
    conn: sqlite3.Connection,
    tables: List[str],
    allowed_regions: List[str]
) -> None:
    """
    Alias-aware region pruning, but NEVER remove rows tied to pinned ids/techs.
    Keeps any row whose `region` matches any alias token for the allowed regions.
    Pinned guards:
      - if table has data_id: keep rows whose data_id is in PinnedDataIDs
      - if table has tech:    keep rows whose tech    is in PinnedTechs
    """
    if not allowed_regions:
        logger.warning("No allowed regions specified; skipping region pruning.")
        return

    # Build alias-aware token set
    allowed_tokens: Set[str] = set()
    for r in allowed_regions:
        allowed_tokens.update(region_tokens(r))
        allowed_tokens.add(r)

    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = OFF;")
    placeholders = ",".join(["?"] * len(allowed_tokens))
    params = list(allowed_tokens)

    for table in tables:
        try:
            cur.execute(f"PRAGMA table_info({table});")
            cols = [row[1] for row in cur.fetchall()]
            if "region" not in cols:
                continue

            has_data_id = "data_id" in cols
            has_tech = "tech" in cols

            guard_sql = ""
            if has_data_id and has_tech:
                guard_sql = " AND data_id NOT IN (SELECT data_id FROM PinnedDataIDs) AND tech NOT IN (SELECT tech FROM PinnedTechs)"
            elif has_data_id:
                guard_sql = " AND data_id NOT IN (SELECT data_id FROM PinnedDataIDs)"
            elif has_tech:
                guard_sql = " AND tech NOT IN (SELECT tech FROM PinnedTechs)"

            sql = f"DELETE FROM {table} WHERE region NOT IN ({placeholders}){guard_sql};"
            cur.execute(sql, params)
            logger.debug("Region-pruned table: %s (kept tokens: %s)", table, sorted(allowed_tokens))
        except sqlite3.Error as e:
            logger.exception("Skipped region filter for %s: %s", table, e)
            return

    conn.commit()
    cur.execute("PRAGMA foreign_keys = ON;")



# ------------------------------
# SQLite Aggregation
# ------------------------------

def aggregate_sqlite_files(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    csv_ids: Set[str],  # ignored; we gather ids from DB
    global_settings: Dict[str, Any],
    get_current_regions,
    output_filename: str,
    input_filename: str,
) -> None:
    try:
        # 1) discover ids that actually exist
        available_ids = collect_db_data_ids(input_filename)
        if not available_ids:
            raise RuntimeError("No data_id values found in the input database.")

        # 2) choose desired ids from the UI (directional interties), then restrict to available
        desired_ids = build_desired_ids_from_matrix(
            matrix=matrix,
            csv_ids=available_ids,
            global_settings=global_settings,
            get_current_regions=get_current_regions,
        ) or set()
        selected_data_ids = sorted(desired_ids & available_ids)

        # 3) create schema and copy tables (data_id-filtered where present)
        if os.path.exists(output_filename):
            os.remove(output_filename)
        out_conn = sqlite3.connect(output_filename)
        out_cur  = out_conn.cursor()
        with open(SCHEMA_FILE, "r", encoding="utf-8") as fh:
            out_cur.executescript(fh.read())
        out_conn.commit()

        in_conn = sqlite3.connect(input_filename)
        in_cur  = in_conn.cursor()
        in_cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {t[0] for t in in_cur.fetchall()}
        tables -= {
            'CommodityType','Operator','TechnologyType','TimePeriodType',
            'DataQualityCredibility','DataQualityDataQualityGeography',
            'DataQualityStructure','DataQualityTechnology','DataQualityTime',
            'TechnologyLabel','CommodityLabel','DataSourceLabel'
        }
        out_cur.execute("PRAGMA foreign_keys = OFF;")
        id_sql = "('" + "', '".join(selected_data_ids) + "')" if selected_data_ids else None
        for t in tables:
            cols = [c[1] for c in in_cur.execute(f"PRAGMA table_info({t});")]
            if 'data_id' in cols and id_sql:
                in_cur.execute(f"SELECT * FROM {t} WHERE data_id IN {id_sql};")
            else:
                in_cur.execute(f"SELECT * FROM {t};")
            rows = in_cur.fetchall()
            if rows:
                placeholders = ", ".join(["?"] * len(in_cur.description))
                out_cur.executemany(f"INSERT OR IGNORE INTO {t} VALUES ({placeholders});", rows)
            out_conn.commit()

        # 4) prune wrong-direction interties BEFORE anything else
        try:
            selected_regions = list(get_current_regions()) if callable(get_current_regions) else []
        except Exception:
            selected_regions = []
        prune_unrequested_interties(out_conn, selected_regions)

        # 4.5) NEW: data_id-aware region prune for tables without a 'region' column
        prune_unselected_data_ids(out_conn, selected_regions)

        # 5) REGION PRUNE *BEFORE* pinning and demand-pruning
        delete_rows_not_in_regions(out_conn, list(tables), selected_regions)
        out_conn.commit()


        # 6) create pins AFTER region prune
        out_cur.executescript("""
        CREATE TABLE IF NOT EXISTS PinnedDataIDs(data_id TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS PinnedTechs(tech TEXT PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS PinnedComms(name TEXT PRIMARY KEY);
        """)
        if selected_data_ids:
            out_cur.executemany("INSERT OR IGNORE INTO PinnedDataIDs(data_id) VALUES (?)", [(i,) for i in selected_data_ids])
        out_cur.execute("""
            INSERT OR IGNORE INTO PinnedTechs(tech)
            SELECT DISTINCT tech FROM Technology WHERE data_id IN (SELECT data_id FROM PinnedDataIDs)
        """)
        out_cur.execute("""
            INSERT OR IGNORE INTO PinnedComms(name)
            SELECT DISTINCT input_comm  FROM Efficiency WHERE tech IN (SELECT tech FROM PinnedTechs)
            UNION
            SELECT DISTINCT output_comm FROM Efficiency WHERE tech IN (SELECT tech FROM PinnedTechs)
            UNION
            SELECT DISTINCT emis_comm   FROM EmissionActivity WHERE tech IN (SELECT tech FROM PinnedTechs)
        """)
        out_conn.commit()

        # 7) demand-led pruning (now working on AB/BC-scoped rows only; respect pins)
        com_list, tech_list = get_demand_lists_region_aware(
            output_filename, out_conn, power_system_model=global_settings.get("power_system_model", False)
        )
        if tech_list:
            tech_str = "('" + "', '".join(tech_list) + "')"
            for tbl in ("Efficiency","Technology","LifetimeTech","CostVariable","EmissionActivity"):
                out_cur.execute(
                    f"DELETE FROM {tbl} WHERE tech NOT IN {tech_str} AND tech NOT IN (SELECT tech FROM PinnedTechs)"
                )
        if com_list:
            com_str = "('" + "', '".join(com_list) + "')"
            out_cur.execute(
                f"DELETE FROM Commodity WHERE name NOT IN {com_str} AND name NOT IN (SELECT name FROM PinnedComms)"
            )
        out_conn.commit()

        # 8) final cleanup honoring pinned techs
        out_cur.execute("SELECT tech FROM PinnedTechs")
        pinned_techs = {r[0] for r in out_cur.fetchall()}
        in_conn.close(); out_conn.close()
        filter_func(output_filename, pinned_techs)

        logger.info("Aggregation complete. Output: %s", output_filename)
    except Exception as e:
        logger.exception("Unhandled error during aggregation (input=%s output=%s): %s", input_filename, output_filename, e)
        raise





# ------------------------------
# Flet UI (layout preserved; CCS removed)
# ------------------------------

def main(page: ft.Page) -> None:
    page.title = "CANOE UI"
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1200
    page.window_height = 700

    # Sectors and options
    sectors = ["Ind", "Res", "Comm", "Elc", "Tran"]
    # All sectors use the same levels
    levels_all_sectors = ["Low", "High", "NA"]
    
    # UI storage
    matrix: Dict[Tuple[str, str], ft.Dropdown] = {}

    # Global flags
    global_settings = {
        "scenario": "Current Measure",    # Default scenario
        "power_system_model": False # Default PSM
    }

    # Widgets
    status_text = ft.Text("")
    in_filename_text_field = ft.TextField(label="Input file path (e.g. dataset.sqlite)", value="", width=260)
    out_filename_text_field = ft.TextField(label="Output file path (e.g. canoe.sqlite)", value="", width=260)
    
    # Scenario Dropdown
    scenario_dropdown = ft.Dropdown(
        label="Low-level Scenario",
        options=[
            ft.dropdown.Option("Current Measure"),
            ft.dropdown.Option("Global Net Zero"),
        ],
        value="Current Measure", # Default
        width=200,
    )
    
    # RE-ADDED: Power System Checkbox
    power_system_checkbox = ft.Checkbox(label="Power system model", value=False)
    
    image = ft.Container(content=ft.Image(src="./assets/logo.png", height=50, width=60), alignment=ft.alignment.top_right)

    # load saved config (will be applied once UI elements are created)
    saved_cfg = load_config()

    def save_config() -> None:
        """Persist current UI state to CONFIG_FILE (JSON)."""
        try:
            cfg = {
                "input_filename": (in_filename_text_field.value or "").strip(),
                "output_filename": (out_filename_text_field.value or "").strip(),
                "scenario": (scenario_dropdown.value or "Current Measure"),
                "power_system_model": bool(power_system_checkbox.value), # RE-ADDED
                "matrix": { f"{r}|{s}": (matrix[(r, s)].value if (r, s) in matrix and matrix[(r, s)] is not None else None)
                            for (r, s) in matrix.keys() },
            }
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, ensure_ascii=False)
            logger.debug("Saved config to %s", CONFIG_FILE)
        except Exception as e:
            logger.exception("Failed to save config %s: %s", CONFIG_FILE, e)
            pass



    # --- UI builders / updaters ---

    def on_sector_dropdown_change(e: ft.ControlEvent) -> None:
        """Auto-fill sensible defaults when user selects a sector (normal mode)."""
        current_region = e.control.data
        is_na_selected = (e.control.value == "NA")

        # RE-ADDED: Do not auto-fill if in power system mode
        if not global_settings["power_system_model"]:
            if is_na_selected:
                # if entire row inactive, keep row NA
                row_is_active = any(
                    (matrix.get((current_region, s)) and matrix[(current_region, s)].value != "NA")
                    for s in sectors
                )
                if not row_is_active:
                    for s in sectors:
                        dd = matrix.get((current_region, s))
                        if dd:
                            dd.value = "NA"
            else:
                # if any is selected, default others (Elc=High, others Low)
                for s in sectors:
                    dd = matrix.get((current_region, s))
                    if not dd:
                        continue
                    if s != "Elc" and dd.value == "NA":
                        dd.value = "Low"
                    elif s == "Elc" and dd.value == "NA":
                        dd.value = "High"
        
        # persist change
        save_config()
        page.update()

    def update_ui_matrix() -> None:
        """Rebuild the matrix rows."""
        # clear matrix grid container (index 4 -> matrix area)
        try:
            main_content.controls[4].controls[0].controls.clear()
        except Exception as e:
            logger.exception("Failed to clear UI matrix container: %s", e)
            return

        current_regions = get_current_regions()

        header_row = [ft.Text("Region/Sector", weight=ft.FontWeight.BOLD, width=120)] + [
            ft.Text(s, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER, width=80) for s in sectors
        ]
        matrix_rows = [ft.Row(header_row, alignment=ft.MainAxisAlignment.START)]

        matrix.clear()
        for region in current_regions:
            row_elements = [ft.Text(region, weight=ft.FontWeight.BOLD, width=120)]
            for sector in sectors:
                # RE-ADDED: PSM check for initial disabled state
                initial_disabled_state = global_settings["power_system_model"] and sector != "Elc"
                
                dd = ft.Dropdown(
                    options=[ft.dropdown.Option(level) for level in levels_all_sectors],
                    value="NA",
                    width=80,
                    content_padding=ft.padding.only(left=5, right=5),
                    disabled=initial_disabled_state, # Apply PSM state
                )
                matrix[(region, sector)] = dd
                row_elements.append(dd)

                dd.data = region
                dd.on_change = on_sector_dropdown_change

                # If saved config contains a value for this cell, apply it
                try:
                    key = f"{region}|{sector}"
                    if saved_cfg and saved_cfg.get("matrix") and key in saved_cfg["matrix"]:
                        val = saved_cfg["matrix"].get(key)
                        if val is not None:
                            dd.value = val
                except Exception as e:
                    logger.exception("Failed applying saved value for %s|%s: %s", region, sector, e)
                
                # RE-ADDED: Enforce power-system rule AFTER loading saved value
                try:
                    if global_settings.get("power_system_model", False) and sector != "Elc":
                        dd.value = "NA"
                        dd.disabled = True
                except Exception as e:
                    logger.exception("Failed enforcing power-system rule for %s|%s: %s", region, sector, e)

            matrix_rows.append(ft.Row(row_elements, alignment=ft.MainAxisAlignment.START))

        # push rows into the scrollable column
        try:
            main_content.controls[4].controls[0].controls.extend(matrix_rows)
            page.update()
        except Exception as e:
            logger.exception("Failed to render UI matrix rows: %s", e)

    def apply_dropdown_disabling() -> None:
        """
        RE-ADDED: Enable/disable row dropdowns depending on power system mode.
        """
        for region in get_current_regions():
            for sector in sectors:
                dd = matrix.get((region, sector))
                if not dd:
                    continue
                try:
                    if global_settings["power_system_model"]:
                        dd.disabled = (sector != "Elc")
                        if sector != "Elc":
                            dd.value = "NA"
                    else:
                        dd.disabled = False
                except Exception as e:
                    logger.exception("Error applying disabling for %s|%s: %s", region, sector, e)
        try:
            page.update()
        except Exception:
            pass
        save_config()

    def reset_matrix(e: ft.ControlEvent | None = None) -> None:
        """Hard reset: reset all settings, set all cells to 'NA', re-enable controls,
        clear status text, and overwrite the saved config."""
        nonlocal saved_cfg
    
        # 1) Reset global flags / controls
        default_scenario = "Current Measure"
        global_settings["scenario"] = default_scenario
        global_settings["power_system_model"] = False # RE-ADDED
        scenario_dropdown.value = default_scenario
        power_system_checkbox.value = False # RE-ADDED
    
        # 2) Reset all dropdowns in the current matrix
        for (r, s), dd in matrix.items():
            dd.disabled = False
            dd.value = "NA"
    
        # 3) Clear any status message
        status_text.value = ""
    
        # 4) Overwrite saved config in-memory and on-disk with a clean matrix
        saved_cfg = {
            "input_filename": (in_filename_text_field.value or "").strip(),
            "output_filename": (out_filename_text_field.value or "").strip(),
            "scenario": default_scenario,
            "power_system_model": False, # RE-ADDED
            "matrix": {f"{r}|{s}": "NA" for (r, s) in matrix.keys()},
        }
        try:
            os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
            with open(CONFIG_FILE, "w", encoding="utf-8") as fh:
                json.dump(saved_cfg, fh, indent=2, ensure_ascii=False)
            logger.debug("Reset config written to %s", CONFIG_FILE)
        except Exception as ex:
            logger.exception("Failed to write reset config: %s", ex)
    
        # 5) Rebuild UI and re-apply disabling logic (PSM is off, so all enabled)
        update_ui_matrix()
        apply_dropdown_disabling()
    
        # 6) Final paint
        try:
            page.update()
        except Exception:
            pass
    # Use the actual UI selections (rows with any non-NA cell)
    def selected_regions_for_run() -> List[str]:
        try:
            return sorted(list(infer_selected_regions_from_matrix(matrix)))
        except Exception:
            return []

    # --- Settings events ---
    def get_current_regions() -> List[str]:
        # Regions always include all
        return ALL_REGIONS
    def on_scenario_change(e: ft.ControlEvent) -> None:
        """Handle change for the scenario dropdown."""
        global_settings["scenario"] = str(e.control.value)
        save_config()
    
    scenario_dropdown.on_change = on_scenario_change
    
    # RE-ADDED: Handler for power system checkbox
    def on_power_system_change(e: ft.ControlEvent) -> None:
        global_settings["power_system_model"] = bool(e.control.value)
        update_ui_matrix()
        apply_dropdown_disabling()
        save_config()

    power_system_checkbox.on_change = on_power_system_change

    # --- Submit / run aggregation ---

    # attach saving to text field changes
    in_filename_text_field.on_change = lambda e: save_config()
    out_filename_text_field.on_change = lambda e: save_config()

    def on_submit(e: ft.ControlEvent) -> None:
        status_text.value = "Processing..."
        page.update()

        input_filename = in_filename_text_field.value.strip()
        if not input_filename:
            status_text.value = "Error: Filename cannot be empty."
            page.update()
            return

        output_filename = out_filename_text_field.value.strip()
        if not output_filename:
            status_text.value = "Error: Filename cannot be empty."
            page.update()
            return

        try:
            # persist current config before running
            save_config()
            # NOTE: csv_ids parameter is ignored now; keep an empty set for signature compatibility
            aggregate_sqlite_files(
                matrix=matrix,
                csv_ids=set(),  # ignored by the implementation
                global_settings=global_settings,
                get_current_regions=selected_regions_for_run,  # <-- use the UI-derived regions
                input_filename=input_filename,
                output_filename=output_filename,
            )
            status_text.value = f"Aggregation complete. Output: {output_filename}"
            logger.info("Aggregation finished successfully (input=%s output=%s)", input_filename, output_filename)
        except Exception as ex:
            logger.exception("Aggregation failed (input=%s output=%s): %s", input_filename, output_filename, ex)
            status_text.value = f"Error: {ex}"
        page.update()


    # --- Build initial matrix UI (all regions from the start) ---
    header_row = [ft.Text("Region/Sector", weight=ft.FontWeight.BOLD, width=120)] + [
        ft.Text(s, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER, width=80) for s in sectors
    ]
    matrix_rows = [ft.Row(header_row, alignment=ft.MainAxisAlignment.START)]

    for region in ALL_REGIONS:
        row_elements = [ft.Text(region, weight=ft.FontWeight.BOLD, width=120)]
        for sector in sectors:
            # All sectors use same options
            dd = ft.Dropdown(
                options=[ft.dropdown.Option(level) for level in levels_all_sectors],
                value="NA",
                width=80,
                content_padding=ft.padding.only(left=5, right=5),
                disabled=False,
            )
            matrix[(region, sector)] = dd
            row_elements.append(dd)
            dd.data = region
            dd.on_change = on_sector_dropdown_change
        matrix_rows.append(ft.Row(row_elements, alignment=ft.MainAxisAlignment.START))

    # Apply saved text inputs / controls if present
    try:
        if saved_cfg:
            in_filename_text_field.value = saved_cfg.get("input_filename", in_filename_text_field.value)
            out_filename_text_field.value = saved_cfg.get("output_filename", out_filename_text_field.value)
            
            # Apply saved scenario
            scenario_saved = saved_cfg.get("scenario", scenario_dropdown.value)
            scenario_dropdown.value = scenario_saved
            global_settings["scenario"] = scenario_saved
            
            # RE-ADDED: Apply saved PSM checkbox
            psm_saved = bool(saved_cfg.get("power_system_model", power_system_checkbox.value))
            power_system_checkbox.value = psm_saved
            global_settings["power_system_model"] = psm_saved
            
    except Exception:
        pass

    # Controls
    reset_button = ft.ElevatedButton("Reset Matrix", on_click=reset_matrix)
    submit_button = ft.ElevatedButton("Submit", on_click=on_submit)

    # Settings column
    settings_column = ft.Column(
        [
            ft.Text("Aggregation Settings", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            scenario_dropdown,
            power_system_checkbox, # RE-ADDED
            ft.Divider(),
            reset_button,
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.START,
        spacing=10,
        width=300,
    )

    # Main content layout (structure/spacing preserved)
    global main_content
    main_content = ft.Column(
        [
            image,
            ft.Text("CANOE UI", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Read accompanying document for details about choices and instructions", size=18),
            ft.Divider(),
            ft.Row(
                [
                    ft.Column(
                        matrix_rows,
                        scroll=ft.ScrollMode.ADAPTIVE,
                        height=400,
                        width=700,
                        horizontal_alignment=ft.CrossAxisAlignment.START,
                    ),
                    ft.VerticalDivider(),
                    settings_column,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.MainAxisAlignment.START,
            ),
            ft.Divider(),
            ft.Row(
                [in_filename_text_field, out_filename_text_field, submit_button, status_text],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=30,
            ),
        ],
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )

    page.add(main_content)
    page.update()

    # Build matrix from saved state first, then enforce disabling rules
    update_ui_matrix()
    apply_dropdown_disabling() # This will now apply PSM rules if loaded from config

if __name__ == "__main__":
    ft.app(target=main, assets_dir="./assets")