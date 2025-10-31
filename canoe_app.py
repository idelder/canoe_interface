# -*- coding: utf-8 -*-
"""
CANOE UI Script (Refactored w/ PSM)
By David Turnbull
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterable, List, Set, Tuple
import flet as ft
import pandas as pd
from enum import Enum
from time import sleep
from itertools import product

# ------------------------------
# Paths & Constants
# ------------------------------
SQLITE_FOLDER = "input"
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

class Region(Enum):
    ALBERTA = 'AB'
    BRITISH_COLUMBIA = 'BC'
    MANITOBA = 'MB'
    NEW_BRUNSWICK = 'NB'
    NEWFOUNDLAND_AND_LABRADOR = 'NLLAB'
    NOVA_SCOTIA = 'NS'
    ONTARIO = 'ON'
    PRINCE_EDWARD_ISLAND = 'PEI'
    QUEBEC = 'QC'
    SASKATCHEWAN = 'SK'
    USA = 'USA'
    NONE = ''

class Sector(Enum):
    AGRICULTURE = 'AGRI'
    COMMERCIAL = 'COM'
    ELECTRICITY = 'ELC'
    FUEL = 'FUEL'
    INDUSTRY = 'IND'
    RESIDENTIAL = 'RES'
    TRANSPORTATION = 'TRP'

class Variant(Enum):
    HIGH_RESOLUTION = 'HR'
    CURRENT_MEASURES = 'CM'
    NET_ZERO = 'NZ'

class Levels(Enum):
    HIGH_RESOLUTION = 'HIGH'
    LOW_RESOLUTION = 'LOW'
    EXCLUDED = '-'

LOW_SCENARIOS = [
    "Current Measures",
    "Global Net Zero"
]
DEFAULT_LOW = 'Current Measures'

TABLE_SECTORS = sorted(
    s.value for s in [
        Sector.COMMERCIAL,
        Sector.ELECTRICITY,
        Sector.INDUSTRY,
        Sector.RESIDENTIAL,
        Sector.TRANSPORTATION
    ]
)

TABLE_REGIONS = sorted(
    r.value for r in [
        Region.ALBERTA,
        Region.BRITISH_COLUMBIA,
        Region.MANITOBA,
        Region.NEW_BRUNSWICK,
        Region.NEWFOUNDLAND_AND_LABRADOR,
        Region.NOVA_SCOTIA,
        Region.ONTARIO,
        Region.PRINCE_EDWARD_ISLAND,
        Region.QUEBEC,
        Region.SASKATCHEWAN
    ]
)

# SECTVARREG001
def make_id(
    sector: Sector,
    variant: Variant,
    region: Region,
    version: int = 1
) -> str:
    version = '0'*(3-len(str(version))) + str(version)
    return f"{sector.value}{variant.value}{region.value}{version}"

# These are handled by the schema and do not come from the dataset
INDEX_TABLES = {
    'CommodityType','Operator','TechnologyType','TimePeriodType',
    'DataQualityCredibility','DataQualityDataQualityGeography',
    'DataQualityStructure','DataQualityTechnology','DataQualityTime',
    'TechnologyLabel','CommodityLabel','DataSourceLabel'
}

# ------------------------------
# Post-Aggregation Filter / Cleanup
# ------------------------------

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

        curs.execute("PRAGMA foreign_keys = OFF;")

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
                logger.warning("Pass 1 made no progress in iter %d; stopping to avoid endless loop.", it)
                break

        # ---- Pass 2: timing pruning ----
        for it in range(1, MAX_ITERS_PASS2 + 1):
            time_all = [int(p[0]) for p in curs.execute('SELECT period FROM TimePeriod').fetchall()]

            lifetime_process: Dict[Tuple[str,str,int], int] = {}
            DEFAULT_LT = 40

            for r, t, v in curs.execute('SELECT region, tech, vintage FROM Efficiency').fetchall():
                lifetime_process[(r, t, int(v))] = DEFAULT_LT

            for r, t, lt in curs.execute('SELECT region, tech, lifetime FROM LifetimeTech').fetchall():
                for v in time_all:
                    lifetime_process[(r, t, int(v))] = int(lt)

            for r, t, v, lp in curs.execute('SELECT region, tech, vintage, lifetime FROM LifetimeProcess').fetchall():
                lifetime_process[(r, t, int(v))] = int(lp)

            df_eff = pd.read_sql_query('SELECT * FROM Efficiency', conn)
            df_eff['vintage'] = pd.to_numeric(df_eff['vintage'], errors='coerce').fillna(0).astype(int)

            snap5_max2050 = lambda y: min(2050, (y // 5) * 5)
            df_eff['last_out'] = [
                snap5_max2050(v + int(lifetime_process.get((r, t, v), DEFAULT_LT)))
                for r, t, v in df_eff[['region','tech','vintage']].itertuples(index=False, name=None)
            ]
            demand_comms = {c[0] for c in curs.execute("SELECT name FROM Commodity WHERE flag = 'd'").fetchall()}
            df_nd = df_eff.loc[~df_eff['output_comm'].isin(demand_comms)].copy()

            df_last_in = (
                df_eff.groupby(['region','input_comm'], as_index=True)['last_out']
                      .max()
                      .rename('last_in')
            )
            df_nd = df_nd.merge(df_last_in, left_on=['region','output_comm'], right_index=True, how='left')
            df_nd['last_in'] = pd.to_numeric(df_nd['last_in'], errors='coerce').fillna(0).astype(int)

            df_remove = df_nd.loc[df_nd['last_in'] < df_nd['last_out']].copy()

            deleted_total = 0
            for region, input_comm, tech, vintage, output_comm in df_remove[['region','input_comm','tech','vintage','output_comm']].itertuples(index=False, name=None):
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
                    curs.execute(
                        f"DELETE FROM {tbl} WHERE region = ? AND tech = ? AND CAST(vintage AS INTEGER) = ?",
                        (region, tech, int(vintage))
                    )
                    if curs.rowcount and curs.rowcount > 0: deleted_total += curs.rowcount

            conn.commit()
            if deleted_total == 0:
                logger.warning("Pass 2 made no progress in iter %d; stopping to avoid endless loop.", it)
                break

        curs.execute("PRAGMA foreign_keys = ON;")

        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception("filter_func failed for %s: %s", output_db, e)
        raise RuntimeError(f"filter_func failed for {output_db}: {e}")

# ------------------------------
# Initialisation
# ------------------------------

def collect_db_data_ids(db_path: str) -> Set[str]:
    """
    Get available data IDs from the dataset
    """
    ids: Set[str] = set()
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT data_id DataSet")
        ids = {id[0] for id in cur.fetchall()}
    except Exception as e:
        logger.exception("collect_db_data_ids failed for %s: %s", db_path, e)
        raise RuntimeError(f"collect_db_data_ids failed for {db_path}: {e}")
    return ids

# New: lightweight config loader (returns dict)
def load_config() -> Dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
                logger.debug("Loaded config from %s", CONFIG_FILE)
                return cfg
    except Exception as e:
        logger.exception("Failed to load config %s: %s", CONFIG_FILE, e)
        raise RuntimeError(f"Failed to load config: {e}")
    return {}

# ------------------------------
# Desired IDs Builder
# ------------------------------

def build_desired_ids_from_matrix(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    global_settings: Dict[str, Any],
) -> Set[str]:
    """
    Use only ids that actually exist in the current DB.
    Interties are *directional*: endogenous keeps both ways iff both regions selected;
    boundary keeps only origin->dest when origin is selected.
    """

    # read global flags
    low_scenario = global_settings.get("low_scenario", DEFAULT_LOW)
    is_psm  = bool(global_settings.get("power_system_model", False))

    # which regions are actually active in the UI (row not all NA)
    try:
        current_regions = sorted(list(infer_selected_regions_from_matrix(matrix)))
    except Exception:
        current_regions = []

    selected_regions: Set[str] = set()
    for r in current_regions:
        row_active = any(
            (matrix.get((r, s)) and str(matrix[(r, s)].value) != Levels.EXCLUDED.value)
            for s in TABLE_SECTORS
        )
        if row_active:
            selected_regions.add(r)

    desired: Set[str] = set()

    # electricity (High only; drop DEM/EINT/BINT here; they’re added explicitly below)
    for r in current_regions:
        dd = matrix.get((r, Sector.ELECTRICITY.value))
        if dd and dd.value == Levels.HIGH_RESOLUTION.value:
            desired.add(make_id(Sector.ELECTRICITY, Variant.HIGH_RESOLUTION, r))

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
        if any((matrix.get((r, "Elc")) and str(matrix[(r,"Elc")].value).upper() != "") for r in current_regions):
            maybe_add_generic("Elc")
    for sec in ("Ind","Res","Comm","Tran"):
        if any((matrix.get((r, sec)) and str(matrix[(r,sec)].value).upper() != "") for r in current_regions):
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
        raise RuntimeError(f"DB traversal failed for {output_db}: {e}")

    com_list = sorted(commodities)
    tech_list = sorted(technologies)

    return com_list, tech_list


def infer_selected_regions_from_matrix(matrix: Dict[Tuple[str, str], ft.Dropdown]) -> Set[str]:
    """Regions that have any sector set to something other than 'NA'."""
    regions = {r for (r, _s) in matrix.keys()}
    picked: Set[str] = set()
    for r in regions:
        any_on = any(
            (dd := matrix.get((r, s))) is not None and str(dd.value).upper() != ""
            for s in ("Ind", "Res", "Comm", "Elc", "Tran")
        )
        if any_on:
            picked.add(r)
    return picked

# ------------------------------
# SQLite Aggregation
# ------------------------------

def aggregate_sqlite_files(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    global_settings: Dict[str, Any],
    output_filename: str,
    input_filename: str,
) -> None:
    """
    Order of operations:
      1) Collect existing data_ids from input DB
      2) Build desired ids (directional interties, UI-selected regions)
      3) Copy tables (filter by data_id where present)
      4) Final cleanup filter_func (respect pinned techs)
    """
    try:
        # 1) Discover ids that actually exist
        available_ids = collect_db_data_ids(input_filename)
        if not available_ids:
            msg = "No data_id values found in the input database."
            logger.error(msg)
            raise RuntimeError(msg)

        # 2) Choose desired ids from the UI (directional interties), then restrict to available
        desired_ids = build_desired_ids_from_matrix(
            matrix=matrix,
            csv_ids=available_ids,
            global_settings=global_settings,
        ) or set()
        selected_data_ids = sorted(desired_ids & available_ids)

        if not selected_data_ids:
            msg = "Got no selected data IDs to merge"
            logger.error(msg)
            raise RuntimeError(msg)

        # 3) Create schema and copy tables (data_id-filtered where present)
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
        out_conn.execute(f"ATTACH {input_filename} AS dataset")

        tables = {t[0] for t in in_cur.fetchall()}
        # Exclude label/index-only tables handled by schema
        tables -= INDEX_TABLES
        out_cur.execute("PRAGMA foreign_keys = OFF;")
        for t in tables:
            cols = [c[1] for c in in_cur.execute(f"PRAGMA table_info({t});")]
            if 'data_id' in cols:
                for data_id in selected_data_ids:
                    out_cur.execute(f"INSERT INTO {t} SELECT * FROM dataset.{t} WHERE data_id == '{data_id}';")
                    out_conn.commit()
            else:
                out_cur.execute(f"INSERT INTO {t} SELECT * FROM dataset.{t};")
                out_conn.commit()

        # 4) Final cleanup honoring pinned techs
        in_conn.close(); out_conn.close()
        filter_func(output_filename)

        logger.info("Aggregation complete. Output: %s", output_filename)
    except Exception as e:
        logger.exception("Unhandled error during aggregation (input=%s output=%s): %s", input_filename, output_filename, e)
        raise RuntimeError(f"Unhandled error during aggregation: {e}")

# ------------------------------
# Flet UI (layout preserved; CCS removed)
# ------------------------------

def main(page: ft.Page) -> None:
    page.title = "CANOE UI"
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window_width = 1200
    page.window_height = 700
    
    # UI storage
    matrix: Dict[Tuple[str, str], ft.Dropdown] = {}

    # Global flags
    global_settings = {
        "low_scenario": DEFAULT_LOW,    # Default scenario
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
            ft.dropdown.Option(scen)
            for scen in LOW_SCENARIOS
        ],
        value=DEFAULT_LOW, # Default
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
                "low_scenario": (scenario_dropdown.value or DEFAULT_LOW),
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
            raise RuntimeError(f"Failed to save config: {e}")

    # --- UI builders / updaters ---

    def on_sector_dropdown_change(e: ft.ControlEvent) -> None:
        """Auto-fill sensible defaults when user selects a sector (normal mode)."""
        r, s = e.control.data
        is_na_selected = (e.control.value == Levels.EXCLUDED.value)

        sleep(0.05) # slight delay to allow UI update

        # RE-ADDED: Do not auto-fill if in power system mode
        if not global_settings["power_system_model"]:
            if is_na_selected:
                if s == Sector.ELECTRICITY.value:
                    for _s in TABLE_SECTORS:
                        if _s == s: continue
                        dd = matrix.get((r, _s))
                        if not dd:
                            continue
                        dd.value = Levels.EXCLUDED.value
                else:
                    # if row is active cant have an inactive sector
                    row_is_active = any(
                        (matrix.get((r, _s)) and matrix[(r, _s)].value != Levels.EXCLUDED.value)
                        for _s in TABLE_SECTORS
                    )
                    if row_is_active:
                        e.control.value = Levels.LOW_RESOLUTION.value
                        e.control.update()
            else:
                # if any is selected, default others (Elc=High, others Low)
                for _s in TABLE_SECTORS:
                    if _s == s: continue
                    dd = matrix.get((r, _s))
                    if not dd:
                        continue
                    if _s != Sector.ELECTRICITY.value and dd.value == Levels.EXCLUDED.value:
                        dd.value = Levels.LOW_RESOLUTION.value

                    elif _s == Sector.ELECTRICITY.value and dd.value == Levels.EXCLUDED.value:
                        dd.value = Levels.HIGH_RESOLUTION.value
        
        # persist change
        save_config()
        page.update()

    def update_ui_matrix() -> None:
        """Rebuild the matrix rows."""

        for region, sector in product(TABLE_REGIONS, TABLE_SECTORS):

            dd = matrix.get((region, sector))
            if not dd:
                continue
            
            # RE-ADDED: Enforce power-system rule AFTER loading saved value
            try:
                if global_settings.get("power_system_model", False) and sector != Sector.ELECTRICITY.value:
                    dd.disabled = True
                else:
                    dd.disabled = False
            except Exception as e:
                logger.exception("Failed enforcing power-system rule for %s|%s: %s", region, sector, e)
                raise RuntimeError(f"Failed enforcing power-system rule for {region}|{sector}: {e}")
                
        page.update()
        save_config()

    def reset_matrix(e: ft.ControlEvent | None = None) -> None:
        """Hard reset: reset all settings, set all cells to 'NA', re-enable controls,
        clear status text, and overwrite the saved config."""
    
        # 1) Reset global flags / controls
        global_settings["low_scenario"] = DEFAULT_LOW
        global_settings["power_system_model"] = False # RE-ADDED
        scenario_dropdown.value = DEFAULT_LOW
        power_system_checkbox.value = False # RE-ADDED
    
        # 2) Reset all dropdowns in the current matrix
        for (r, s), dd in matrix.items():
            dd.disabled = False
            dd.value = Levels.EXCLUDED.value
    
        # 3) Clear any status message
        status_text.value = ""
    
        # 4) Rebuild UI and re-apply disabling logic (PSM is off, so all enabled)
        update_ui_matrix()
        save_config()
    
        # 5) Final paint
        try:
            page.update()
        except Exception:
            pass

    # --- Settings events ---
    def on_scenario_change(e: ft.ControlEvent) -> None:
        """Handle change for the scenario dropdown."""
        global_settings["low_scenario"] = str(e.control.value)
        save_config()
    
    scenario_dropdown.on_change = on_scenario_change
    
    # RE-ADDED: Handler for power system checkbox
    def on_power_system_change(e: ft.ControlEvent) -> None:
        global_settings["power_system_model"] = bool(e.control.value)
        update_ui_matrix()
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
            aggregate_sqlite_files(
                matrix=matrix,
                global_settings=global_settings,
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
        ft.Text(s, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER, width=80) for s in TABLE_SECTORS
    ]
    matrix_rows = [ft.Row(header_row, alignment=ft.MainAxisAlignment.START)]

    for region in TABLE_REGIONS:
        row_elements = [ft.Text(region, weight=ft.FontWeight.BOLD, width=120)]
        for sector in TABLE_SECTORS:

            # Update matrix values
            key = f"{region}|{sector}"
            if saved_cfg and saved_cfg.get("matrix") and key in saved_cfg["matrix"]:
                try:
                    val = saved_cfg["matrix"].get(key)
                    if val is None:
                        val = Levels.EXCLUDED.value
                except Exception as e:
                    logger.exception("Failed applying saved value for %s|%s: %s", region, sector, e)
                    raise RuntimeError(f"Failed applying saved value for {region}|{sector}: {e}")

            if sector == Sector.ELECTRICITY.value:
                level_options = [
                    Levels.HIGH_RESOLUTION,
                    Levels.EXCLUDED
                ]
            else:
                level_options = Levels
            # All sectors use same options
            dd = ft.Dropdown(
                options=[ft.dropdown.Option(l.value) for l in level_options],
                value=val,
                width=80,
                content_padding=ft.padding.only(left=5, right=5),
                disabled=False,
            )
            matrix[(region, sector)] = dd
            row_elements.append(dd)
            dd.data = (region, sector)
            dd.on_change = on_sector_dropdown_change
        matrix_rows.append(ft.Row(row_elements, alignment=ft.MainAxisAlignment.START))

    # Apply saved text inputs / controls if present
    try:
        if saved_cfg:
            in_filename_text_field.value = saved_cfg.get("input_filename", in_filename_text_field.value)
            out_filename_text_field.value = saved_cfg.get("output_filename", out_filename_text_field.value)
            
            # Apply saved scenario
            scenario_saved = saved_cfg.get("low_scenario", scenario_dropdown.value)
            scenario_dropdown.value = scenario_saved
            global_settings["low_scenario"] = scenario_saved
            
            # Apply saved PSM checkbox
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

if __name__ == "__main__":
    ft.app(target=main, assets_dir="./assets")