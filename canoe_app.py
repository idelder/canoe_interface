# -*- coding: utf-8 -*-
"""
CANOE UI Script 
By David Turnbull
"""

from __future__ import annotations

import csv
import os
import re
import sqlite3
from typing import Dict, Iterable, List, Set, Tuple

import flet as ft
import pandas as pd

# ------------------------------
# Paths & Constants
# ------------------------------
SQLITE_FOLDER = "input"
# MASTER_DB = "master.sqlite"
DATASETS_CSV = "input/datasets.csv"
SCHEMA_FILE = "input/schema.sql"

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
    "Ind": "INDHR",
    "Res": "RESHR",
    "Comm": "COMHR",
    "Tran": "TRPHR",
}

GENERIC_BY_SECTOR = {
    "Elc": "ELCHR001",
    "Ind": "INDHR001",
    "Res": "RESHR001",
    "Comm": "COMHR001",
    "Tran": "TRPHR001",
}

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
    "LAB": ["NLLAB", "NL", "LAB"],
    "USA": ["USA"],
}

ALL_ALIAS_TOKENS = sorted(
    {tok for toks in REGION_ALIASES.values() for tok in toks}, key=len, reverse=True
)
CONFLICT_SUFFIXES = {
    t: [u for u in ALL_ALIAS_TOKENS if u != t and u.endswith(t)] for t in ALL_ALIAS_TOKENS
}

# ------------------------------
# Post-Aggregation Filter / Cleanup
# ------------------------------

def filter_func(output_db: str) -> None:
    """Run cleanup transformations on the aggregated SQLite file."""
    conn = sqlite3.connect(output_db)
    curs = conn.cursor()

    # Iteratively prune orphan processes/techs
    finished = False
    while not finished:
        regions = [r[0] for r in curs.execute('SELECT region FROM Region').fetchall()]
        _ = regions  # reserved for future use

        # techs that output a non-demand commodity that isn't consumed anywhere
        bad_rt = curs.execute(
            """
            SELECT region, tech 
            FROM Efficiency 
            WHERE output_comm NOT IN (SELECT name FROM Commodity WHERE flag == 'd')
              AND (region, output_comm) NOT IN (
                    SELECT region, input_comm FROM Efficiency
              )
            """
        ).fetchall()

        tables = [t[0] for t in curs.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]

        for table in tables:
            cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
            if 'region' in cols and 'tech' in cols:
                for rt in bad_rt:
                    curs.execute(f"DELETE FROM {table} WHERE region == ? AND tech == ?", (rt[0], rt[1]))

        tech_remaining = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Efficiency').fetchall()}
        tech_before = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Technology').fetchall()}
        tech_gone = tech_before - tech_remaining

        for table in tables:
            cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
            if 'tech' in cols:
                for tech in tech_gone:
                    curs.execute(f'DELETE FROM {table} WHERE tech == ?', (tech,))

        finished = len(bad_rt) > 0

    # Timing-based pruning
    finished = False
    while not finished:
        # Time horizon (exclude final marker period)
        time_all = [p[0] for p in curs.execute('SELECT period FROM TimePeriod').fetchall()][:-1]

        # Build lifetimes map with sensible defaults
        lifetime_process = {}
        for r, t, v in curs.execute('SELECT region, tech, vintage FROM Efficiency').fetchall():
            lifetime_process[(r, t, v)] = 40
        for r, t, lt in curs.execute('SELECT region, tech, lifetime FROM LifetimeTech').fetchall():
            for v in time_all:
                lifetime_process[(r, t, v)] = lt
        for r, t, v, lp in curs.execute('SELECT region, tech, vintage, lifetime FROM LifetimeProcess').fetchall():
            lifetime_process[(r, t, v)] = lp

        df_eff = pd.read_sql_query('SELECT * FROM Efficiency', conn)
        df_eff['last_out'] = df_eff.apply(lambda row: row['vintage'] + int(lifetime_process[(row['region'], row['tech'], row['vintage'])]), axis=1)
        df_eff['last_out'] = df_eff['last_out'].apply(lambda p: min(2050, 5 * ((p - 1) // 5)))

        demand_comms = [c[0] for c in curs.execute("SELECT name FROM Commodity WHERE flag == 'd'").fetchall()]
        df_eff_nondem = df_eff.loc[~df_eff['output_comm'].isin(demand_comms)].copy()

        df_last_in = (
            df_eff_nondem.groupby(['region', 'input_comm'])['last_out'].max().rename('last_in')
        )
        df_eff_nondem = df_eff_nondem.join(
            df_last_in, on=['region', 'output_comm'], how='left'
        )
        df_remove = df_eff_nondem.loc[df_eff_nondem['last_in'] < df_eff_nondem['last_out']]

        for _, row in df_remove.iterrows():
            region, input_comm, tech, vintage, output_comm = (
                row['region'], row['input_comm'], row['tech'], row['vintage'], row['output_comm']
            )
            curs.execute(
                """
                DELETE FROM Efficiency 
                WHERE region = ? AND input_comm = ? AND tech = ? AND vintage = ? AND output_comm = ?
                """,
                (region, input_comm, tech, vintage, output_comm),
            )
            for tbl in ("CostVariable", "CostFixed", "EmissionActivity"):
                curs.execute(
                    f"DELETE FROM {tbl} WHERE region = ? AND tech = ? AND vintage = ?",
                    (region, tech, vintage),
                )

        finished = len(df_remove) > 0

    conn.commit()
    conn.close()


# ------------------------------
# ID Matching helpers
# ------------------------------

def region_tokens(region: str) -> List[str]:
    return REGION_ALIASES.get(region, [region])


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


# ------------------------------
# CSV Loader
# ------------------------------

def get_data_ids_from_csv(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        #print(f"[WARN] datasets.csv not found at {file_path}")
        return []
    # Try pandas first
    try:
        df = pd.read_csv(file_path)
        if 'data_id' in df.columns:
            return [str(x).strip() for x in df['data_id'].dropna().astype(str)]
        if df.shape[1] == 1:
            col = df.columns[0]
            return [str(x).strip() for x in df[col].dropna().astype(str)]
    except Exception:
        pass
    # Fallback CSV reader
    out: List[str] = []
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
    return out


# ------------------------------
# Desired IDs Builder (CCS removed)
# ------------------------------

def build_desired_ids_from_matrix(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    csv_ids: Set[str],
    global_settings: Dict[str, bool],
    get_current_regions,
) -> Set[str]:
    desired: Set[str] = set()
    selected_regions: Set[str] = set()
    current_regions = get_current_regions()

    # Track whether any Low CM / Low GNZ was chosen per sector (for adding generics)
    low_cm_seen = {"Ind": False, "Res": False, "Comm": False, "Tran": False}
    low_gnz_seen = {"Ind": False, "Res": False, "Comm": False, "Tran": False}

    # Active regions
    for region in current_regions:
        row_active = any(
            (matrix.get((region, s)) and matrix[(region, s)].value != "NA")
            for s in HIGH_PREFIXES.keys() | {"Elc"}
        )
        if row_active:
            selected_regions.add(region)

    # Electricity (High only)
    for region in current_regions:
        dd = matrix.get((region, "Elc"))
        if dd and str(dd.value).strip().upper() == "HIGH":
            base_el = csv_match_prefix_region(csv_ids, "ELCHR", region)
            # Exclude DEM/EINT/BINT; CCS removed entirely
            base_el -= csv_match_prefix_region(csv_ids, "ELCHRDEM", region)
            base_el -= csv_match_prefix_region(csv_ids, "ELCHREINT", region)
            base_el -= csv_match_prefix_region(csv_ids, "ELCHRBINT", region)
            desired |= base_el

    # Non-electric (Ind, Res, Comm, Tran)
    for region in current_regions:
        for sector, hr_prefix in HIGH_PREFIXES.items():
            dd = matrix.get((region, sector))
            if not dd:
                continue

            val = (str(dd.value) or "").strip().upper()

            if val == "HIGH":
                desired |= csv_match_prefix_region(csv_ids, hr_prefix, region)

            elif val == "LOW CM":
                low_cm_seen[sector] = True
                desired |= csv_match_prefix_region(csv_ids, LOW_PREFIXES[sector]["CM"], region)

            elif val == "LOW GNZ":
                low_gnz_seen[sector] = True
                desired |= csv_match_prefix_region(csv_ids, LOW_PREFIXES[sector]["GNZ"], region)
            # NA -> do nothing

    # After scanning all regions, add generic CM/NZ IDs if that mode was selected anywhere for the sector.
    for sector in ("Ind", "Res", "Comm", "Tran"):
        if low_cm_seen[sector]:
            gen_cm = GENERIC_LOW_BY_SECTOR.get(sector, {}).get("CM")
            if gen_cm and gen_cm in csv_ids:
                desired.add(gen_cm)
        if low_gnz_seen[sector]:
            gen_nz = GENERIC_LOW_BY_SECTOR.get(sector, {}).get("GNZ")
            if gen_nz and gen_nz in csv_ids:
                desired.add(gen_nz)

    # DEM (power-system demand expansions)
    if global_settings.get("power_system_model", False):
        for r in selected_regions:
            desired |= csv_match_prefix_region(csv_ids, "ELCHRDEM", r)

    # Interties
    intertie_pairs = [
        ("AB", "BC"), ("AB", "SK"), ("SK", "MB"),
        ("MB", "ON"), ("ON", "QC"),
        ("NB", "NS"), ("NB", "QC"), ("NB", "PEI"),
        ("NS", "PEI"), ("LAB", "NS"), ("LAB", "QC"),
        ("BC", "USA"), ("AB", "USA"), ("SK", "USA"),
        ("MB", "USA"), ("ON", "USA"), ("QC", "USA"), ("NB", "USA"),
    ]

    for a, b in intertie_pairs:
        sel_a = a in selected_regions
        sel_b = b in selected_regions

        if sel_a and sel_b:
            # Both sides selected -> use EINT only
            desired |= csv_match_intertie(csv_ids, "ELCHREINT", a, b)
        elif sel_a ^ sel_b:
            # Exactly one side selected -> use BINT only
            desired |= csv_match_intertie(csv_ids, "ELCHRBINT", a, b)

    # AGRI/DIST rules
    if not global_settings.get("power_system_model", False):
        if "AGRIHR001" in csv_ids:
            desired.add("AGRIHR001")
        for r in selected_regions:
            desired |= csv_match_prefix_region(csv_ids, "AGRIHR", r)

    if selected_regions:
        for r in selected_regions:
            desired |= csv_match_prefix_region(csv_ids, "FUELHR", r)
    if "FUELHR001" in csv_ids:
        desired.add("FUELHR001")

    # Generic sector HR IDs (only when that sector appears anywhere in the matrix)
    def _maybe_add_generic(sec_key: str) -> None:
        gen = GENERIC_BY_SECTOR.get(sec_key)
        if gen and gen in csv_ids:
            desired.add(gen)

    if global_settings.get("power_system_model", False):
        _maybe_add_generic("Elc")
    else:
        if any((matrix.get((r, "Elc")) and matrix[(r, "Elc")].value != "NA") for r in current_regions):
            _maybe_add_generic("Elc")

    for _sec in ("Ind", "Res", "Comm", "Tran"):
        if any((matrix.get((r, _sec)) and matrix[(r, _sec)].value != "NA") for r in current_regions):
            _maybe_add_generic(_sec)

    # Final guard: never include AGRIHR* in power system model
    if global_settings.get("power_system_model", False):
        desired = {x for x in desired if not x.startswith("AGRIHR")}

    #print(f"[DEBUG] Final desired IDs count: {len(desired)}")
    for ex in sorted(desired)[:20]:
        #print(f"[DEBUG]  - {ex}")
        continue
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
    In PSM mode, exclude R_ethos.
    """
    cursor = conn.cursor()
    #print(f"Using existing connection for {output_db} (power_system_model={power_system_model})")

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

        if power_system_model and 'R_ethos' in commodities:
            commodities.discard('R_ethos')

    except sqlite3.Error as e:
        #print(f"[ERROR] DB traversal failed: {e}")
        return [], []

    com_list = sorted(commodities)
    tech_list = sorted(technologies)

    #print(f"\nFound {len(com_list)} unique commodities.")
    #print(f"Found {len(tech_list)} unique technologies.")

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


def delete_rows_not_in_regions(conn: sqlite3.Connection, tables: List[str], allowed_regions: List[str]) -> None:
    if not allowed_regions:
        #print("[WARN] No regions selected; skipping region filtering to avoid empty DB.")
        return

    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = OFF;")

    placeholders = ",".join(["?"] * len(allowed_regions))
    for table in tables:
        try:
            if table_has_region_column(cur, table):
                cur.execute(f"DELETE FROM {table} WHERE region NOT IN ({placeholders});", allowed_regions)
                #print(f"[DEBUG] Region-pruned table: {table} (kept regions: {allowed_regions})")
        except sqlite3.Error as e:
            #print(f"[WARN] Skipped region filter for {table}: {e}")
            return
    conn.commit()
    cur.execute("PRAGMA foreign_keys = ON;")


# ------------------------------
# SQLite Aggregation
# ------------------------------

def aggregate_sqlite_files(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    csv_ids: Set[str],
    global_settings: Dict[str, bool],
    get_current_regions,
    output_filename: str,
    input_filename: str,
) -> None:
    desired_ids = build_desired_ids_from_matrix(matrix, csv_ids, global_settings, get_current_regions)
    if not desired_ids:
        #print("No matching IDs.")
        return

    selected_data_ids = sorted(desired_ids)
    id_str = "('" + "', '".join(selected_data_ids) + "')"
    #print(f"[DEBUG] IN-clause: {id_str}")

    master_db_path = input_filename
    output_db = output_filename

    if os.path.exists(output_db):
        os.remove(output_db)

    # Create output schema
    output_conn = sqlite3.connect(output_db)
    output_cursor = output_conn.cursor()
    with open(SCHEMA_FILE, 'r') as file:
        schema = file.read()
    output_cursor.executescript(schema)
    output_conn.commit()

    # Read from master
    master_conn = sqlite3.connect(master_db_path)
    master_cursor = master_conn.cursor()

    master_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in master_cursor.fetchall()]

    # Full-copy tables (no data_id filtering)
    full_copy_tables = [
        'CommodityType', 'MetaData', 'MetaDataReal', 'TechnologyType',
        'DataQualityCredibility', 'DataQualityGeography', 'Operator', 'Region',
        'DataQualityStructure', 'SeasonLabel', 'SectorLabel', 'DataQualityTechnology',
        'DataQualityTime', 'TimeOfDay', 'TimePeriod', 'TimePeriodType', 'TimeSeason',
        'TimeSeasonSequential', 'TimeSegmentFraction'
    ]

    output_cursor.execute('PRAGMA foreign_keys = OFF;')
    for table in tables:
        if table in full_copy_tables:
            master_cursor.execute(f"SELECT * FROM {table};")
        else:
            master_cursor.execute(f"SELECT * FROM {table} WHERE data_id IN {id_str};")
        data = master_cursor.fetchall()
        if data:
            columns = [col[0] for col in master_cursor.description]
            placeholders = ', '.join(['?'] * len(columns))
            output_cursor.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({placeholders});", data)
           # print(f"[DEBUG] Copied table: {table}{' (full copy)' if table in full_copy_tables else ' (filtered)'}")
        output_conn.commit()

    # Demand-led pruning
    com_list, tech_list = get_demand_lists_region_aware(
        output_db, output_conn, power_system_model=global_settings.get("power_system_model", False)
    )
    if tech_list:
        tech_str = "('" + "', '".join(tech_list) + "')"
        output_cursor.execute(f'DELETE FROM Efficiency WHERE tech NOT IN {tech_str}')
        output_cursor.execute(f'DELETE FROM Technology WHERE tech NOT IN {tech_str}')
        output_cursor.execute(f'DELETE FROM LifetimeTech WHERE tech NOT IN {tech_str}')
        output_cursor.execute(f'DELETE FROM CostVariable WHERE tech NOT IN {tech_str}')
        output_cursor.execute(f'DELETE FROM EmissionActivity WHERE tech NOT IN {tech_str}')
    if com_list:
        com_str = "('" + "', '".join(com_list) + "')"
        output_cursor.execute(f'DELETE FROM Commodity WHERE name NOT IN {com_str}')

    output_conn.commit()

    # Region pruning
    selected_regions = sorted(infer_selected_regions_from_matrix(matrix))
    #print(f"[DEBUG] Selected regions for pruning: {selected_regions if selected_regions else '(none)'}")
    delete_rows_not_in_regions(output_conn, tables, selected_regions)

    master_conn.close()
    output_conn.close()

    # Final post-aggregation cleanup
    filter_func(output_db)
    #print(f"[DEBUG] Aggregation complete. Output: {output_db}")


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
    levels_other_sectors = ["Low CM", "Low GNZ", "High", "NA"]
    levels_elc_sector = ["NA", "High"]

    # UI storage
    matrix: Dict[Tuple[str, str], ft.Dropdown] = {}

    # Global flag (CCS removed)
    global_settings = {"power_system_model": False}

    # Widgets
    status_text = ft.Text("")
    in_filename_text_field = ft.TextField(label="Input file path (e.g. dataset.sqlite)", value="", width=260)
    out_filename_text_field = ft.TextField(label="Output file path (e.g. canoe.sqlite)", value="", width=260)
    power_system_checkbox = ft.Checkbox(label="Power system model", value=False)
    image = ft.Container(content=ft.Image(src="./assets/logo.png", height=50, width=60), alignment=ft.alignment.top_right)

    def get_current_regions() -> List[str]:
        # Regions always include all
        return ALL_REGIONS

    # --- UI builders / updaters ---

    def on_sector_dropdown_change(e: ft.ControlEvent) -> None:
        """Auto-fill sensible defaults when user selects a sector (normal mode)."""
        current_region = e.control.data
        is_na_selected = (e.control.value == "NA")

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
                # if any is selected, default others (Elc=High, others Low CM)
                for s in sectors:
                    dd = matrix.get((current_region, s))
                    if not dd:
                        continue
                    if s != "Elc" and dd.value == "NA":
                        dd.value = "Low CM"
                    elif s == "Elc" and dd.value == "NA":
                        dd.value = "High"
        page.update()

    def update_ui_matrix() -> None:
        """Rebuild the matrix rows whenever regions set changes."""
        # clear matrix grid container (index 4 -> matrix area)
        main_content.controls[4].controls[0].controls.clear()

        current_regions = get_current_regions()

        header_row = [ft.Text("Region/Sector", weight=ft.FontWeight.BOLD, width=120)] + [
            ft.Text(s, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER, width=80) for s in sectors
        ]
        matrix_rows = [ft.Row(header_row, alignment=ft.MainAxisAlignment.START)]

        matrix.clear()
        for region in current_regions:
            row_elements = [ft.Text(region, weight=ft.FontWeight.BOLD, width=120)]
            for sector in sectors:
                initial_disabled_state = global_settings["power_system_model"] and sector != "Elc"
                options = levels_elc_sector if sector == "Elc" else levels_other_sectors
                dd = ft.Dropdown(
                    options=[ft.dropdown.Option(level) for level in options],
                    value="NA",
                    width=80,
                    content_padding=ft.padding.only(left=5, right=5),
                    disabled=initial_disabled_state,
                )
                matrix[(region, sector)] = dd
                row_elements.append(dd)

                dd.data = region
                dd.on_change = on_sector_dropdown_change

            matrix_rows.append(ft.Row(row_elements, alignment=ft.MainAxisAlignment.START))

        # push rows into the scrollable column
        main_content.controls[4].controls[0].controls.extend(matrix_rows)
        page.update()

    def apply_dropdown_disabling() -> None:
        """Enable/disable row dropdowns depending on power system mode."""
        for region in get_current_regions():
            for sector in sectors:
                dd = matrix.get((region, sector))
                if not dd:
                    continue
                if global_settings["power_system_model"]:
                    dd.disabled = (sector != "Elc")
                    if sector != "Elc":
                        dd.value = "NA"
                else:
                    dd.disabled = False
        page.update()

    def reset_matrix(e: ft.ControlEvent | None = None) -> None:
        """Reset toggles and rebuild UI."""
        global_settings["power_system_model"] = False
        power_system_checkbox.value = False
        update_ui_matrix()
        page.update()

    # --- Settings events ---

    def on_power_system_change(e: ft.ControlEvent) -> None:
        global_settings["power_system_model"] = bool(e.control.value)
        update_ui_matrix()
        apply_dropdown_disabling()

    power_system_checkbox.on_change = on_power_system_change

    # --- Submit / run aggregation ---

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

        csv_ids = set(get_data_ids_from_csv(DATASETS_CSV))
        if not csv_ids:
            status_text.value = f"Error: No data_ids found in {DATASETS_CSV}"
            page.update()
            return

        try:
            aggregate_sqlite_files(
                matrix=matrix,
                csv_ids=csv_ids,
                global_settings=global_settings,
                get_current_regions=get_current_regions,
                input_filename=input_filename,
                output_filename=output_filename,
            )
            status_text.value = f"Aggregation complete. Output: {output_filename}.sqlite"
        except Exception as ex:
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
            options = levels_elc_sector if sector == "Elc" else levels_other_sectors
            dd = ft.Dropdown(
                options=[ft.dropdown.Option(level) for level in options],
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

    # Controls
    reset_button = ft.ElevatedButton("Reset Matrix", on_click=reset_matrix)
    submit_button = ft.ElevatedButton("Submit", on_click=on_submit)

    # Settings column — layout preserved (CCS checkbox removed)
    settings_column = ft.Column(
        [
            ft.Text("Aggregation Settings", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            power_system_checkbox,
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

    # Initial apply
    apply_dropdown_disabling()


if __name__ == "__main__":
    ft.app(target=main)
