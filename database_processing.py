"""
CANOE UI database processing elements
By David Turnbull
"""

import sqlite3
from typing import Dict, Set, Tuple, Any
from log_setup import setup_logging
import os
import pandas as pd

from constants import INDEX_TABLES, LTT_DEFAULT
from directories import SCHEMA_FILE

# Get logger for this module
logger = setup_logging("database_processing")

#########################################
# Helpers
#########################################

def vers_to_str(v: int) -> str:
    """
    Convert version integer to zero-padded string
    """
    return '0'*(3-len(str(v))) + str(v)

def snap5_max2050(year: int):
    """
    Snap year to nearest prior 5-year period, max 2050
    """
    return min(2050, ((year-1) // 5) * 5)

#########################################
# Initialisation
#########################################

def collect_db_data_ids(db_path: str) -> Set[str]:
    """
    Get available data IDs from the dataset
    """
    ids: Set[str] = set()
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT data_id FROM DataSet")
        ids = {id[0][0:-3] for id in cur.fetchall()}
    except Exception as e:
        logger.exception("collect_db_data_ids failed for %s: %s", db_path, e)
        raise RuntimeError(f"collect_db_data_ids failed for {db_path}: {e}")
    return ids

def collect_latest_id(db_path: str, base_id: str) -> str:
    """
    Get latest versioned data ID from the dataset for a given base ID
    """
    latest_id = ""
    latest_version = -1
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT data_id FROM DataSet WHERE data_id LIKE ?", (f"{base_id}%",))
        for id in cur.fetchall():
            version = int(id[0][-3:])
            if version > latest_version:
                latest_version = version
                latest_id = id[0]
    except Exception as e:
        logger.exception("collect_latest_id failed for %s and base_id %s: %s", db_path, base_id, e)
        raise RuntimeError(f"collect_latest_id failed for {db_path} and base_id {base_id}: {e}")
    return latest_id

def get_viable_data_ids(
    input_filename: str,
    desired_ids: Set[str]
) -> Set[str]:
    """
    Get intersection of desired data IDs and those available in the input DB
    """

    available_ids = collect_db_data_ids(input_filename)
    if not available_ids:
        msg = "No data_id values found in the input database."
        logger.error(msg)
        raise RuntimeError(msg)

    if not desired_ids:
        msg = "Got no selected data IDs to merge"
        logger.error(msg)
        raise RuntimeError(msg)
    
    good_ids = desired_ids.intersection(available_ids)
    if not good_ids:
        msg = "No selected data_id values found in the input database."
        logger.error(msg)
        raise RuntimeError(msg)
    
    return good_ids

def get_latest_data_ids(
        input_filename: str,
        desired_ids: Set[str]
    ) -> Set[str]:
    """
    From a set of data IDs without versions, get the latest versioned IDs available in the input DB
    """
    viable_ids = get_viable_data_ids(input_filename, desired_ids)
    latest_ids = sorted(set(
        collect_latest_id(input_filename, base_id)
        for base_id in viable_ids
    ))
    logger.debug("Transferring the following data_IDs: %s", latest_ids)
    return latest_ids

def initialize_output_database(output_filename: str) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """
    Create output database and initialize schema
    """
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
        conn = sqlite3.connect(output_filename)
        curs = conn.cursor()
        with open(SCHEMA_FILE, 'r') as file:
            curs.executescript(file.read())
        logger.info("Initialized output database schema: %s", output_filename)
        return conn, curs
    except Exception as e:
        logger.exception("initialize_output_filename failed for %s: %s", output_filename, e)
        raise RuntimeError(f"initialize_output_filename failed for {output_filename}: {e}")

#########################################
# SQLite Aggregation
#########################################

def aggregate_sqlite_files(
    input_filename: str,
    output_filename: str,
    global_settings: Dict[str, Any],
    desired_ids: Set[str],
) -> None:
    """
    Order of operations:
      1) Get viable data IDs
      2) Initialise database connections
      3) Copy tables (filter by data_id where present)
      4) Final cleanup post_process
    """

    conn = None
    curs = None

    try:
        cmd: str = None # initialise for error logging if needed

        # 1) Get viable data IDs to transfer
        data_ids = get_latest_data_ids(input_filename, desired_ids)
        
        # 2) Initialise database connections
        conn, curs = initialize_output_database(output_filename)
        conn.execute(f'ATTACH "{input_filename}" AS dataset')
        conn.execute("PRAGMA foreign_keys = OFF;")

        # 3) Copy data table by table (data_id-filtered where present)
        curs.execute('SELECT name FROM dataset.sqlite_master WHERE type="table";')
        tables = {t[0] for t in curs.fetchall()}
        tables -= INDEX_TABLES # exclude label/index-only tables handled by schema
        tables = sorted(tables)
        
        logger.debug("Executing SQLite transfers:")
        for t in tables:

            if not global_settings.get("is_processing", True):
                return
            
            cols = [c[1] for c in curs.execute(f"PRAGMA dataset.table_info({t});")]
            if 'data_id' in cols:
                for data_id in data_ids:

                    if not global_settings.get("is_processing", True):
                        return

                    cmd = f"INSERT OR IGNORE INTO {t} SELECT * FROM dataset.{t} WHERE data_id == '{data_id}';"
                    curs.execute(cmd)
                    conn.commit()
            else:
                cmd = f"INSERT OR IGNORE INTO {t} SELECT * FROM dataset.{t};"
                curs.execute(cmd)
                conn.commit()

        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute('DETACH dataset')
        conn.close()

        # 4) Final cleanup
        post_process(output_filename = output_filename, global_settings = global_settings)

        logger.info("Aggregation complete. Output: %s", output_filename)

    except Exception as e:
        logger.exception("Unhandled error during aggregation (input=%s output=%s): %s", input_filename, output_filename, e)
        if cmd:
            logger.debug("Last SQLite command: %s", cmd)
        raise RuntimeError(f"Unhandled error during aggregation: {e}")
    
    finally:
        if conn:
            conn.execute('DETACH dataset')
            conn.close()
    
#########################################
# Post-Aggregation Filter / Cleanup
#########################################

def post_process(
        output_filename: str,
        global_settings: Dict[str, Any]
    ) -> None:
    """
    Removes supply-side orphans (region-tech combos that lead nowhere because
    of excluded regions/sectors/resolutions or lifetime pruning).
    """

    MAX_ITERS = 20
    conn = None

    try:
        conn = sqlite3.connect(output_filename)
        curs = conn.cursor()

        curs.execute("PRAGMA foreign_keys = OFF;")

        # ---- Pass 1: Remove supply orphans by region (lifetime naive) ----
        for iter in range(MAX_ITERS):

            if not global_settings.get("is_processing", True):
                return

            bad_rt = curs.execute(
                """
                SELECT DISTINCT region, tech
                FROM Efficiency 
                WHERE output_comm NOT IN (SELECT name FROM Commodity WHERE flag = 'd')
                  AND (region, output_comm) NOT IN (SELECT region, input_comm FROM Efficiency)
                """
            ).fetchall()
            if not bad_rt:
                logger.debug(
                    "Lifetime-naive supply-side orphan removal complete after %d iterations.",
                    iter
                )
                break
            else:
                logger.debug(
                    "Removing the following region-tech orphans: %s",
                    bad_rt
                )

            deleted_total = 0
            tables = [t[0] for t in curs.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
            for table in tables:              
                cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
                if 'region' in cols and 'tech' in cols:
                    for region, tech in bad_rt:

                        if not global_settings.get("is_processing", True):
                            return
            
                        try:
                            curs.execute(f"DELETE FROM {table} WHERE region = ? AND tech = ?", (region, tech))
                            if curs.rowcount and curs.rowcount > 0:
                                deleted_total += curs.rowcount
                        except sqlite3.Error:
                            pass

            tech_remaining = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Efficiency').fetchall()}
            tech_before    = {t[0] for t in curs.execute('SELECT DISTINCT tech FROM Technology').fetchall()}
            tech_gone = tech_before - tech_remaining
            if tech_gone:
                logger.debug(
                    "Removing the following orphan techs: %s",
                    tech_gone
                )
                for table in tables:
                    cols = [c[1] for c in curs.execute(f'PRAGMA table_info({table});')]
                    if 'tech' in cols:
                        for tech in tech_gone:

                            if not global_settings.get("is_processing", True):
                                return
            
                            try:
                                curs.execute(f"DELETE FROM {table} WHERE tech = ?", (tech,))
                                if curs.rowcount and curs.rowcount > 0:
                                    deleted_total += curs.rowcount
                            except sqlite3.Error:
                                pass
                
            conn.commit()
            if deleted_total == 0:
                logger.debug(
                    "Lifetime-naive supply-side orphan removal complete after %d iterations.",
                    iter
                )
                break

        # ---- Pass 2: Remove supply orphans due to lifetime pruning ----
        for iter in range(MAX_ITERS):

            if not global_settings.get("is_processing", True):
                return

            time_all = [int(p[0]) for p in curs.execute('SELECT period FROM TimePeriod').fetchall()]

            lifetime_process: Dict[Tuple[str,str,int], int] = {}

            for r, t, v in curs.execute('SELECT region, tech, vintage FROM Efficiency').fetchall():
                lifetime_process[(r, t, int(v))] = LTT_DEFAULT

            for r, t, ltt in curs.execute('SELECT region, tech, lifetime FROM LifetimeTech').fetchall():
                for v in time_all:
                    lifetime_process[(r, t, int(v))] = int(ltt)

            for r, t, v, ltp in curs.execute('SELECT region, tech, vintage, lifetime FROM LifetimeProcess').fetchall():
                lifetime_process[(r, t, int(v))] = int(ltp)

            df_eff = pd.read_sql_query('SELECT * FROM Efficiency', conn)
            df_eff['vintage'] = pd.to_numeric(df_eff['vintage'], errors='coerce').fillna(0).astype(int)

            df_eff['last_out'] = [
                snap5_max2050(v + int(lifetime_process[r, t, v]))
                for r, t, v in df_eff[['region','tech','vintage']].itertuples(index=False, name=None)
            ]

            df_last_in = df_eff.groupby(['region','input_comm'], as_index=True)['last_out'].max().rename('last_in')
            df_eff = df_eff.merge(df_last_in, left_on=['region','output_comm'], right_index=True, how='left')
            df_eff['last_in'] = pd.to_numeric(df_eff['last_in'], errors='coerce').fillna(0).astype(int)

            demand_comms = {c[0] for c in curs.execute("SELECT name FROM Commodity WHERE flag = 'd'").fetchall()}
            df_eff = df_eff.loc[~df_eff['output_comm'].isin(demand_comms)].copy()
            
            df_remove = df_eff.loc[df_eff['last_in'] < df_eff['last_out']].copy()
            ritvo_remove = list(df_remove[['region','input_comm','tech','vintage','output_comm']].itertuples(index=False, name=None))

            if ritvo_remove:
                logger.debug(
                    "Removing the following ritvo orphans: %s",
                    ritvo_remove
                )

            deleted_total = 0
            for region, input_comm, tech, vintage, output_comm in ritvo_remove:

                if not global_settings.get("is_processing", True):
                    return
            
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

                    if not global_settings.get("is_processing", True):
                        return
            
                    curs.execute(
                        f"DELETE FROM {tbl} WHERE region = ? AND tech = ? AND CAST(vintage AS INTEGER) = ?",
                        (region, tech, int(vintage))
                    )
                    if curs.rowcount and curs.rowcount > 0: deleted_total += curs.rowcount

            conn.commit()
            if deleted_total == 0:
                logger.debug(
                    "Lifetime-aware supply-side orphan removal complete after %d iterations.",
                    iter
                )
                break

        if not global_settings.get("is_processing", True):
            return

        # Delete any unused commodities (techs already cleaned up)
        curs.execute(
            "DELETE FROM Commodity "
            "WHERE flag != 'e' "
                "AND name NOT IN (SELECT DISTINCT input_comm FROM Efficiency) "
                "AND name NOT IN (SELECT DISTINCT output_comm FROM Efficiency)"
        )
        conn.commit()

        curs.execute("PRAGMA foreign_keys = ON;")

        conn.commit()
        conn.close()

    except Exception as e:
        logger.exception("post processing failed for %s: %s", output_filename, e)
        raise RuntimeError(f"post processing failed for {output_filename}: {e}")
    
    finally:
        if conn:
            conn.close()