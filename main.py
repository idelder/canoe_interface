# -*- coding: utf-8 -*-
"""
CANOE UI Script
By David Turnbull
"""

from __future__ import annotations

import subprocess
import json
import os
import sys
import atexit
from typing import Any, Dict, Set, Tuple
import flet as ft
from time import sleep
from itertools import product
from log_setup import setup_logging
import database_processing as dbp

from constants import (
    Region,
    Sector,
    Variant,
    Feature,
    Level,
    LOW_SCENARIOS,
    DEFAULT_LOW,
    TABLE_REGIONS,
    TABLE_SECTORS,
)
from directories import CONFIG_FILE, ASSETS_DIR

# Get logger for this module
logger = setup_logging("main")

#########################################
# Helpers
#########################################

# SECTVARR1R2
def make_id(
    sector: Sector,
    variant: Variant,
    subvariant: Feature,
    region_1: Region,
    region_2: Region = Region.NONE,
) -> str:
    return f"{sector.value}{variant.value}{subvariant.value}{region_1.value}{region_2.value}"

#########################################
# Initialisation
#########################################

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

#########################################
# Desired IDs Builder
#########################################

def rs_active(
        matrix: Dict[Tuple[str, str], ft.Dropdown],
        r: str,
        s: str
) -> bool:
    if all((
        dd := matrix.get((r, s)),
        not dd.disabled,
        str(dd.value) != Level.EXCLUDED.value
    )):
        return dd
    else:
        return False

def get_matrix_selection(
    matrix: Dict[Tuple[str, str], ft.Dropdown],
    global_settings: Dict[str, Any],
) -> Set[Sector, Variant, Feature, Region, Region]:
    """
    1. Get the active region-sector combos and their generic (non-regional) variant
    2. If only electricity in a region, add grid demand
    3. Add endogenous and boundary interties based on selected regions
    4. Add fuels for all active regions
    5. Add agriculture for active regions where anything but electricity is present
    """

    # Get the low-res scenario
    low_scenario = global_settings.get("low_scenario", DEFAULT_LOW)

    selections: Set[Sector, Variant, Feature, Region, Region] = set()
    regions: Set[Region] = set()
    for r in TABLE_REGIONS:
        for s in TABLE_SECTORS:
            if not (dd := rs_active(matrix, r, s)):
                continue

            if str(dd.value) == Level.HIGH_RESOLUTION.value:
                v = Variant.HIGH_RESOLUTION
            elif low_scenario == "Current Measures":
                v = Variant.CURRENT_MEASURES
            else:
                v = Variant.NET_ZERO

            regions.add(Region(r))

            # Add variant-region data and variant-generic (no region) data
            selections.add((
                Sector(s),
                v,
                Feature.NONE,
                Region(r),
                Region.NONE,
            ))
            selections.add((
                Sector(s),
                v,
                Feature.NONE,
                Region.NONE,
                Region.NONE,
            ))

            # Special case: if only Elc is active for a region, add electricity demand
            if s == Sector.ELECTRICITY.value and not any((
                rs_active(matrix, r, _s) for _s in TABLE_SECTORS if _s != s
            )):
                selections.add((
                    Sector.ELECTRICITY,
                    v,
                    Feature.DEMAND,
                    Region(r),
                    Region.NONE,
                ))

        # Add agriculture if any sector other than electricity active in this region
        if any((
            rs_active(matrix, r, _s)
            for _s in TABLE_SECTORS if _s != Sector.ELECTRICITY.value
        )):
            # Generic non-regional components
            selections.add((
                Sector.AGRICULTURE,
                Variant.HIGH_RESOLUTION,
                Feature.NONE,
                Region.NONE,
                Region.NONE
            ))
            # Regional components
            selections.add((
                Sector.AGRICULTURE,
                Variant.HIGH_RESOLUTION,
                Feature.NONE,
                Region(r),
                Region.NONE
            ))
                
    # Endogenous intertie only if both regions selected
    region_combos = {
        tuple(sorted((r1, r2)))
        for r1 in regions
        for r2 in regions if r1 != r2
    }
    for r1, r2 in region_combos:
        selections.add((
            Sector.ELECTRICITY,
            Variant.HIGH_RESOLUTION,
            Feature.ENDOGENOUS_INTERTIE,
            r1,
            r2,
        ))

    # Boundary intertie only if origin region selected and dest not selected
    region_combos = {
        (r1, r2)
        for r1 in regions
        for r2 in Region if r1 != r2
    }
    for r1, r2 in region_combos:
        if r2 not in regions:
            selections.add((
                Sector.ELECTRICITY,
                Variant.HIGH_RESOLUTION,
                Feature.BOUNDARY_INTERTIE,
                r1,
                r2,
            ))

    # Fuels generic non-regional components always
    selections.add((
            Sector.FUEL,
            Variant.HIGH_RESOLUTION,
            Feature.NONE,
            Region.NONE,
            Region.NONE
    ))
    # Fuels for all active regions
    for r in regions:
        selections.add((
            Sector.FUEL,
            Variant.HIGH_RESOLUTION,
            Feature.NONE,
            r,
            Region.NONE
        ))
                
    logger.debug('Selections for aggregation:')
    desired_ids: Set[str] = set()
    for sector, variant, subvariant, region_1, region_2 in sorted(selections):
        logger.debug(
            "Sector=%s Variant=%s SubVariant=%s Region1=%s Region2=%s",
            sector.value, variant.value, subvariant.value, region_1.value, region_2.value
        )
        desired_ids.add(
            make_id(
                sector,
                variant,
                subvariant,
                region_1,
                region_2
            )
        )

    return desired_ids


def infer_selected_regions_from_matrix(matrix: Dict[Tuple[str, str], ft.Dropdown]) -> Set[str]:
    """Regions that have any sector set to something other than 'NA'."""
    picked: Set[str] = set()
    for r in TABLE_REGIONS:
        if any(
            (dd := matrix.get((r, s))) is not None and str(dd.value).upper() != Level.EXCLUDED.value
            for s in TABLE_SECTORS
        ):
            picked.add(r)
    return picked

#########################################
# Flet UI
#########################################

def main(page: ft.Page) -> None:
    page.title = "CANOE UI"
    page.vertical_alignment = ft.CrossAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    if sys.platform != "darwin":
        icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.ico")
        page.window.icon = icon_path
    page.window_width = 1200
    page.window_height = 700

    # UI storage
    matrix: Dict[Tuple[str, str], ft.Dropdown] = {}

    # Global flags
    global_settings = {
        "low_scenario": DEFAULT_LOW,
        "power_system_model": False,
        "is_processing": False
    }

    # Stop everything if window closed
    def on_window_event(e: ft.WindowEvent):
        if (
            getattr(e, "type", None) == "close_request"
            or getattr(e, "data", None) in ("close", "close_request")
        ):  
            status_text.value = "Exiting..."
            page.update()
            sleep(0.05)
            global_settings['is_processing'] = False
            logger.info("Processing cancelled due to window close")
            page.window.destroy()

    page.window.prevent_close = True
    page.window.on_event = on_window_event

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
    
    image = ft.Container(
        content=ft.Image(
            src=os.path.join("assets", "logo.png"),
            height=50,
            width=60
        ),
        alignment=ft.alignment.top_right
    )
    
    # load saved config (will be applied once UI elements are created)
    saved_cfg = load_config()

    def save_config() -> None:
        """Persist current UI state to CONFIG_FILE (JSON)."""
        try:
            cfg = {
                "input_filename": os.path.normpath(
                    (in_filename_text_field.value or "").strip().strip("'").strip('"')
                ),
                "output_filename": os.path.normpath(
                    (out_filename_text_field.value or "").strip().strip("'").strip('"')
                ),
                "low_scenario": (scenario_dropdown.value or DEFAULT_LOW),
                "power_system_model": bool(power_system_checkbox.value), # RE-ADDED
                "matrix": {
                    f"{r}|{s}": (
                        matrix[(r, s)].value
                        if (r, s) in matrix and matrix[(r, s)] is not None
                        else None
                    )
                    for (r, s) in matrix.keys()
                },
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
        is_na_selected = (e.control.value == Level.EXCLUDED.value)

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
                        dd.value = Level.EXCLUDED.value
                else:
                    # if demand is active in this region cant have an inactive demand sector
                    row_is_active = any(
                        (matrix.get((r, _s))
                        and matrix[(r, _s)].value != Level.EXCLUDED.value)
                        for _s in TABLE_SECTORS if _s != Sector.ELECTRICITY.value
                    )
                    if row_is_active:
                        e.control.value = Level.LOW_RESOLUTION.value
                        e.control.update()
            elif s != Sector.ELECTRICITY.value:
                # If any sector other than elc activated, all must be active. Default to low-res
                for _s in TABLE_SECTORS:
                    if _s == s: continue
                    dd = matrix.get((r, _s))
                    if not dd:
                        continue

                    if _s != Sector.ELECTRICITY.value and dd.value == Level.EXCLUDED.value:
                        dd.value = e.control.value
                    elif _s == Sector.ELECTRICITY.value and dd.value == Level.EXCLUDED.value:
                        dd.value = Level.HIGH_RESOLUTION.value
        
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
            dd.value = Level.EXCLUDED.value
    
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
    in_filename_text_field.on_change = lambda e: save_config
    out_filename_text_field.on_change = lambda e: save_config

    def on_submit(e: ft.ControlEvent) -> None:

        if global_settings["is_processing"]:
            # Cancel button action
            global_settings["is_processing"] = False
            logger.info("Processing cancelled by user")
            submit_button.text = "Submit"
            status_text.value = "Processing cancelled."
            page.update()
            return
        
        global_settings["is_processing"] = True
        submit_button.text = "Cancel"
        status_text.value = "Processing..."
        page.update()

        input_filename = in_filename_text_field.value.strip().strip("'").strip('"')
        if not input_filename:
            status_text.value = "Error: Filename cannot be empty."
            page.update()
            return

        output_filename = out_filename_text_field.value.strip().strip("'").strip('"')
        if not output_filename:
            status_text.value = "Error: Filename cannot be empty."
            page.update()
            return

        try:
            # persist current config before running
            save_config()
            desired_ids = get_matrix_selection(matrix, global_settings)
            # Send to processing module
            dbp.aggregate_sqlite_files(
                input_filename = input_filename,
                output_filename = output_filename,
                global_settings = global_settings,
                desired_ids = desired_ids,
            )
            if global_settings.get("is_processing", True):
                status_text.value = f"Aggregation complete. Output: {output_filename}"
                logger.info("Aggregation finished successfully (input=%s output=%s)", input_filename, output_filename)
        except Exception as ex:
            logger.exception("Aggregation failed (input=%s output=%s): %s", input_filename, output_filename, ex)
            status_text.value = f"Error processing database. Check log file."
        finally:
            # Reset processing state
            global_settings["is_processing"] = False
            submit_button.text = "Submit"
            page.update()
            
        page.update()


    # --- Build initial matrix UI (all regions from the start) ---
    header_row = [ft.Text("Region/Sector", weight=ft.FontWeight.BOLD, width=120)] + [
        ft.Text(s, weight=ft.FontWeight.BOLD, text_align=ft.TextAlign.CENTER, width=100) for s in TABLE_SECTORS
    ]
    matrix_rows = [ft.Row(header_row, alignment=ft.MainAxisAlignment.CENTER)]

    for region in TABLE_REGIONS:
        row_elements = [ft.Text(region, weight=ft.FontWeight.BOLD, width=120)]
        for sector in TABLE_SECTORS:

            # Update matrix values
            key = f"{region}|{sector}"
            val = Level.EXCLUDED.value
            if saved_cfg and saved_cfg.get("matrix") and key in saved_cfg["matrix"]:
                try:
                    val = saved_cfg["matrix"].get(key)
                    if val is None:
                        val = Level.EXCLUDED.value
                except Exception as e:
                    logger.exception("Failed applying saved value for %s|%s: %s", region, sector, e)
                    raise RuntimeError(f"Failed applying saved value for {region}|{sector}: {e}")

            if sector == Sector.ELECTRICITY.value:
                level_options = [
                    Level.HIGH_RESOLUTION,
                    Level.EXCLUDED
                ]
            else:
                level_options = Level
            # All sectors use same options
            dd = ft.Dropdown(
                options=[ft.dropdown.Option(l.value) for l in level_options],
                value=val,
                width=100,
                content_padding=ft.padding.only(left=5, right=5),
                disabled=False,
            )
            matrix[(region, sector)] = dd
            row_elements.append(dd)
            dd.data = (region, sector)
            dd.on_change = on_sector_dropdown_change
        matrix_rows.append(ft.Row(row_elements, alignment=ft.MainAxisAlignment.CENTER))

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
        logger.exception("Failed applying saved configuration on load")
        pass

    # Controls
    reset_button = ft.ElevatedButton("Reset Matrix", on_click=reset_matrix, width=200, height=40)
    submit_button = ft.ElevatedButton("Submit", on_click=on_submit, width=80, height=40)

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
    middle_row = ft.Row(
        [
            ft.Column(
                matrix_rows,
                expand=True,                      # take remaining width in the row
                scroll=ft.ScrollMode.ADAPTIVE,    # or ALWAYS
                # remove fixed height here
                horizontal_alignment=ft.CrossAxisAlignment.START,
            ),
            ft.VerticalDivider(),
            settings_column,  # give this a fixed width internally if needed
        ],
        expand=True,  # <-- this row grows vertically to fill space
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START,
    )

    footer = ft.Container(
        height=50,  # <-- fixed footer height
        content=ft.Row(
            [in_filename_text_field, out_filename_text_field, submit_button, status_text],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=30,
        ),
        alignment=ft.alignment.center,
    )

    main_content = ft.Column(
        [
            image,
            ft.Text("CANOE RSS Selector", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Read accompanying document for details about choices and instructions", size=18),
            ft.Divider(),
            middle_row,  # expands
            ft.Divider(),
            footer,      # fixed height
        ],
        expand=True,  # <-- make the whole layout fill the window
        alignment=ft.MainAxisAlignment.START,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=10,
    )

    page.add(main_content)
    page.update()

    # Build matrix from saved state first, then enforce disabling rules
    update_ui_matrix()

if __name__ == "__main__":
    ft.app(target=main, assets_dir=ASSETS_DIR)