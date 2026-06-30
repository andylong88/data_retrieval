"""
Download groundwater level data and site information from Oregon Water
Resources Department (OWRD) hydrograph page using Selenium.

Downloads both discrete measurements and daily (continuous/recorder) data
by extracting all records directly from the w2ui JavaScript grids.  Also
scrapes site metadata (location, depth, aquifer, etc.) from the page header.

Reads a list of well IDs (gw_logid) from an external CSV file and downloads
data for each site.

Usage:
    python owrd_well_level_data.py

Requirements:
    conda install -c conda-forge selenium pandas lxml
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import json
import time
import re
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SITE_LIST_CSV = os.path.join(SCRIPT_DIR, "script_input\WallaWalla_GW_OWRD_sites.csv")  # CSV with a 'gw_logid' column
SITE_LIST_COLUMN = "gw_logid"  # Column name containing well IDs
BASE_URL = "https://apps.wrd.state.or.us/apps/gw/gw_info/gw_hydrograph/Hydrograph.aspx"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "downloaded")
HEADLESS = True  # Set False to see the browser window for debugging


def create_driver(headless=False):
    """Create and configure a Chrome WebDriver instance."""
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    return driver


def extract_w2ui_grid_data(driver, grid_name):
    """
    Extract all records from a w2ui grid by accessing the JavaScript object.

    Parameters
    ----------
    driver : webdriver.Chrome
        Active Selenium driver with the page loaded.
    grid_name : str
        Name of the w2ui grid (e.g., 'water_level_grid', 'daily_water_level_grid').

    Returns
    -------
    pd.DataFrame or None
        DataFrame with all records from the grid.
    """
    try:
        # Get record count
        count = driver.execute_script(
            f"return w2ui['{grid_name}'].records ? w2ui['{grid_name}'].records.length : 0;"
        )
        if not count:
            print(f"  Grid '{grid_name}' has no records.")
            return None

        print(f"  Grid '{grid_name}' contains {count} records.")

        # Extract all records as JSON
        all_data = driver.execute_script(
            f"return JSON.stringify(w2ui['{grid_name}'].records);"
        )
        records = json.loads(all_data)

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Clean up internal w2ui columns
        drop_cols = [c for c in df.columns if c.startswith("w2ui") or c == "recid"]
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    except Exception as e:
        print(f"  Error extracting grid '{grid_name}': {e}")
        return None


def extract_site_info(driver, gw_logid):
    """
    Extract key site metadata from the OWRD hydrograph page header table.

    The page renders site info in an HTML table with label/value cells.
    This function reads the table cells directly via JavaScript.

    Parameters
    ----------
    driver : webdriver.Chrome
        Active Selenium driver with the hydrograph page loaded.
    gw_logid : str
        The groundwater log ID.

    Returns
    -------
    dict
        Dictionary with gw_logid plus extracted fields.
    """
    try:
        info = {"gw_logid": gw_logid}

        # Extract all table cell text as a flat list of label/value pairs.
        # The site info table has cells where labels end with ":" and the
        # next cell contains the value.
        all_pairs = driver.execute_script("""
            var pairs = {};
            var tables = document.querySelectorAll('table');
            for (var t = 0; t < tables.length; t++) {
                var cells = tables[t].querySelectorAll('td, th');
                for (var i = 0; i < cells.length; i++) {
                    var text = (cells[i].textContent || '').trim();
                    if (text.endsWith(':') && i + 1 < cells.length) {
                        var label = text.slice(0, -1).trim();
                        var value = (cells[i+1].textContent || '').trim();
                        if (label && !value.endsWith(':')) {
                            pairs[label] = value;
                        }
                    }
                }
            }
            return pairs;
        """)

        if all_pairs:
            print(f"  Extracted {len(all_pairs)} fields from page table:")
            for label, value in all_pairs.items():
                print(f"    '{label}' = '{value}'")
        else:
            print("  Warning: No table data extracted.")

        # Include all extracted fields, using cleaned-up column names.
        # Convert labels to snake_case-style keys for CSV friendliness.
        label_to_key = {
            "Well Location": "well_location_trs",
            "Total Depth (bls)": "total_depth_ft",
            "Water Level Count": "wl_count",
            "Log ID": "log_id",
            "Land Surface Elevation": "land_surface_elevation_ft",
            "Wtr Lvl Date Range": "wl_date_range",
            "Well Tag": "well_tag",
            "Vertical Reference Datum": "vertical_datum",
            "Wtr Lvl Depth Min-Max": "wl_depth_min_max_ft",
            "State Observation": "state_observation_id",
            "Primary Use of Well": "primary_use",
            "Recorder Wtr Lvl Count": "recorder_wl_count",
            "USGS Site": "usgs_site",
            "Primary Aquifer System": "aquifer",
            "Recorder Wtr Lvl Date Range": "recorder_wl_date_range",
            "More information": "more_info",
            "Groundwater Mapping Tool": "gw_mapping_tool",
            "Recorder Wtr Lvl Depth Min-Max": "recorder_wl_depth_min_max_ft",
        }

        for label, value in all_pairs.items():
            if not value:
                continue
            key = label_to_key.get(label)
            if key is None:
                # Auto-generate a key from the label
                key = re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')
            # Clean numeric suffixes like "1001 ft" -> "1001"
            if key.endswith("_ft") and value:
                m = re.search(r'([0-9,.]+)', value)
                if m:
                    value = m.group(1).replace(",", "")
            info[key] = value

        # Try to extract lat/lon from page source (JS variables or hidden data)
        try:
            page_source = driver.page_source
            lat_m = re.search(
                r'["\']?lat(?:itude)?["\']?\s*[:=]\s*["\']?([0-9]{2}\.[0-9]{3,})',
                page_source, re.IGNORECASE
            )
            lon_m = re.search(
                r'["\']?lon(?:g(?:itude)?)?["\']?\s*[:=]\s*["\']?(-?[0-9]{2,3}\.[0-9]{3,})',
                page_source, re.IGNORECASE
            )
            if lat_m:
                info["latitude"] = lat_m.group(1)
            if lon_m:
                info["longitude"] = lon_m.group(1)
        except Exception:
            pass

        return info

    except Exception as e:
        print(f"  Error extracting site info: {e}")
        return {"gw_logid": gw_logid, "error": str(e)}


def fetch_all_data(gw_logid, headless=True):
    """
    Navigate to the OWRD hydrograph page and extract both discrete and daily
    water level data from the w2ui JavaScript grids, plus site metadata.

    Parameters
    ----------
    gw_logid : str
        The groundwater log ID (e.g., 'UMAT0005065').
    headless : bool
        Run browser without a visible window.

    Returns
    -------
    tuple of (pd.DataFrame or None, pd.DataFrame or None, dict or None)
        (discrete_df, daily_df, site_info_dict)
    """
    url = f"{BASE_URL}?gw_logid={gw_logid}"
    print(f"Loading page: {url}")

    driver = create_driver(headless=headless)

    try:
        driver.get(url)

        # Wait for the w2ui grids to initialize
        WebDriverWait(driver, 20).until(
            lambda d: d.execute_script("return typeof w2ui !== 'undefined' && Object.keys(w2ui).length > 0;")
        )
        # Give grids time to fully populate with data
        time.sleep(5)

        # Verify w2ui grids are available
        grid_names = driver.execute_script("return Object.keys(w2ui);")
        print(f"  Available w2ui grids: {grid_names}")

        # Extract site information from page header
        print("\nExtracting site information...")
        site_info = extract_site_info(driver, gw_logid)

        # Click the "Daily Water Level" tab to ensure that grid loads its data
        try:
            daily_tab = driver.find_element(By.ID, "ui-id-2")
            daily_tab.click()
            time.sleep(3)
        except Exception:
            print("  Note: Could not click Daily Water Level tab (may already be loaded).")

        # Extract discrete water level data
        print("\nExtracting discrete water level data...")
        df_discrete = extract_w2ui_grid_data(driver, "water_level_grid")

        # Extract daily water level data
        print("\nExtracting daily water level data...")
        df_daily = extract_w2ui_grid_data(driver, "daily_water_level_grid")

        return df_discrete, df_daily, site_info

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    finally:
        driver.quit()
        print("\n  Browser closed.")


def clean_discrete_data(df):
    """Clean and format the discrete water level DataFrame."""
    if df is None:
        return None

    # Rename columns to be more readable
    rename_map = {
        "gw_logid": "well_id",
        "measured_date": "date",
        "measured_time": "time",
        "waterlevel_ft_below_land_surface": "wl_ft_below_land_surface",
        "waterlevel_ft_above_mean_sea_level": "wl_ft_above_msl",
        "measurement_source_organization": "source_org",
        "measurement_source_owrd": "source_owrd",
        "method_of_water_level_measurement": "measurement_method",
        "measurement_status_desc": "measurement_status",
        "measuring_point_height": "measuring_point_height_ft",
        "waterlevel_accuracy": "accuracy_ft",
        "reviewed_status_desc": "review_status",
    }
    df = df.rename(columns=rename_map)

    # Parse date and sort
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "measured_date_" in df.columns:
        df = df.drop(columns=["measured_date_"], errors="ignore")

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


def clean_daily_data(df):
    """Clean and format the daily water level DataFrame."""
    if df is None:
        return None

    # Rename columns
    rename_map = {
        "gw_logid": "well_id",
        "record_date": "date",
        "waterlevel_ft_below_land_surface": "wl_ft_below_land_surface",
        "waterlevel_ft_above_mean_sea_level": "wl_ft_above_msl",
        "method_of_water_level_measurement": "measurement_method",
        "waterlevel_accuracy": "accuracy_ft",
        "reviewed_status": "review_status",
    }
    df = df.rename(columns=rename_map)

    # Parse date and sort
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    if "record_date_" in df.columns:
        df = df.drop(columns=["record_date_"], errors="ignore")

    df = df.sort_values("date", ascending=True).reset_index(drop=True)
    return df


def save_data(df, output_dir, filename):
    """Save DataFrame to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved to: {filepath}")
    return filepath


def load_site_list(csv_path, column_name):
    """
    Load well IDs from an external CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    column_name : str
        Column name containing the well IDs.

    Returns
    -------
    list of str
        List of gw_logid values.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Site list CSV not found: {csv_path}\n"
            f"Create a CSV file with a '{column_name}' column."
        )

    df = pd.read_csv(csv_path)

    if column_name not in df.columns:
        raise ValueError(
            f"Column '{column_name}' not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    sites = df[column_name].dropna().astype(str).str.strip().tolist()
    sites = [s for s in sites if s]  # Remove empty strings

    if not sites:
        raise ValueError(f"No site IDs found in column '{column_name}' of {csv_path}.")

    return sites


def main():
    print(f"{'='*60}")
    print(f"OWRD Groundwater Level Data Retrieval")
    print(f"{'='*60}")

    # Load site list from CSV
    print(f"\nReading site list from: {SITE_LIST_CSV}")
    sites = load_site_list(SITE_LIST_CSV, SITE_LIST_COLUMN)
    print(f"  Found {len(sites)} site(s): {sites}\n")

    # Track results
    results = []
    all_site_info = []

    for i, gw_logid in enumerate(sites, start=1):
        print(f"\n{'#'*60}")
        print(f"  Site {i}/{len(sites)}: {gw_logid}")
        print(f"{'#'*60}")

        df_discrete, df_daily, site_info = fetch_all_data(gw_logid, headless=HEADLESS)

        # Collect site info
        if site_info:
            all_site_info.append(site_info)
            print(f"\n  Site info fields: {[k for k in site_info.keys() if not k.startswith('_')]}")

        # Process discrete data
        n_discrete = 0
        if df_discrete is not None:
            df_discrete = clean_discrete_data(df_discrete)
            n_discrete = len(df_discrete)
            print(f"\n  Discrete: {n_discrete} records", end="")
            if n_discrete > 0:
                print(f" ({df_discrete['date'].min()} to {df_discrete['date'].max()})")
                save_data(df_discrete, OUTPUT_DIR, f"OWRD_{gw_logid}_discrete_WL.csv")
            else:
                print()
        else:
            print(f"\n  Discrete: no data")

        # Process daily data
        n_daily = 0
        if df_daily is not None:
            df_daily = clean_daily_data(df_daily)
            n_daily = len(df_daily)
            print(f"  Daily:    {n_daily} records", end="")
            if n_daily > 0:
                print(f" ({df_daily['date'].min()} to {df_daily['date'].max()})")
                save_data(df_daily, OUTPUT_DIR, f"OWRD_{gw_logid}_daily_WL.csv")
            else:
                print()
        else:
            print(f"  Daily:    no data")

        results.append({
            "gw_logid": gw_logid,
            "discrete_records": n_discrete,
            "daily_records": n_daily,
        })

    # --- Save site information ---
    if all_site_info:
        df_site_info = pd.DataFrame(all_site_info)
        # Keep only the desired output columns (in order)
        output_columns = [
            "gw_logid",
            "land_surface_elevation_ft",
            "log_id",
            "more_info",
            "aquifer",
            "primary_use",
            "recorder_wl_count",
            "recorder_wl_date_range",
            "total_depth_ft",
            "vertical_datum",
            "wl_count",
            "well_location_trs",
            "well_tag",
            "wl_date_range",
        ]
        # Only include columns that actually exist in the data
        cols_present = [c for c in output_columns if c in df_site_info.columns]
        df_site_info = df_site_info[cols_present]
        save_data(df_site_info, OUTPUT_DIR, "OWRD_site_info.csv")
        print(f"\n  Site info saved for {len(all_site_info)} site(s).")
    else:
        print("\n  No site information was retrieved.")

    # --- Final Summary ---
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"{'Site':<20} {'Discrete':>10} {'Daily':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*10}")
    for r in results:
        print(f"{r['gw_logid']:<20} {r['discrete_records']:>10} {r['daily_records']:>10}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
