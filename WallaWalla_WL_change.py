"""
Walla Walla Water Level Change

Plots the change in water level for both Washington (USGS) and Oregon (OWRD)
monitoring wells on a single figure. All hydrographs start at zero, showing the
change from the initial water level at the start of the plot period.

Positive values indicate a rise in water level; negative values indicate a decline.

Produces two plots:
  1. Raw daily data
  2. LOESS-smoothed data
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import glob
from pathlib import Path

# Define plot time window
t_start = pd.Timestamp('2022-07-01')
t_end = pd.Timestamp('2024-08-01')

# Path to downloaded data
data_dir = Path('downloaded')

# Output directory for plots
plot_dir = Path('plots')
plot_dir.mkdir(exist_ok=True)

# Plot options
PLOT_SD_CONTOURS = False  # Set to True to enable SD interpolation color flood on SD map
PLOT_SD_BUBBLES  = True   # Set to False to disable SD-scaled open circles on SD map

# =============================================================================
# Load Washington (USGS) Data
# =============================================================================
usgs_frames = []
for f in sorted(data_dir.glob('USGS*.csv')):
    df = pd.read_csv(f)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    usgs_frames.append(df)

df_usgs = pd.concat(usgs_frames, ignore_index=True)

# Read site info for labels
df_site = pd.read_csv(data_dir / 'site_info.csv')
name_lookup = df_site.set_index('monitoring_location_id')['monitoring_location_name'].to_dict()
usgs_label_lookup = {site: name.split('-')[-1][-5:] for site, name in name_lookup.items()}

# Resample USGS hourly data to daily means (strip timezone for compatibility)
df_usgs['date'] = df_usgs['time'].dt.normalize().dt.tz_localize(None)
df_usgs_daily = (
    df_usgs.groupby(['monitoring_location_id', 'date'])['value']
    .mean()
    .reset_index()
)

# Filter to plot window
df_usgs_daily = df_usgs_daily[
    (df_usgs_daily['date'] >= t_start) & (df_usgs_daily['date'] <= t_end)
].copy()

print(f"USGS sites: {df_usgs_daily['monitoring_location_id'].nunique()}")
print(f"  Date range: {df_usgs_daily['date'].min().date()} to {df_usgs_daily['date'].max().date()}")

# =============================================================================
# Load Oregon (OWRD) Data
# =============================================================================
owrd_files = sorted(glob.glob(str(data_dir / 'OWRD_*_daily_WL.csv')))

owrd_frames = []
for filepath in owrd_files:
    df = pd.read_csv(filepath, parse_dates=['date'])
    owrd_frames.append(df)

df_owrd = pd.concat(owrd_frames, ignore_index=True)

# Filter to plot window
df_owrd = df_owrd[
    (df_owrd['date'] >= t_start) & (df_owrd['date'] <= t_end)
].copy()

print(f"OWRD sites: {df_owrd['well_id'].nunique()}")
print(f"  Date range: {df_owrd['date'].min().date()} to {df_owrd['date'].max().date()}")

# =============================================================================
# Compute Change from Initial Water Level
# =============================================================================
# USGS: 'value' is depth to water (ft below land surface)
# A decrease in depth means water level rose, so change = -(value - initial_value)
usgs_change_data = []
for site, group in df_usgs_daily.groupby('monitoring_location_id'):
    group = group.sort_values('date').copy()
    initial_value = group['value'].iloc[0]
    group['wl_change_ft'] = -(group['value'] - initial_value)
    usgs_change_data.append(group)

df_usgs_change = pd.concat(usgs_change_data, ignore_index=True)

# OWRD: 'wl_ft_below_land_surface' is depth to water
owrd_change_data = []
for well_id, group in df_owrd.groupby('well_id'):
    group = group.sort_values('date').copy()
    initial_value = group['wl_ft_below_land_surface'].iloc[0]
    group['wl_change_ft'] = -(group['wl_ft_below_land_surface'] - initial_value)
    owrd_change_data.append(group)

df_owrd_change = pd.concat(owrd_change_data, ignore_index=True)

print("Change from initial WL computed for all sites.")

# LOESS smoothing fraction (0 to 1; larger = smoother)
loess_frac = 0.1

# =============================================================================
# Plot 3: Well Location Map
# =============================================================================
import re


def parse_wkt_point(wkt_str):
    """Extract (lon, lat) from a WKT POINT string like 'POINT (-118.47 46.00)'."""
    m = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(wkt_str))
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def trs_to_latlon(trs_str):
    """
    Convert an Oregon TRS string (Willamette Meridian) to approximate lat/lon.

    TRS format example: '5.00N/34.00E-16DCC'
      - Township 5 North, Range 34 East, Section 16, quarter DCC

    Willamette Meridian reference:
      - Baseline (T0): ~45.5236 N latitude
      - Principal Meridian (R0): ~122.7633 W longitude
      - Each township/range ~ 6 miles
      - 1 degree latitude ~ 69 miles -> 6 miles ~ 0.08696 degrees
      - 1 degree longitude at 46N ~ 48 miles -> 6 miles ~ 0.125 degrees
    """
    if not trs_str or pd.isna(trs_str):
        return None, None

    m = re.match(r'(\d+\.?\d*)([NS])/(\d+\.?\d*)([EW])-(\d+)', str(trs_str))
    if not m:
        return None, None

    township = float(m.group(1))
    t_dir = m.group(2)
    range_num = float(m.group(3))
    r_dir = m.group(4)
    section = int(m.group(5))

    # Willamette Meridian reference point
    base_lat = 45.5236  # baseline latitude
    base_lon = -122.7633  # principal meridian longitude

    # Degrees per township/range
    deg_per_township = 6.0 / 69.0  # ~0.08696
    deg_per_range = 6.0 / 48.0  # ~0.125 at ~46N

    # Compute township center latitude
    if t_dir == 'N':
        lat = base_lat + (township - 0.5) * deg_per_township
    else:
        lat = base_lat - (township - 0.5) * deg_per_township

    # Compute range center longitude
    if r_dir == 'E':
        lon = base_lon + (range_num - 0.5) * deg_per_range
    else:
        lon = base_lon - (range_num - 0.5) * deg_per_range

    # Adjust for section within township (6x6 grid, numbered serpentine)
    # Row and column of section (approximate)
    # Sections numbered 1-36 in a serpentine pattern starting top-right
    row = (section - 1) // 6  # 0-5, top to bottom
    col_in_row = (section - 1) % 6
    if row % 2 == 0:
        col = 5 - col_in_row  # even rows go right to left
    else:
        col = col_in_row  # odd rows go left to right

    # Each section is ~1 mile square = 1/6 of a township
    section_deg_lat = deg_per_township / 6.0
    section_deg_lon = deg_per_range / 6.0

    # Offset from township center to section center
    lat += (2.5 - row) * section_deg_lat
    lon += (col - 2.5) * section_deg_lon

    return lon, lat


# --- Get USGS well coordinates ---
usgs_coords = []
for _, row in df_site.iterrows():
    lon, lat = parse_wkt_point(row['geometry'])
    if lon is not None:
        # Classify aquifer: basalt vs basin-fill
        aquifer_code = str(row.get('aquifer_code', ''))
        # Known basalt codes: 122SDLM (Saddle Mtns), 122GDRD (Grande Ronde)
        if aquifer_code.startswith('122'):
            aquifer_type = 'basalt'
        else:
            aquifer_type = 'basin-fill'
        usgs_coords.append({
            'site_id': row['monitoring_location_id'],
            'label': usgs_label_lookup.get(row['monitoring_location_id'],
                                           row['monitoring_location_id']),
            'lon': lon,
            'lat': lat,
            'state': 'WA',
            'aquifer_type': aquifer_type
        })

df_usgs_coords = pd.DataFrame(usgs_coords)

# --- Get OWRD well coordinates from GWIS site file ---
df_gwis = pd.read_csv(Path('script_input') / 'GWIS_sites20250903.csv')
df_owrd_site = pd.read_csv(data_dir / 'OWRD_site_info.csv')

owrd_coords = []
for _, row in df_owrd_site.iterrows():
    gw_logid = row['gw_logid']
    # Look up coordinates in GWIS file
    match = df_gwis[df_gwis['gw_logid'] == gw_logid]
    if not match.empty and pd.notna(match.iloc[0]['latitude_d']) and pd.notna(match.iloc[0]['longitude_']):
        lat = match.iloc[0]['latitude_d']
        lon = match.iloc[0]['longitude_']
    else:
        # Fall back to TRS conversion
        lon, lat = trs_to_latlon(row['well_location_trs'])

    if lon is not None and lat is not None:
        # Classify aquifer: basalt vs basin-fill (sediment)
        aquifer = str(row.get('aquifer', '')).lower()
        if 'basalt' in aquifer:
            aquifer_type = 'basalt'
        else:
            aquifer_type = 'basin-fill'
        owrd_coords.append({
            'site_id': gw_logid,
            'label': gw_logid,
            'lon': float(lon),
            'lat': float(lat),
            'state': 'OR',
            'aquifer_type': aquifer_type
        })

df_owrd_coords = pd.DataFrame(owrd_coords)

# --- Combine and plot ---
df_all_coords = pd.concat([df_usgs_coords, df_owrd_coords], ignore_index=True)

import contextily as cx
from pyproj import Transformer

# --- Compute standard deviation of WL change for each well ---
usgs_sd = (
    df_usgs_change.groupby('monitoring_location_id')['wl_change_ft']
    .std()
    .rename('sd_ft')
    .reset_index()
    .rename(columns={'monitoring_location_id': 'site_id'})
)
owrd_sd = (
    df_owrd_change.groupby('well_id')['wl_change_ft']
    .std()
    .rename('sd_ft')
    .reset_index()
    .rename(columns={'well_id': 'site_id'})
)
df_sd = pd.concat([usgs_sd, owrd_sd], ignore_index=True)

# Merge SD into coordinates
df_all_coords = df_all_coords.merge(df_sd, on='site_id', how='left')

# Transform coordinates from EPSG:4326 (lon/lat) to EPSG:2286 (WA State Plane South, ft)
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:2286', always_xy=True)
x_proj, y_proj = transformer.transform(
    df_all_coords['lon'].values, df_all_coords['lat'].values
)
df_all_coords['x'] = x_proj
df_all_coords['y'] = y_proj

# Separate by aquifer type
basin_fill = df_all_coords[df_all_coords['aquifer_type'] == 'basin-fill']
basalt = df_all_coords[df_all_coords['aquifer_type'] == 'basalt']

# --- Plot 3a: Map with Site IDs only ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(basin_fill['x'], basin_fill['y'], c='blue', s=80, marker='o',
           label='Basin-fill', zorder=5, edgecolors='black', linewidths=0.5)
ax.scatter(basalt['x'], basalt['y'], c='red', s=25, marker='s',
           label='Basalt', zorder=6, edgecolors='black', linewidths=0.5)

# SD-scaled open circles (optional)
if PLOT_SD_BUBBLES:
    # Only include wells with valid SD values
    valid_bubble = df_all_coords.dropna(subset=['sd_ft'])
    sd_vals = valid_bubble['sd_ft']
    sd_min, sd_max = sd_vals.min(), sd_vals.max()
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0
    bubble_sizes = 100 + ((sd_vals - sd_min) / sd_range) * 1100
    ax.scatter(valid_bubble['x'], valid_bubble['y'],
               s=bubble_sizes, facecolors='none', edgecolors='black',
               linewidths=1.0, zorder=7, label='SD magnitude')

from adjustText import adjust_text

texts_3a = []
for _, row in df_all_coords.iterrows():
    texts_3a.append(ax.text(row['x'], row['y'], row['label'],
                            fontsize=5.5, alpha=0.9, zorder=10))

adjust_text(texts_3a, ax=ax)

cx.add_basemap(ax, crs='EPSG:2286',
               source=cx.providers.USGS.USTopo, zoom=11)

ax.set_xlabel('Easting (ft)')
ax.set_ylabel('Northing (ft)')
ax.set_title('Walla Walla Basin - Well Locations (WA & OR)\n'
             'NAD 1983 StatePlane Washington South FIPS 4602 (ft)')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(plot_dir / 'WL_well_locations_map.png', dpi=150)
print(f"Plot 3a saved to {plot_dir / 'WL_well_locations_map.png'}")
plt.close()

# --- Plot 3b: Map with SD only and interpolated color flood ---
fig, ax = plt.subplots(figsize=(10, 8))

# Interpolate SD values across the map area
from scipy.interpolate import griddata

# Only use points with valid SD values for interpolation (exclude 23R01)
valid_sd = df_all_coords.dropna(subset=['sd_ft']).copy()
valid_sd = valid_sd[~valid_sd['site_id'].str.contains('461935118081501')].copy()

if len(valid_sd) >= 4 and PLOT_SD_CONTOURS:
    # Create interpolation grid
    x_min, x_max = df_all_coords['x'].min(), df_all_coords['x'].max()
    y_min, y_max = df_all_coords['y'].min(), df_all_coords['y'].max()
    # Add 10% padding
    x_pad = (x_max - x_min) * 0.10
    y_pad = (y_max - y_min) * 0.10
    grid_x = np.linspace(x_min - x_pad, x_max + x_pad, 200)
    grid_y = np.linspace(y_min - y_pad, y_max + y_pad, 200)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Interpolate using cubic method, fall back to linear for extrapolation
    points = valid_sd[['x', 'y']].values
    values = valid_sd['sd_ft'].values
    grid_sd = griddata(points, values, (grid_xx, grid_yy), method='cubic')
    # Fill NaN areas (outside convex hull) with nearest-neighbor
    grid_sd_nearest = griddata(points, values, (grid_xx, grid_yy), method='nearest')
    mask = np.isnan(grid_sd)
    grid_sd[mask] = grid_sd_nearest[mask]

    # Plot color flood
    cf = ax.contourf(grid_xx, grid_yy, grid_sd, levels=20, cmap='viridis', alpha=0.6, zorder=2)
    plt.colorbar(cf, ax=ax, label='SD of WL Change (ft)', shrink=0.8)

ax.scatter(basin_fill['x'], basin_fill['y'], c='blue', s=80, marker='o',
           label='Basin-fill', zorder=5, edgecolors='black', linewidths=0.5)
ax.scatter(basalt['x'], basalt['y'], c='red', s=25, marker='s',
           label='Basalt', zorder=6, edgecolors='black', linewidths=0.5)

# SD-scaled open circles (optional)
if PLOT_SD_BUBBLES:
    # Only include wells with valid SD values
    valid_bubble = df_all_coords.dropna(subset=['sd_ft'])
    sd_vals = valid_bubble['sd_ft']
    sd_min, sd_max = sd_vals.min(), sd_vals.max()
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0
    bubble_sizes = 100 + ((sd_vals - sd_min) / sd_range) * 1100
    ax.scatter(valid_bubble['x'], valid_bubble['y'],
               s=bubble_sizes, facecolors='none', edgecolors='black',
               linewidths=1.0, zorder=7, label='SD magnitude')

texts_3b = []
for _, row in df_all_coords.iterrows():
    sd_str = f"{row['sd_ft']:.1f}" if pd.notna(row['sd_ft']) else "N/A"
    texts_3b.append(ax.text(row['x'], row['y'], sd_str,
                            fontsize=5.5, alpha=0.9, zorder=10))

adjust_text(texts_3b, ax=ax)

cx.add_basemap(ax, crs='EPSG:2286',
               source=cx.providers.USGS.USTopo, zoom=11)

ax.set_xlabel('Easting (ft)')
ax.set_ylabel('Northing (ft)')
ax.set_title('Walla Walla Basin - WL Change Standard Deviation (WA & OR)\n'
             'NAD 1983 StatePlane Washington South FIPS 4602 (ft)')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(plot_dir / 'WL_well_locations_SD_map.png', dpi=150)
print(f"Plot 3b saved to {plot_dir / 'WL_well_locations_SD_map.png'}")
plt.close()


SD_THRESHOLD = 3.9

# Compute SD per site for both datasets
usgs_sd_lookup = (
    df_usgs_change.groupby('monitoring_location_id')['wl_change_ft']
    .std().to_dict()
)
owrd_sd_lookup = (
    df_owrd_change.groupby('well_id')['wl_change_ft']
    .std().to_dict()
)


# =============================================================================
# Plot 5: Normalized (z-score) hydrographs
# =============================================================================
# Each well's depth-to-water time series is standardized to zero mean and unit
# standard deviation, then plotted together. This highlights timing/pattern
# similarities regardless of absolute magnitude.

fig, ax = plt.subplots(figsize=(14, 7))

# USGS sites
for site, group in df_usgs_daily.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date').copy()
    vals = group_sorted['value']
    mean_val = vals.mean()
    std_val = vals.std()
    if std_val == 0 or pd.isna(std_val):
        continue
    group_sorted['z_score'] = (vals - mean_val) / std_val
    group_sorted = group_sorted.dropna(subset=['z_score'])
    if len(group_sorted) < 10:
        continue
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['z_score'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"WA - {usgs_label_lookup.get(site, site)}"
    ax.plot(pd.to_datetime(lowess[:, 0], unit='D', origin=t_start),
            lowess[:, 1], label=label, linewidth=0.8)

# OWRD sites
for well_id, group in df_owrd.groupby('well_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date').copy()
    vals = group_sorted['wl_ft_below_land_surface']
    mean_val = vals.mean()
    std_val = vals.std()
    if std_val == 0 or pd.isna(std_val):
        continue
    group_sorted['z_score'] = (vals - mean_val) / std_val
    group_sorted = group_sorted.dropna(subset=['z_score'])
    if len(group_sorted) < 10:
        continue
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['z_score'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"OR - {well_id}"
    ax.plot(pd.to_datetime(lowess[:, 0], unit='D', origin=t_start),
            lowess[:, 1], label=label, linewidth=0.8, linestyle='--')

# Reference line at zero
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

# Formatting
ax.set_xlim(t_start, t_end)
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Water Level (z-score)')
ax.set_title('Walla Walla Basin - Normalized Hydrographs, LOESS Smoothed (WA & OR)')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.legend(fontsize=7, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'WL_normalized_hydrogr.png', dpi=150)
print(f"Plot 5 saved to {plot_dir / 'WL_normalized_hydrogr.png'}")
plt.close()


# =============================================================================
# Plot 6: Deviation from mean, split by SD threshold (stacked, matched scales)
# =============================================================================
# Similar to Plot 4 but plots deviation from each well's mean depth-to-water
# rather than change from initial value.

# Compute deviation from mean for USGS
usgs_dev_data = []
for site, group in df_usgs_daily.groupby('monitoring_location_id'):
    group = group.sort_values('date').copy()
    mean_val = group['value'].mean()
    # Negative sign so rising WL is positive deviation
    group['wl_dev_ft'] = -(group['value'] - mean_val)
    usgs_dev_data.append(group)

df_usgs_dev = pd.concat(usgs_dev_data, ignore_index=True)

# Compute deviation from mean for OWRD
owrd_dev_data = []
for well_id, group in df_owrd.groupby('well_id'):
    group = group.sort_values('date').copy()
    mean_val = group['wl_ft_below_land_surface'].mean()
    group['wl_dev_ft'] = -(group['wl_ft_below_land_surface'] - mean_val)
    owrd_dev_data.append(group)

df_owrd_dev = pd.concat(owrd_dev_data, ignore_index=True)

# SD lookups (reuse from earlier)
# usgs_sd_lookup and owrd_sd_lookup already computed above

# Determine top panel y-range
_y_mins_top6, _y_maxs_top6 = [], []
for site, group in df_usgs_dev.groupby('monitoring_location_id'):
    sd_val = usgs_sd_lookup.get(site, 0)
    if sd_val >= SD_THRESHOLD:
        vals = group['wl_dev_ft'].dropna().values
        if len(vals):
            _y_mins_top6.append(np.nanmin(vals))
            _y_maxs_top6.append(np.nanmax(vals))
for well_id, group in df_owrd_dev.groupby('well_id'):
    sd_val = owrd_sd_lookup.get(well_id, 0)
    if sd_val >= SD_THRESHOLD:
        vals = group['wl_dev_ft'].dropna().values
        if len(vals):
            _y_mins_top6.append(np.nanmin(vals))
            _y_maxs_top6.append(np.nanmax(vals))

top6_y_lo = min(_y_mins_top6) if _y_mins_top6 else -20
top6_y_hi = max(_y_maxs_top6) if _y_maxs_top6 else 10
top6_pad = (top6_y_hi - top6_y_lo) * 0.05
top6_range = (top6_y_hi + top6_pad) - (top6_y_lo - top6_pad)
bot6_range = 30  # -20 to 10

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={'height_ratios': [top6_range, bot6_range]}
)

# Plot USGS
for site, group in df_usgs_dev.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('date')
    sd_val = usgs_sd_lookup.get(site, 0)
    label = f"WA - {usgs_label_lookup.get(site, site)}"
    if sd_val >= SD_THRESHOLD:
        ax_top.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
                    label=label, linewidth=0.8)
    else:
        ax_bot.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
                    label=label, linewidth=0.8)

# Plot OWRD
for well_id, group in df_owrd_dev.groupby('well_id'):
    group_sorted = group.sort_values('date')
    sd_val = owrd_sd_lookup.get(well_id, 0)
    label = f"OR - {well_id}"
    if sd_val >= SD_THRESHOLD:
        ax_top.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
                    label=label, linewidth=0.8, linestyle='--')
    else:
        ax_bot.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
                    label=label, linewidth=0.8, linestyle='--')

# Set y-limits
ax_top.set_ylim(top6_y_lo - top6_pad, top6_y_hi + top6_pad)
ax_bot.set_ylim(-20, 10)

# Formatting
for ax in (ax_top, ax_bot):
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel('Deviation from Mean WL (ft)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

ax_top.set_title(f'Walla Walla Basin - Deviation from Mean WL: SD >= {SD_THRESHOLD} ft')
ax_bot.set_title(f'Walla Walla Basin - Deviation from Mean WL: SD < {SD_THRESHOLD} ft')
ax_bot.set_xlabel('Date')

plt.tight_layout()
plt.savefig(plot_dir / 'WL_deviation_by_SD_group.png', dpi=150)
print(f"Plot 6 saved to {plot_dir / 'WL_deviation_by_SD_group.png'}")
plt.close()


# =============================================================================
# Plot 7: Deviation from mean - all wells on one plot (like WL_change_hydrogr)
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

# Plot USGS (WA) sites with solid lines
for site, group in df_usgs_dev.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('date')
    label = f"WA - {usgs_label_lookup.get(site, site)}"
    ax.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
            label=label, linewidth=0.8)

# Plot OWRD (OR) sites with dashed lines
for well_id, group in df_owrd_dev.groupby('well_id'):
    group_sorted = group.sort_values('date')
    label = f"OR - {well_id}"
    ax.plot(group_sorted['date'], group_sorted['wl_dev_ft'],
            label=label, linewidth=0.8, linestyle='--')

# Reference line at zero
ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

# Formatting
ax.set_xlim(t_start, t_end)
ax.set_xlabel('Date')
ax.set_ylabel('Deviation from Mean WL (ft)')
ax.set_title('Walla Walla Basin - Deviation from Mean Water Level (WA & OR)')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.legend(fontsize=7, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'WL_deviation_hydrogr.png', dpi=150)
print(f"Plot 7 saved to {plot_dir / 'WL_deviation_hydrogr.png'}")
plt.close()


# =============================================================================
# Plot 8: Deviation from mean - all wells, LOESS smoothed (like Plot 7 + LOESS)
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 7))

# USGS (WA) sites - LOESS
for site, group in df_usgs_dev.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date')
    group_sorted = group_sorted.dropna(subset=['wl_dev_ft'])
    if len(group_sorted) < 10:
        continue
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['wl_dev_ft'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"WA - {usgs_label_lookup.get(site, site)}"
    ax.plot(pd.to_datetime(lowess[:, 0], unit='D', origin=t_start),
            lowess[:, 1], label=label, linewidth=0.8)

# OWRD (OR) sites - LOESS
for well_id, group in df_owrd_dev.groupby('well_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date')
    group_sorted = group_sorted.dropna(subset=['wl_dev_ft'])
    if len(group_sorted) < 10:
        continue
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['wl_dev_ft'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"OR - {well_id}"
    ax.plot(pd.to_datetime(lowess[:, 0], unit='D', origin=t_start),
            lowess[:, 1], label=label, linewidth=0.8, linestyle='--')

ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
ax.set_xlim(t_start, t_end)
ax.set_xlabel('Date')
ax.set_ylabel('Deviation from Mean WL (ft)')
ax.set_title('Walla Walla Basin - Deviation from Mean WL, LOESS Smoothed (WA & OR)')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.legend(fontsize=7, loc='best', ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(plot_dir / 'WL_deviation_loess_hydrogr.png', dpi=150)
print(f"Plot 8 saved to {plot_dir / 'WL_deviation_loess_hydrogr.png'}")
plt.close()

# =============================================================================
# Plot 9: Deviation from mean, split by SD - LOESS smoothed (like Plot 6 + LOESS)
# =============================================================================

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={'height_ratios': [top6_range, bot6_range]}
)

# USGS
for site, group in df_usgs_dev.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date')
    group_sorted = group_sorted.dropna(subset=['wl_dev_ft'])
    if len(group_sorted) < 10:
        continue
    sd_val = usgs_sd_lookup.get(site, 0)
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['wl_dev_ft'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"WA - {usgs_label_lookup.get(site, site)}"
    dates_smooth = pd.to_datetime(lowess[:, 0], unit='D', origin=t_start)
    if sd_val >= SD_THRESHOLD:
        ax_top.plot(dates_smooth, lowess[:, 1], label=label, linewidth=0.8)
    else:
        ax_bot.plot(dates_smooth, lowess[:, 1], label=label, linewidth=0.8)

# OWRD
for well_id, group in df_owrd_dev.groupby('well_id'):
    group_sorted = group.sort_values('date').drop_duplicates(subset='date')
    group_sorted = group_sorted.dropna(subset=['wl_dev_ft'])
    if len(group_sorted) < 10:
        continue
    sd_val = owrd_sd_lookup.get(well_id, 0)
    x = (group_sorted['date'] - t_start).dt.days.values.astype(float)
    y = group_sorted['wl_dev_ft'].values
    lowess = sm.nonparametric.lowess(y, x, frac=loess_frac)
    label = f"OR - {well_id}"
    dates_smooth = pd.to_datetime(lowess[:, 0], unit='D', origin=t_start)
    if sd_val >= SD_THRESHOLD:
        ax_top.plot(dates_smooth, lowess[:, 1], label=label, linewidth=0.8,
                    linestyle='--')
    else:
        ax_bot.plot(dates_smooth, lowess[:, 1], label=label, linewidth=0.8,
                    linestyle='--')

# Set y-limits
ax_top.set_ylim(top6_y_lo - top6_pad, top6_y_hi + top6_pad)
ax_bot.set_ylim(-20, 10)

# Formatting
for ax in (ax_top, ax_bot):
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
    ax.set_xlim(t_start, t_end)
    ax.set_ylabel('Deviation from Mean WL (ft)')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

ax_top.set_title(f'Walla Walla Basin - Deviation from Mean WL, LOESS: SD >= {SD_THRESHOLD} ft')
ax_bot.set_title(f'Walla Walla Basin - Deviation from Mean WL, LOESS: SD < {SD_THRESHOLD} ft')
ax_bot.set_xlabel('Date')

plt.tight_layout()
plt.savefig(plot_dir / 'WL_deviation_loess_by_SD_group.png', dpi=150)
print(f"Plot 9 saved to {plot_dir / 'WL_deviation_loess_by_SD_group.png'}")
plt.close()
