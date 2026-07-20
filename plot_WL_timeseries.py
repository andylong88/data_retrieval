import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Path to downloaded data
data_dir = Path(r'downloaded')

# Output directory for plots
plot_dir = Path('plots')
plot_dir.mkdir(exist_ok=True)

# Read and combine all USGS files
frames = []
for f in sorted(data_dir.glob('USGS*.csv')):
    df = pd.read_csv(f, parse_dates=['time'])
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

# Read site info for land-surface altitudes
df_site = pd.read_csv(data_dir / 'site_info.csv')
altitude_lookup = df_site.set_index('monitoring_location_id')['altitude'].to_dict()

# Build short label lookup from monitoring_location_name (last 5 chars after dash)
name_lookup = df_site.set_index('monitoring_location_id')['monitoring_location_name'].to_dict()
label_lookup = {site: name.split('-')[-1][-5:] for site, name in name_lookup.items()}

# Calculate WL altitude (land-surface altitude minus depth to water)
df_all['wl_altitude'] = df_all.apply(
    lambda row: altitude_lookup.get(row['monitoring_location_id'], None) - row['value'], axis=1
)

# Separate the deep site for the right axis
right_site = 'USGS-460014118281805'
df_left = df_all[df_all['monitoring_location_id'] != right_site]
df_right = df_all[df_all['monitoring_location_id'] == right_site]

# --- Plot 1: Depth to water ---
left_min, left_max = df_left['value'].min(), df_left['value'].max()
right_min, right_max = df_right['value'].min(), df_right['value'].max()
span = max(left_max - left_min, right_max - right_min)
padding = span * 0.05

fig, ax_left = plt.subplots(figsize=(12, 6))
ax_right = ax_left.twinx()

for site, group in df_left.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('time')
    ax_left.plot(group_sorted['time'], group_sorted['value'], label=label_lookup[site], linewidth=0.8)

group_sorted = df_right.sort_values('time')
ax_right.plot(group_sorted['time'], group_sorted['value'],
              label=label_lookup[right_site], linewidth=0.8, color='tab:red', linestyle='--')

ax_left.set_ylim(left_min - padding, left_min + span + padding)
ax_right.set_ylim(right_min - padding, right_min + span + padding)
ax_left.invert_yaxis()
ax_right.invert_yaxis()

ax_left.set_xlim(pd.Timestamp('2022-07-01'), pd.Timestamp('2024-08-01'))
ax_left.set_xlabel('Date')
ax_left.set_ylabel('Depth to Water (ft below land surface)')
ax_right.set_ylabel(f'{label_lookup[right_site]}\nDepth to Water (ft)')
ax_left.set_title('Water Level Time Series - Depth')

lines_left, labels_left = ax_left.get_legend_handles_labels()
lines_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc='best')

# Set major ticks at every 3 months, minor ticks every month
ax_left.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax_left.xaxis.set_minor_locator(mdates.MonthLocator())
ax_left.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig(plot_dir / 'WL_depth_hydrogr.png', dpi=150)
print(f"Plot saved to {plot_dir / 'WL_depth_hydrogr.png'}")
plt.close()

# --- Plot 2: WL altitude above NAVD88 (excluding 23R01) ---
df_alt = df_all[df_all['monitoring_location_id'] != 'USGS-461935118081501']

fig, ax = plt.subplots(figsize=(12, 6))

for site, group in df_alt.groupby('monitoring_location_id'):
    group_sorted = group.sort_values('time')
    ax.plot(group_sorted['time'], group_sorted['wl_altitude'], label=label_lookup[site], linewidth=0.8)

ax.set_xlim(pd.Timestamp('2022-07-01'), pd.Timestamp('2024-08-01'))
ax.set_xlabel('Date')
ax.set_ylabel('WL Altitude (ft above NAVD88)')
ax.set_title('Water Level Time Series - Altitude')
ax.legend(fontsize=8, loc='best')

# Set major ticks at every 3 months, minor ticks every month
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.tight_layout()
plt.savefig(plot_dir / 'WL_altitude_hydrogr.png', dpi=150)
print(f"Plot saved to {plot_dir / 'WL_altitude_hydrogr.png'}")
plt.close()

# plt.show()  # Uncomment to display interactively
