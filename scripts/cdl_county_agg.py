"""
County-Level Aggregation of CDL Conversion Layer
--------------------------------------------------
Aggregates pixel-level grassland/wetland conversion data to county level,
then plots a choropleth showing conversion rate by county.

Requirements:
    pip install rasterstats geopandas matplotlib mapclassify

Inputs:
    ../data/processed-data/conversion_layer.tif
    ../data/raw-data/tl_2012_us_county.zip   (or wherever you saved it)

Output:
    ../outputs/cdl_conversion_county_map.png
"""

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathlib import Path
from rasterstats import zonal_stats

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR  = Path('../data/processed-data')
RAW_DIR        = Path('../data/raw-data')
VIZ_DIR        = Path('../outputs')
VIZ_DIR.mkdir(exist_ok=True)

TARGET_CRS       = 'EPSG:5070'
conversion_path  = PROCESSED_DIR / 'conversion_layer.tif'

CORN_BELT_FIPS = [
    '17', '18', '19', '20', '27', '29', '31', '38', '39', '46', '55'
]

# ── Step 1: Load and prepare county shapefile ─────────────────────────────────

print("Loading county shapefile...")
counties = gpd.read_file(RAW_DIR / 'tl_2012_us_county.zip')

# Filter to Corn Belt states and reproject to match raster CRS
counties = counties[counties['STATEFP'].isin(CORN_BELT_FIPS)].copy()
counties = counties.to_crs(TARGET_CRS)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']

print(f"  Counties loaded: {len(counties)}")

# ── Step 2: Zonal statistics ──────────────────────────────────────────────────

# For each county polygon, count pixels with value 1 (converted)
# and value 2 (retained natural land).
# We use categorical=True to get a count per pixel value.

print("Running zonal statistics (this may take a few minutes)...")

stats = zonal_stats(
    counties,
    str(conversion_path),
    categorical=True,    # returns count of each unique pixel value
    nodata=0             # treat 0 (background) as nodata
)

print("  Done.")

# ── Step 3: Build summary dataframe ──────────────────────────────────────────

# stats is a list of dicts, one per county, like:
# {1: 423, 2: 8921}  <- {converted_pixels: N, retained_pixels: N}
# Counties with no natural land will have empty dicts or missing keys.

converted_counts = [s.get(1, 0) for s in stats]
retained_counts  = [s.get(2, 0) for s in stats]

counties['converted_pixels'] = converted_counts
counties['retained_pixels']  = retained_counts
counties['total_natural']    = counties['converted_pixels'] + counties['retained_pixels']

# Conversion rate = converted / total natural land
# Only calculate for counties that had meaningful natural land
# (avoid division by zero and noisy small-sample counties)
MIN_NATURAL_PIXELS = 100   # ~9 hectares at 30m resolution — filters out noise

counties['conversion_rate'] = np.where(
    counties['total_natural'] >= MIN_NATURAL_PIXELS,
    counties['converted_pixels'] / counties['total_natural'] * 100,
    np.nan
)

# Convert pixel counts to hectares for reference
pixel_area_ha = 30**2 / 10_000
counties['converted_ha'] = counties['converted_pixels'] * pixel_area_ha
counties['retained_ha']  = counties['retained_pixels']  * pixel_area_ha

print(f"\nCounty conversion rate summary:")
print(counties['conversion_rate'].describe().round(2))
print(f"\nTop 10 counties by conversion rate:")
top10 = counties.nlargest(10, 'conversion_rate')[
    ['FIPS', 'NAME', 'STATEFP', 'conversion_rate', 'converted_ha']
]
print(top10.to_string(index=False))

# ── Step 4: Plot choropleth ───────────────────────────────────────────────────

print("\nGenerating map...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Counties with no natural land data shown in light grey
counties[counties['conversion_rate'].isna()].plot(
    ax=ax,
    color='#eeeeee',
    edgecolor='white',
    linewidth=0.2
)

# Counties with data colored by conversion rate
# Using a white -> orange -> red sequential colormap
# vmax capped at 30% — most counties are well below this,
# capping prevents a few extreme outliers from washing out the rest
counties[counties['conversion_rate'].notna()].plot(
    ax=ax,
    column='conversion_rate',
    cmap='YlOrRd',
    vmin=0,
    vmax=30,
    edgecolor='white',
    linewidth=0.2,
    legend=False
)

# State boundaries on top for readability
state_boundaries = counties.dissolve(by='STATEFP')
state_boundaries.boundary.plot(
    ax=ax,
    color='#333333',
    linewidth=0.8
)

# Colorbar
sm = plt.cm.ScalarMappable(
    cmap='YlOrRd',
    norm=mcolors.Normalize(vmin=0, vmax=30)
)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=ax,
    orientation='horizontal',
    fraction=0.03, pad=0.02, aspect=40
)
cbar.set_label('% of Natural Land (Grassland/Wetland) Converted to Corn', fontsize=10)
cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])
cbar.set_ticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '≥30%'])

# No-data legend patch
no_data_patch = mpatches.Patch(color='#eeeeee', label='No significant natural land')
ax.legend(handles=[no_data_patch], loc='lower left', fontsize=9, framealpha=0.9)

# Titles
ax.set_title(
    'Grassland & Wetland Converted to Corn by County, 2006–2012',
    fontsize=16, fontweight='bold', pad=12
)

total_converted_ha = counties['converted_ha'].sum()
ax.set_xlabel(
    f'Total natural land converted across Corn Belt: {total_converted_ha:,.0f} hectares  '
    f'({total_converted_ha / 1e6:.1f} million ha)',
    fontsize=9, color='#444444'
)
ax.axis('off')

plt.tight_layout()
map_path = VIZ_DIR / 'cdl_conversion_county_map.png'
plt.savefig(map_path, dpi=150, bbox_inches='tight')
print(f"\nMap saved to {map_path}")
plt.show()