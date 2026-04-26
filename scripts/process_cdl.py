"""
CDL Processing Pipeline
------------------------
Produces maps showing year-over-year natural land flows across the
Corn Belt between 2006 and 2012, including both:
  - Natural→Corn conversions
  - Corn→Natural reversions

Steps:
  1. Reproject all files to EPSG:5070 (NAD83 Albers) onto a shared grid
  2. Resample all 56m files (2006, 2007) to 30m (nearest neighbor)
  3. Mosaic all 11 states into one raster per year
  4. Compute flow layer per consecutive year pair
  5. Visualize pixel-level map (2011→2012 interval)
  6. Aggregate to county level and visualize choropleth (2011→2012)
  7. Export GeoJSON per year for Leaflet time slider

Flow layer pixel values:
  0 = no relevant change
  1 = natural→corn (conversion)
  2 = corn→natural (reversion)

Requirements:
    pip install rasterio numpy matplotlib geopandas rasterstats affine

Outputs (written to ../outputs/):
    cdl_flow_map.png                - pixel-level flow map (2011→2012)
    cdl_flow_county_map.png         - county-level net change choropleth
    corn_belt_YEAR.geojson          - one GeoJSON per year for Leaflet
"""

###################################################################
# %% Imports and configuration
import os
import math
import rasterio
import rasterio.merge
import rasterio.warp
import rasterio.enums
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import shutil
from pathlib import Path
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterstats import zonal_stats

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR      = Path('../data/raw-data/cdl_data')
PROCESSED_DIR = Path('../data/processed-data')
RAW_DIR       = Path('../data/raw-data')
VIZ_DIR       = Path('../outputs')

PROCESSED_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

STATES = ['IA', 'IL', 'IN', 'KS', 'MN', 'MO', 'ND', 'NE', 'OH', 'SD', 'WI']
YEARS  = [2006, 2007, 2008, 2009, 2010, 2011, 2012]

# Flow layers cover consecutive year pairs: 2006→2007, 2007→2008, ...
FLOW_YEARS = YEARS[1:]   # [2007, 2008, 2009, 2010, 2011, 2012]

# 2006 and 2007 are 56m resolution, all others are 30m
RESOLUTION_MAP = {year: '56m' if year <= 2007 else '30m' for year in YEARS}

TARGET_CRS = 'EPSG:5070'
TARGET_RES = 30

MIN_NATURAL_PIXELS = 100

# Land cover codes
CODES_OF_INTEREST = {
    1:   'Corn',
    176: 'Grassland/Pasture',
    190: 'Woody Wetlands',
    195: 'Herbaceous Wetlands',
}

CORN_BELT_FIPS = [
    '17', '18', '19', '20', '27', '29', '31', '38', '39', '46', '55'
]

# ── Build file path from state + year ─────────────────────────────────────────

def get_tif_path(state, year):
    state_lower = state.lower()
    res = RESOLUTION_MAP[year]
    filename = f'cdl_{res}_r_{state_lower}_{year}_albers.tif'
    return BASE_DIR / state / filename

###################################################################
# %% Exploration
print("Exploring data...")

rows = []
missing_files = []

for state in STATES:
    for year in YEARS:
        path = get_tif_path(state, year)

        if not path.exists():
            missing_files.append(str(path))
            continue

        try:
            with rasterio.open(path) as src:
                crs   = src.crs.to_string()
                res_m = round(src.res[0], 1)
                shape = src.shape
                # Downsample for speed — just need approximate counts
                data  = src.read(
                    1,
                    out_shape=(src.height // 10, src.width // 10),
                    resampling=Resampling.nearest
                )

            unique, counts = np.unique(data, return_counts=True)
            code_counts = dict(zip(unique.tolist(), counts.tolist()))

            row = {'state': state, 'year': year, 'crs': crs,
                   'res_m': res_m, 'rows': shape[0], 'cols': shape[1]}
            for code, name in CODES_OF_INTEREST.items():
                row[name] = code_counts.get(code, 0)

            rows.append(row)
            print(f"  ✓ {state} {year} — shape {shape}, res {res_m}m")

        except Exception as e:
            print(f"  ✗ {state} {year} — ERROR: {e}")
            missing_files.append(str(path))

print("\n" + "="*60)

if missing_files:
    print(f"\n⚠ MISSING OR FAILED FILES ({len(missing_files)}):")
    for f in missing_files:
        print(f"    {f}")
else:
    print(f"\n✓ All {len(STATES) * len(YEARS)} files loaded successfully")

if rows:
    df_explore = pd.DataFrame(rows)
    display_cols = ['state', 'year', 'Corn', 'Grassland/Pasture',
                    'Woody Wetlands', 'Herbaceous Wetlands']
    print("\n── Pixel counts (downsampled ~1%) ──\n")
    print(df_explore[display_cols].to_string(index=False))
    print("\n── CRS check ──")
    print("Unique CRS values:", df_explore['crs'].nunique(), "distinct")
    if df_explore['crs'].nunique() > 1:
        print("⚠ CRS mismatch — reprojection required")
    else:
        print("✓ All files share the same CRS")

###################################################################
# %% Step 1 & 2: Reproject and resample onto shared grid

# ── Compute a shared reference grid ───────────────────────────────────────────
# Each file's calculate_default_transform() produces a slightly different
# pixel origin, causing misalignment when comparing years.
# Fix: compute the combined bounding box of ALL files, snap to a clean 30m
# grid boundary, and reproject every file onto this identical shared grid.

print("\nComputing shared reference grid from all source files...")

target_crs_obj = CRS.from_string(TARGET_CRS)
all_bounds_x   = []
all_bounds_y   = []

for year in YEARS:
    for state in STATES:
        src_path = get_tif_path(state, year)
        if not src_path.exists():
            continue
        with rasterio.open(src_path) as src:
            bounds = transform_bounds(src.crs, target_crs_obj, *src.bounds)
            all_bounds_x += [bounds[0], bounds[2]]
            all_bounds_y += [bounds[1], bounds[3]]

grid_left    = math.floor(min(all_bounds_x) / TARGET_RES) * TARGET_RES
grid_bottom  = math.floor(min(all_bounds_y) / TARGET_RES) * TARGET_RES
grid_right   = math.ceil( max(all_bounds_x) / TARGET_RES) * TARGET_RES
grid_top     = math.ceil( max(all_bounds_y) / TARGET_RES) * TARGET_RES

SHARED_TRANSFORM = Affine(TARGET_RES, 0, grid_left, 0, -TARGET_RES, grid_top)
SHARED_WIDTH     = int((grid_right  - grid_left) / TARGET_RES)
SHARED_HEIGHT    = int((grid_top    - grid_bottom) / TARGET_RES)

print(f"  Shared grid origin: ({grid_left:.2f}, {grid_top:.2f})")
print(f"  Shared grid size:   {SHARED_WIDTH} x {SHARED_HEIGHT} pixels")


def process_file(src_path, dst_path, target_crs):
    """Reproject onto the shared grid using nearest-neighbor resampling."""
    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({
            'crs':       target_crs,
            'transform': SHARED_TRANSFORM,
            'width':     SHARED_WIDTH,
            'height':    SHARED_HEIGHT,
            'dtype':     'uint8',
            'compress':  'lzw'
        })
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            reproject(
                source      =rasterio.band(src, 1),
                destination =rasterio.band(dst, 1),
                src_crs     =src.crs,
                dst_crs     =target_crs,
                resampling  =Resampling.nearest
            )

print("\nStep 1 & 2: Reprojecting and resampling all files onto shared grid...")
print(f"  Target CRS: {TARGET_CRS}  |  Resolution: {TARGET_RES}m\n")

temp_dir = PROCESSED_DIR / 'temp'
temp_dir.mkdir(exist_ok=True)

for year in YEARS:
    print(f"  Processing {year}:")
    for state in STATES:
        src_path = get_tif_path(state, year)
        dst_path = temp_dir / f'{state}_{year}_processed.tif'
        if dst_path.exists():
            print(f"    {state} — skipping (exists)")
            continue
        process_file(src_path, dst_path, TARGET_CRS)
        print(f"    {state} ✓")

###################################################################
# %% Step 3: Mosaic

def build_mosaic(year, out_path):
    paths    = [temp_dir / f'{state}_{year}_processed.tif' for state in STATES]
    datasets = [rasterio.open(p) for p in paths]
    mosaic, transform = rasterio.merge.merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update({
        'height': mosaic.shape[1], 'width': mosaic.shape[2],
        'transform': transform, 'compress': 'lzw'
    })
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(mosaic)
    for ds in datasets:
        ds.close()
    print(f"  Mosaic saved: {out_path}")
    return out_path

print("\nStep 3: Building mosaics...")

mosaic_paths = {}
for year in YEARS:
    mosaic_path = PROCESSED_DIR / f'mosaic_{year}.tif'
    mosaic_paths[year] = mosaic_path
    if not mosaic_path.exists():
        build_mosaic(year, mosaic_path)
    else:
        print(f"  mosaic_{year}.tif already exists, skipping")

# Clean up temp files
if temp_dir.exists() and temp_dir.is_dir():
    shutil.rmtree(temp_dir)
    print("  Temp files cleaned up")

###################################################################
# %% Step 4: Compute flow layers
print("\nStep 4: Computing flow layers...")

"""
Flow layer for each consecutive year pair (prev→curr):
  Pixel values:
    0 = no relevant change
    1 = natural→corn  (conversion: was natural in prev, is corn in curr)
    2 = corn→natural  (reversion:  was corn in prev, is natural in curr)

flow_layer_YEAR.tif represents the flows between YEAR-1 and YEAR.
e.g. flow_layer_2008.tif = what changed between 2007 and 2008.
"""

corn_code     = [code for code, name in CODES_OF_INTEREST.items() if name == 'Corn'][0]
natural_codes = [code for code, name in CODES_OF_INTEREST.items() if name != 'Corn']
pixel_area_ha = (TARGET_RES ** 2) / 10_000

flow_paths = {}

for i, year in enumerate(FLOW_YEARS, 1):
    prev_year  = YEARS[i - 1]
    flow_path  = PROCESSED_DIR / f'flow_layer_{year}.tif'
    flow_paths[year] = flow_path

    if flow_path.exists():
        print(f"  flow_layer_{year}.tif already exists, skipping")
        continue

    print(f"  Computing {prev_year}→{year}...")

    with rasterio.open(mosaic_paths[prev_year]) as src_prev, \
         rasterio.open(mosaic_paths[year])      as src_curr:

        meta = src_prev.meta.copy()
        meta.update({'compress': 'lzw', 'dtype': 'uint8'})

        with rasterio.open(flow_path, 'w', **meta) as dst:
            chunk_size = 2048

            for row_off in range(0, src_prev.height, chunk_size):
                actual_height = min(chunk_size, src_prev.height - row_off)
                window = rasterio.windows.Window(
                    col_off=0, row_off=row_off,
                    width=src_prev.width, height=actual_height
                )

                prev_chunk = src_prev.read(1, window=window)
                curr_chunk = src_curr.read(
                    1, window=window,
                    out_shape=(actual_height, src_prev.width),
                    resampling=Resampling.nearest
                )

                was_natural = np.isin(prev_chunk, natural_codes)
                was_corn    = (prev_chunk == corn_code)
                is_natural  = np.isin(curr_chunk, natural_codes)
                is_corn     = (curr_chunk == corn_code)

                result = np.zeros_like(prev_chunk, dtype=np.uint8)
                result[was_natural & is_corn]    = 1   # conversion
                result[was_corn    & is_natural] = 2   # reversion

                dst.write(result, 1, window=window)

                pct = min(100, int((row_off + actual_height) / src_prev.height * 100))
                print(f"    Processing... {pct}%", end='\r')

    print(f"\n  flow_layer_{year}.tif saved")

# Quick stats for most recent interval (2011→2012)
print("\n  Stats for 2011→2012:")
converted_px = 0
reverted_px  = 0
with rasterio.open(flow_paths[2012]) as src:
    for row_off in range(0, src.height, 2048):
        actual_height = min(2048, src.height - row_off)
        window = rasterio.windows.Window(
            col_off=0, row_off=row_off,
            width=src.width, height=actual_height
        )
        chunk = src.read(1, window=window)
        converted_px += int(np.sum(chunk == 1))
        reverted_px  += int(np.sum(chunk == 2))
        del chunk

converted_ha = converted_px * pixel_area_ha
reverted_ha  = reverted_px  * pixel_area_ha
print(f"    Natural→Corn: {converted_ha:,.0f} ha")
print(f"    Corn→Natural: {reverted_ha:,.0f} ha")
print(f"    Net loss:     {converted_ha - reverted_ha:,.0f} ha")

###################################################################
# %% Step 5: Pixel-level flow map (2011→2012 interval)
print("\nStep 5: Generating pixel-level flow map (2011→2012)...")

PLOT_DOWNSAMPLE = 5

with rasterio.open(flow_paths[2012]) as src:
    data = src.read(
        1,
        out_shape=(src.height // PLOT_DOWNSAMPLE, src.width // PLOT_DOWNSAMPLE),
        resampling=Resampling.nearest
    )
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

color_map = {
    0: (1, 1, 1, 0),            # transparent
    1: (0.85, 0.18, 0.18, 0.9), # red — natural→corn
    2: (0.20, 0.50, 0.80, 0.7), # blue — corn→natural
}

rgb = np.zeros((*data.shape, 4), dtype=float)
for val, color in color_map.items():
    rgb[data == val] = color

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.imshow(rgb, extent=extent, origin='upper', interpolation='none')

county_shp = gpd.read_file(RAW_DIR / 'tl_2012_us_county.zip')
state_shp  = county_shp.dissolve(by='STATEFP')
state_shp  = state_shp[state_shp.index.isin(CORN_BELT_FIPS)]
state_shp  = state_shp.to_crs(TARGET_CRS)
state_shp.boundary.plot(ax=ax, color='#333333', linewidth=0.8)

legend_patches = [
    mpatches.Patch(color=(0.85, 0.18, 0.18), label='Natural→Corn (2011→2012)'),
    mpatches.Patch(color=(0.20, 0.50, 0.80), label='Corn→Natural (2011→2012)'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=10, framealpha=0.9)
ax.set_title('Grassland & Wetland Land Flows, 2011→2012',
             fontsize=16, fontweight='bold', pad=12)
ax.set_xlabel(
    f'Red = natural land converted to corn  |  '
    f'Blue = corn reverted to natural land  |  '
    f'Net loss: {converted_ha - reverted_ha:,.0f} ha',
    fontsize=9, color='#444444'
)
ax.axis('off')
plt.tight_layout()
map_path = VIZ_DIR / 'cdl_flow_map.png'
plt.savefig(map_path, dpi=150, bbox_inches='tight')
print(f"  Map saved to {map_path}")

###################################################################
# %% Step 6: County-level net change choropleth (2011→2012)
print("\nStep 6: Generating county-level net change choropleth (2011→2012)...")

print("  Loading county shapefile...")
counties = gpd.read_file(RAW_DIR / 'tl_2012_us_county.zip')
counties = counties[counties['STATEFP'].isin(CORN_BELT_FIPS)].copy()
counties = counties.to_crs(TARGET_CRS)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']
print(f"  Counties loaded: {len(counties)}")

# Zonal stats on flow layer for conversions (1) and reversions (2)
print("  Running zonal statistics on flow layer...")
stats_flow = zonal_stats(
    counties, str(flow_paths[2012]), categorical=True, nodata=0
)

# Zonal stats on prev year mosaic for natural land denominator
print("  Running zonal statistics on 2011 mosaic for natural land denominator...")
stats_nat = zonal_stats(
    counties, str(mosaic_paths[2011]), categorical=True, nodata=0
)
print("  Done.")

counties['converted_pixels']     = [s.get(1, 0) for s in stats_flow]
counties['reverted_pixels']      = [s.get(2, 0) for s in stats_flow]
counties['net_pixels']           = counties['converted_pixels'] - counties['reverted_pixels']
counties['natural_start_pixels'] = [
    sum(s.get(c, 0) for c in natural_codes) for s in stats_nat
]

# Net rate as % of natural land at start of interval
counties['net_rate'] = np.where(
    counties['natural_start_pixels'] >= MIN_NATURAL_PIXELS,
    counties['net_pixels'] / counties['natural_start_pixels'] * 100,
    np.nan
)

counties['converted_ha'] = counties['converted_pixels'] * pixel_area_ha
counties['reverted_ha']  = counties['reverted_pixels']  * pixel_area_ha
counties['net_ha']       = counties['net_pixels']       * pixel_area_ha

print(f"\n  Net rate summary (% of natural land):")
print(counties['net_rate'].describe().round(2))

# Diverging colormap: red = net loss, blue = net gain, white = zero
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
vmax = 10   # cap at ±10% for readability

counties[counties['net_rate'].isna()].plot(
    ax=ax, color='#eeeeee', edgecolor='white', linewidth=0.2
)
counties[counties['net_rate'].notna()].plot(
    ax=ax,
    column='net_rate',
    cmap='RdBu_r',   # red = positive (loss), blue = negative (gain)
    vmin=-vmax,
    vmax=vmax,
    edgecolor='white',
    linewidth=0.2,
    legend=False
)

state_boundaries = counties.dissolve(by='STATEFP')
state_boundaries.boundary.plot(ax=ax, color='#333333', linewidth=0.8)

sm = plt.cm.ScalarMappable(
    cmap='RdBu_r',
    norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.03, pad=0.02, aspect=40)
cbar.set_label('Net change in natural land as % of county natural land at start (2011)',
               fontsize=9)
cbar.set_ticks([-vmax, -5, 0, 5, vmax])
cbar.set_ticklabels([f'−{vmax}% (gain)', '−5%', '0', '+5%', f'+{vmax}% (loss)'])

no_data_patch = mpatches.Patch(color='#eeeeee', label='No significant natural land')
ax.legend(handles=[no_data_patch], loc='lower left', fontsize=9, framealpha=0.9)
ax.set_title('Net Natural Land Change by County, 2011→2012\n'
             'Red = net loss (more conversion than reversion)  |  '
             'Blue = net gain (more reversion than conversion)',
             fontsize=13, fontweight='bold', pad=12)
ax.axis('off')
plt.tight_layout()
map_path = VIZ_DIR / 'cdl_flow_county_map.png'
plt.savefig(map_path, dpi=150, bbox_inches='tight')
print(f"  Map saved to {map_path}")

###################################################################
# %% Step 7: Export one GeoJSON per flow year for Leaflet time slider
print("\nStep 7: Exporting GeoJSON files for Leaflet time slider...")

geojson_cols  = ['FIPS', 'NAME', 'STATEFP', 'geometry']
counties_base = counties[geojson_cols].copy()

for i, year in enumerate(FLOW_YEARS, 1):
    prev_year = YEARS[i - 1]
    print(f"  Processing {prev_year}→{year}...")

    # Flow layer: conversions (1) and reversions (2)
    stats_flow_yr = zonal_stats(
        counties, str(flow_paths[year]), categorical=True, nodata=0
    )

    # Previous year mosaic: natural land denominator
    stats_nat_yr = zonal_stats(
        counties, str(mosaic_paths[prev_year]), categorical=True, nodata=0
    )

    counties_year = counties_base.copy()
    counties_year['converted_pixels']     = [s.get(1, 0) for s in stats_flow_yr]
    counties_year['reverted_pixels']      = [s.get(2, 0) for s in stats_flow_yr]
    counties_year['net_pixels']           = (counties_year['converted_pixels'] -
                                              counties_year['reverted_pixels'])
    counties_year['natural_start_pixels'] = [
        sum(s.get(c, 0) for c in natural_codes) for s in stats_nat_yr
    ]

    counties_year['net_rate'] = np.where(
        counties_year['natural_start_pixels'] >= MIN_NATURAL_PIXELS,
        counties_year['net_pixels'] / counties_year['natural_start_pixels'] * 100,
        np.nan
    )

    counties_year['converted_ha'] = counties_year['converted_pixels'] * pixel_area_ha
    counties_year['reverted_ha']  = counties_year['reverted_pixels']  * pixel_area_ha
    counties_year['net_ha']       = counties_year['net_pixels']       * pixel_area_ha

    # Round for file size
    counties_year['net_rate']     = counties_year['net_rate'].round(2)
    counties_year['converted_ha'] = counties_year['converted_ha'].round(1)
    counties_year['reverted_ha']  = counties_year['reverted_ha'].round(1)
    counties_year['net_ha']       = counties_year['net_ha'].round(1)

    # Reproject to WGS84 for Leaflet
    counties_year = counties_year.to_crs('EPSG:4326')

    # Replace NaN with None for clean JSON serialization
    counties_year['net_rate'] = counties_year['net_rate'].where(
        counties_year['net_rate'].notna(), other=None
    )

    # Simplify geometries
    counties_year['geometry'] = counties_year['geometry'].simplify(
        tolerance=0.01, preserve_topology=True
    )

    output_path = VIZ_DIR / f'corn_belt_{year}.geojson'
    counties_year.to_file(output_path, driver='GeoJSON')

    size_kb = os.path.getsize(output_path) / 1024
    print(f"    Saved corn_belt_{year}.geojson ({size_kb:,.0f} KB)")

print("\nAll GeoJSON files exported.")
print("Files ready for Leaflet time slider:")
for year in FLOW_YEARS:
    print(f"  corn_belt_{year}.geojson  ({YEARS[FLOW_YEARS.index(year)]}→{year})")