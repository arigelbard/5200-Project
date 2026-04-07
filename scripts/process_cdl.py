"""
CDL Processing Pipeline
------------------------
Produces two maps showing grassland/wetland land use change across the
Corn Belt between 2006 and 2012:
  1. Pixel-level map of converted vs. retained natural land
  2. County-level choropleth of conversion rate per year

Steps:
  1. Reproject all files to EPSG:5070 (NAD83 Albers)
  2. Resample all 56m files (2006, 2007) to 30m (nearest neighbor)
  3. Mosaic all 11 states into one raster per year
  4. Compute conversion layer per year (always vs. 2006 baseline)
  5. Visualize pixel-level map (2006 vs 2012 only)
  6. Aggregate to county level and visualize choropleth
  7. Export GeoJSON per year for Leaflet time slider

Requirements:
    pip install rasterio numpy matplotlib geopandas rasterstats

Outputs (written to ../outputs/):
    cdl_conversion_map.png              - pixel-level map (2006 vs 2012)
    cdl_conversion_county_map.png       - county-level choropleth (2006 vs 2012)
    corn_belt_YEAR.geojson              - one GeoJSON per year for Leaflet
"""

###################################################################
# %% Imports and configuration
import os
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
from rasterio.warp import calculate_default_transform, reproject, Resampling
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

# 2006 and 2007 are 56m resolution, all others are 30m
RESOLUTION_MAP = {year: '56m' if year <= 2007 else '30m' for year in YEARS}

TARGET_CRS = 'EPSG:5070'
TARGET_RES = 30

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
    """
    Constructs the path based on observed naming pattern:
      BASE_DIR / IA / cdl_56m_r_ia_2006_albers.tif
    2006 and 2007 use 56m resolution, 2008-2012 use 30m resolution.
    """
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
                data  = src.read(1)

            unique, counts = np.unique(data, return_counts=True)
            code_counts = dict(zip(unique.tolist(), counts.tolist()))

            row = {
                'state': state,
                'year':  year,
                'crs':   crs,
                'res_m': res_m,
                'rows':  shape[0],
                'cols':  shape[1],
            }
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
    df = pd.DataFrame(rows)

    print("\n── Pixel counts for key land cover codes ──\n")
    display_cols = ['state', 'year', 'Corn', 'Grassland/Pasture',
                    'Woody Wetlands', 'Herbaceous Wetlands']
    print(df[display_cols].to_string(index=False))

    print("\n── CRS and resolution check ──")
    print("Unique CRS values found:", df['crs'].nunique(), "distinct CRS")
    print("Unique resolutions (m):", df['res_m'].unique())
    if df['crs'].nunique() > 1:
        print("⚠ WARNING: Not all files share the same CRS!")
        print("  You will need to reproject before merging.")
    else:
        print("✓ All files share the same CRS — safe to merge")

###################################################################
# %% Step 1 & 2: Reproject and resample

def process_file(src_path, dst_path, target_crs, target_res):
    """
    Reproject a single raster to target_crs and resample to target_res.
    Uses nearest-neighbor resampling to preserve categorical pixel values.
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=target_res
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs':       target_crs,
            'transform': transform,
            'width':     width,
            'height':    height,
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

print("\nStep 1 & 2: Reprojecting and resampling all files...")
print(f"  Target CRS: {TARGET_CRS}")
print(f"  Target resolution: {TARGET_RES}m\n")

temp_dir = PROCESSED_DIR / 'temp'
temp_dir.mkdir(exist_ok=True)

for year in YEARS:
    print(f"  Processing {year}:")
    for state in STATES:
        src_path = get_tif_path(state, year)
        dst_path = temp_dir / f'{state}_{year}_processed.tif'

        if dst_path.exists():
            print(f"    {state} — already processed, skipping")
            continue

        process_file(src_path, dst_path, TARGET_CRS, TARGET_RES)
        print(f"    {state} ✓")

###################################################################
# %% Step 3: Mosaic

def build_mosaic(year, out_path):
    """
    Merge all processed state rasters for a given year into one mosaic.
    """
    paths    = [temp_dir / f'{state}_{year}_processed.tif' for state in STATES]
    datasets = [rasterio.open(p) for p in paths]

    mosaic, transform = rasterio.merge.merge(datasets)

    meta = datasets[0].meta.copy()
    meta.update({
        'height':    mosaic.shape[1],
        'width':     mosaic.shape[2],
        'transform': transform,
        'compress':  'lzw'
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

###################################################################
# %% Step 4: Compute conversion layers
print("\nStep 4: Computing conversion layers...")

"""
Conversion logic — for each pixel, compared against 2006 baseline:
  - Was it grassland or wetland in 2006?  (natural land)
  - Is it corn in the target year?        (converted)

Output pixel values:
  0 = no change / not relevant
  1 = converted (was natural land in 2006, now corn)
  2 = was natural land in 2006, still natural in target year (retained)
"""

# Derive codes from CODES_OF_INTEREST once
corn_code     = [code for code, name in CODES_OF_INTEREST.items() if name == 'Corn'][0]
natural_codes = [code for code, name in CODES_OF_INTEREST.items() if name != 'Corn']

pixel_area_ha = (TARGET_RES ** 2) / 10_000   # 30m pixel = 0.09 ha

# Always compare against 2006 as the baseline
mosaic_baseline_path = mosaic_paths[2006]

conversion_paths = {}

for year in YEARS:
    conversion_path = PROCESSED_DIR / f'conversion_layer_{year}.tif'
    conversion_paths[year] = conversion_path

    if conversion_path.exists():
        print(f"  conversion_layer_{year}.tif already exists, skipping")
        continue

    print(f"  Computing {year} vs 2006 baseline...")

    with rasterio.open(mosaic_baseline_path) as src_baseline, \
         rasterio.open(mosaic_paths[year]) as src_year:

        if src_baseline.width != src_year.width or src_baseline.height != src_year.height:
            print(f"    Note: dimensions differ — resampling {year} to match 2006")

        meta = src_baseline.meta.copy()
        meta.update({'compress': 'lzw', 'dtype': 'uint8'})

        with rasterio.open(conversion_path, 'w', **meta) as dst:
            chunk_size = 2048

            for row_off in range(0, src_baseline.height, chunk_size):
                actual_height = min(chunk_size, src_baseline.height - row_off)
                window = rasterio.windows.Window(
                    col_off=0,
                    row_off=row_off,
                    width=src_baseline.width,
                    height=actual_height
                )

                data_baseline = src_baseline.read(1, window=window)
                data_year     = src_year.read(
                    1,
                    window=window,
                    out_shape=(actual_height, src_baseline.width),
                    resampling=Resampling.nearest
                )

                was_natural = np.isin(data_baseline, natural_codes)
                is_corn_now = (data_year == corn_code)

                result = np.zeros_like(data_baseline, dtype=np.uint8)
                result[was_natural & is_corn_now]  = 1   # converted to corn
                result[was_natural & ~is_corn_now] = 2   # natural land retained

                dst.write(result, 1, window=window)

                pct = min(100, int((row_off + actual_height) / src_baseline.height * 100))
                print(f"    Processing... {pct}%", end='\r')

    print(f"\n  conversion_layer_{year}.tif saved")

# Delete temp files now that all mosaics and conversion layers are saved
if temp_dir.exists() and temp_dir.is_dir():
    shutil.rmtree(temp_dir)
    print("\n  Temp files cleaned up")

# Quick stats for 2012 (the final year)
print("\n  Stats for 2012 vs 2006:")
with rasterio.open(conversion_paths[2012]) as src:
    data = src.read(1)

converted_pixels = np.sum(data == 1)
retained_pixels  = np.sum(data == 2)
converted_ha     = converted_pixels * pixel_area_ha
retained_ha      = retained_pixels  * pixel_area_ha

print(f"    Natural land converted to corn: {converted_ha:,.0f} hectares")
print(f"    Natural land retained:          {retained_ha:,.0f} hectares")
print(f"    Conversion rate:                {converted_ha/(converted_ha+retained_ha)*100:.1f}%")

###################################################################
# %% Step 5: Pixel-level map (2006 vs 2012 only)
print("\nStep 5: Generating pixel-level map (2006 vs 2012)...")

PLOT_DOWNSAMPLE = 5

with rasterio.open(conversion_paths[2012]) as src:
    data = src.read(
        1,
        out_shape=(
            src.height // PLOT_DOWNSAMPLE,
            src.width  // PLOT_DOWNSAMPLE
        ),
        resampling=Resampling.nearest
    )
    extent = [
        src.bounds.left, src.bounds.right,
        src.bounds.bottom, src.bounds.top
    ]

color_map = {
    0: (1, 1, 1, 0),
    1: (0.85, 0.18, 0.18, 0.9),
    2: (0.55, 0.75, 0.45, 0.5),
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
    mpatches.Patch(color=(0.85, 0.18, 0.18), label='Converted to corn (2006→2012)'),
    mpatches.Patch(color=(0.55, 0.75, 0.45), label='Natural land retained'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=10, framealpha=0.9)

ax.set_title(
    'Grassland & Wetland Converted to Corn, 2006–2012',
    fontsize=16, fontweight='bold', pad=12
)
ax.set_xlabel(
    f'Red pixels = land that was grassland or wetland in 2006 and became corn by 2012\n'
    f'Total converted: {converted_ha:,.0f} hectares  |  '
    f'Conversion rate: {converted_ha/(converted_ha+retained_ha)*100:.1f}% of natural land lost',
    fontsize=9, color='#444444'
)
ax.axis('off')

plt.tight_layout()
map_path = VIZ_DIR / 'cdl_conversion_map.png'
plt.savefig(map_path, dpi=150, bbox_inches='tight')
print(f"  Map saved to {map_path}")
plt.show()

###################################################################
# %% Step 6: County choropleth (2006 vs 2012 only)
print("\nStep 6: Generating county-level choropleth (2006 vs 2012)...")

print("  Loading county shapefile...")
counties = gpd.read_file(RAW_DIR / 'tl_2012_us_county.zip')
counties = counties[counties['STATEFP'].isin(CORN_BELT_FIPS)].copy()
counties = counties.to_crs(TARGET_CRS)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']
print(f"  Counties loaded: {len(counties)}")

print("  Running zonal statistics (this may take a few minutes)...")
stats = zonal_stats(
    counties,
    str(conversion_paths[2012]),
    categorical=True,
    nodata=0
)
print("  Done.")

counties['converted_pixels'] = [s.get(1, 0) for s in stats]
counties['retained_pixels']  = [s.get(2, 0) for s in stats]
counties['total_natural']    = counties['converted_pixels'] + counties['retained_pixels']

MIN_NATURAL_PIXELS = 100
counties['conversion_rate'] = np.where(
    counties['total_natural'] >= MIN_NATURAL_PIXELS,
    counties['converted_pixels'] / counties['total_natural'] * 100,
    np.nan
)

counties['converted_ha'] = counties['converted_pixels'] * pixel_area_ha
counties['retained_ha']  = counties['retained_pixels']  * pixel_area_ha

print(f"\n  County conversion rate summary:")
print(counties['conversion_rate'].describe().round(2))

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

counties[counties['conversion_rate'].isna()].plot(
    ax=ax, color='#eeeeee', edgecolor='white', linewidth=0.2
)
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

state_boundaries = counties.dissolve(by='STATEFP')
state_boundaries.boundary.plot(ax=ax, color='#333333', linewidth=0.8)

sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=mcolors.Normalize(vmin=0, vmax=30))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.03, pad=0.02, aspect=40)
cbar.set_label('% of Natural Land (Grassland/Wetland) Converted to Corn', fontsize=10)
cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])
cbar.set_ticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '30%'])

no_data_patch = mpatches.Patch(color='#eeeeee', label='No significant natural land')
ax.legend(handles=[no_data_patch], loc='lower left', fontsize=9, framealpha=0.9)

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
print(f"  Map saved to {map_path}")
plt.show()

###################################################################
# %% Step 7: Export one GeoJSON per year for Leaflet time slider
print("\nStep 7: Exporting GeoJSON files for Leaflet time slider...")

# Use the county geometries already loaded in Step 6
geojson_cols = ['FIPS', 'NAME', 'STATEFP', 'geometry']
counties_base = counties[geojson_cols].copy()

for year in YEARS:
    print(f"  Processing {year}...")

    stats_year = zonal_stats(
        counties,
        str(conversion_paths[year]),
        categorical=True,
        nodata=0
    )

    counties_year = counties_base.copy()
    counties_year['converted_pixels'] = [s.get(1, 0) for s in stats_year]
    counties_year['retained_pixels']  = [s.get(2, 0) for s in stats_year]
    counties_year['total_natural']    = (counties_year['converted_pixels'] +
                                         counties_year['retained_pixels'])

    counties_year['conversion_rate'] = np.where(
        counties_year['total_natural'] >= MIN_NATURAL_PIXELS,
        counties_year['converted_pixels'] / counties_year['total_natural'] * 100,
        np.nan
    )

    counties_year['converted_ha'] = counties_year['converted_pixels'] * pixel_area_ha
    counties_year['retained_ha']  = counties_year['retained_pixels']  * pixel_area_ha

    # Round for file size
    counties_year['conversion_rate'] = counties_year['conversion_rate'].round(2)
    counties_year['converted_ha']    = counties_year['converted_ha'].round(1)
    counties_year['retained_ha']     = counties_year['retained_ha'].round(1)

    # Reproject to WGS84 for Leaflet
    counties_year = counties_year.to_crs('EPSG:4326')

    # Replace NaN with None for clean JSON serialization
    counties_year['conversion_rate'] = counties_year['conversion_rate'].where(
        counties_year['conversion_rate'].notna(), other=None
    )

    # Simplify geometries
    counties_year['geometry'] = counties_year['geometry'].simplify(
        tolerance=0.01,
        preserve_topology=True
    )

    output_path = VIZ_DIR / f'corn_belt_{year}.geojson'
    counties_year.to_file(output_path, driver='GeoJSON')

    size_kb = os.path.getsize(output_path) / 1024
    print(f"    Saved corn_belt_{year}.geojson ({size_kb:,.0f} KB)")

print("\nAll GeoJSON files exported.")
print("Files ready for Leaflet time slider:")
for year in YEARS:
    print(f"  corn_belt_{year}.geojson")