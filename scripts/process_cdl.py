"""
CDL Processing Pipeline
------------------------
Produces two maps showing grassland/wetland land use change across the
Corn Belt between 2006 and 2012:
  1. Pixel-level map of converted vs. retained natural land
  2. County-level choropleth of conversion rate

Steps:
  1. Reproject all files to EPSG:5070 (NAD83 Albers)
  2. Resample all 2006 files from 56m -> 30m (nearest neighbor)
  3. Mosaic all 11 states into one raster per year
  4. Compute conversion layer
  5. Visualize pixel-level map
  6. Aggregate to county level and visualize choropleth

Requirements:
    pip install rasterio numpy matplotlib geopandas rasterstats

Outputs (written to ../outputs/):
    cdl_conversion_map.png          — pixel-level map
    cdl_conversion_county_map.png   — county-level choropleth
"""

# ── Imports ───────────────────────────────────────────────────────────────────

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
YEARS  = [2006, 2012]

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

# ── Helper: build file path from state + year ─────────────────────────────────

def get_tif_path(state, year):
    """
    Constructs the path based on observed naming pattern:
      BASE_DIR / IA / cdl_56m_r_ia_2006_albers.tif
    2006 files use 56m resolution, 2012 files use 30m resolution.
    """
    state_lower = state.lower()
    res = '56m' if year == 2006 else '30m'
    filename = f'cdl_{res}_r_{state_lower}_{year}_albers.tif'
    return BASE_DIR / state / filename

# ═══════════════════════════════════════════════════════════════════════════════
# EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

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
    print("\n✓ All 22 files loaded successfully")

if rows:
    df = pd.DataFrame(rows)

    print("\n── Pixel counts for key land cover codes ──\n")
    display_cols = ['state', 'year', 'Corn', 'Grassland/Pasture',
                    'Woody Wetlands', 'Herbaceous Wetlands']
    print(df[display_cols].to_string(index=False))

    print("\n── CRS and resolution check ──")
    print("Unique CRS values found:", df['crs'].unique())
    print("Unique resolutions (m):", df['res_m'].unique())
    if df['crs'].nunique() > 1:
        print("⚠ WARNING: Not all files share the same CRS!")
        print("  You will need to reproject before merging.")
    else:
        print("✓ All files share the same CRS — safe to merge")

    print("\n── Grassland + Wetland totals by state ──")
    df['natural_land'] = (df['Grassland/Pasture'] +
                          df['Woody Wetlands'] +
                          df['Herbaceous Wetlands'])
    pivot = df.pivot_table(index='state', columns='year', values='natural_land')
    pivot.columns = ['natural_2006', 'natural_2012']
    pivot['pixel_loss'] = pivot['natural_2006'] - pivot['natural_2012']
    pivot['pct_loss']   = (pivot['pixel_loss'] / pivot['natural_2006'] * 100).round(2)
    pivot = pivot.sort_values('pixel_loss', ascending=False)
    print(pivot.to_string())
    print("\nNote: pixel counts — multiply by (res_m²) to convert to area")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 & 2: REPROJECT + RESAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

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

for year in [2006, 2012]:
    print(f"  Processing {year}:")
    for state in STATES:
        src_path = get_tif_path(state, year)
        dst_path = temp_dir / f'{state}_{year}_processed.tif'

        if dst_path.exists():
            print(f"    {state} — already processed, skipping")
            continue

        process_file(src_path, dst_path, TARGET_CRS, TARGET_RES)
        print(f"    {state} ✓")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: MOSAIC
# ═══════════════════════════════════════════════════════════════════════════════

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

mosaic_2006_path = PROCESSED_DIR / 'mosaic_2006.tif'
mosaic_2012_path = PROCESSED_DIR / 'mosaic_2012.tif'

if not mosaic_2006_path.exists():
    build_mosaic(2006, mosaic_2006_path)
else:
    print("  mosaic_2006.tif already exists, skipping")

if not mosaic_2012_path.exists():
    build_mosaic(2012, mosaic_2012_path)
else:
    print("  mosaic_2012.tif already exists, skipping")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: COMPUTE CONVERSION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 4: Computing conversion layer...")

"""
Conversion logic — for each pixel:
  - Was it grassland or wetland in 2006?  (natural land)
  - Is it corn in 2012?                   (converted)

Output pixel values:
  0 = no change / not relevant
  1 = converted (was natural land, now corn)  <- the story
  2 = was natural land in 2006, still natural in 2012 (retained)
"""

conversion_path = PROCESSED_DIR / 'conversion_layer.tif'

# Derive codes from CODES_OF_INTEREST once, before the loop
corn_code    = [code for code, name in CODES_OF_INTEREST.items() if name == 'Corn'][0]
natural_codes = [code for code, name in CODES_OF_INTEREST.items() if name != 'Corn']

with rasterio.open(mosaic_2006_path) as src_2006, \
     rasterio.open(mosaic_2012_path) as src_2012:

    # Warn if dimensions don't match — will be handled per-chunk below
    if src_2006.width != src_2012.width or src_2006.height != src_2012.height:
        print(f"  Note: mosaic dimensions differ "
              f"(2006: {src_2006.width}x{src_2006.height}, "
              f"2012: {src_2012.width}x{src_2012.height}) "
              f"— resampling 2012 to match 2006")

    meta = src_2006.meta.copy()
    meta.update({'compress': 'lzw', 'dtype': 'uint8'})

    with rasterio.open(conversion_path, 'w', **meta) as dst:

        chunk_size = 2048

        for row_off in range(0, src_2006.height, chunk_size):
            actual_height = min(chunk_size, src_2006.height - row_off)
            window = rasterio.windows.Window(
                col_off=0,
                row_off=row_off,
                width=src_2006.width,
                height=actual_height
            )

            data_2006 = src_2006.read(1, window=window)

            # Read 2012 chunk forced to exactly match 2006 chunk dimensions
            data_2012 = src_2012.read(
                1,
                window=window,
                out_shape=(actual_height, src_2006.width),
                resampling=Resampling.nearest
            )

            # Boolean masks
            was_natural = np.isin(data_2006, natural_codes)
            is_corn_now = (data_2012 == corn_code)

            # Build output layer
            result = np.zeros_like(data_2006, dtype=np.uint8)
            result[was_natural & is_corn_now]  = 1   # converted to corn
            result[was_natural & ~is_corn_now] = 2   # natural land retained

            dst.write(result, 1, window=window)

            pct = min(100, int((row_off + actual_height) / src_2006.height * 100))
            print(f"  Processing... {pct}%", end='\r')

print("\n  Conversion layer saved:", conversion_path)

# Delete temp files now that mosaics and conversion layer are saved
if temp_dir.exists() and temp_dir.is_dir():
    shutil.rmtree(temp_dir)
    print("  Temp files cleaned up")

# Quick stats
with rasterio.open(conversion_path) as src:
    data = src.read(1)
    pixel_area_ha = (TARGET_RES ** 2) / 10_000

converted_pixels = np.sum(data == 1)
retained_pixels  = np.sum(data == 2)
converted_ha     = converted_pixels * pixel_area_ha
retained_ha      = retained_pixels  * pixel_area_ha

print(f"\n  Natural land converted to corn: {converted_ha:,.0f} hectares")
print(f"  Natural land retained:          {retained_ha:,.0f} hectares")
print(f"  Conversion rate:                {converted_ha/(converted_ha+retained_ha)*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: PIXEL-LEVEL MAP
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 5: Generating pixel-level map...")

PLOT_DOWNSAMPLE = 5

with rasterio.open(conversion_path) as src:
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

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: COUNTY-LEVEL CHOROPLETH
# ═══════════════════════════════════════════════════════════════════════════════

print("\nStep 6: Generating county-level choropleth...")

# Load and prepare county shapefile
print("  Loading county shapefile...")
counties = gpd.read_file(RAW_DIR / 'tl_2012_us_county.zip')
counties = counties[counties['STATEFP'].isin(CORN_BELT_FIPS)].copy()
counties = counties.to_crs(TARGET_CRS)
counties['FIPS'] = counties['STATEFP'] + counties['COUNTYFP']
print(f"  Counties loaded: {len(counties)}")

# Zonal statistics — count converted (1) and retained (2) pixels per county
print("  Running zonal statistics (this may take a few minutes)...")
stats = zonal_stats(
    counties,
    str(conversion_path),
    categorical=True,
    nodata=0
)
print("  Done.")

# Build summary columns
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
print(f"\n  Top 10 counties by conversion rate:")
top10 = counties.nlargest(10, 'conversion_rate')[
    ['FIPS', 'NAME', 'STATEFP', 'conversion_rate', 'converted_ha']
]
print(top10.to_string(index=False))

# Plot
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

sm = plt.cm.ScalarMappable(
    cmap='YlOrRd',
    norm=mcolors.Normalize(vmin=0, vmax=30)
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.03, pad=0.02, aspect=40)
cbar.set_label('% of Natural Land (Grassland/Wetland) Converted to Corn', fontsize=10)
cbar.set_ticks([0, 5, 10, 15, 20, 25, 30])
cbar.set_ticklabels(['0%', '5%', '10%', '15%', '20%', '25%', '≥30%'])

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