import rasterio
import rasterio.windows
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('../data/processed-data')
YEARS         = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
PIXEL_AREA_HA = (30 ** 2) / 10_000
CHUNK_SIZE    = 2048

def count_pixels(path, value):
    """Count pixels with a given value in a raster, using chunks."""
    total = 0
    with rasterio.open(path) as src:
        for row_off in range(0, src.height, CHUNK_SIZE):
            actual_height = min(CHUNK_SIZE, src.height - row_off)
            window = rasterio.windows.Window(
                col_off=0, row_off=row_off,
                width=src.width, height=actual_height
            )
            chunk = src.read(1, window=window)
            total += int(np.sum(chunk == value))
            del chunk
    return total

# ── Baseline: total natural land in 2006 ──────────────────────────────────────
# conversion_layer_2006 compares 2006 against itself so all natural pixels = 2
print("Computing 2006 natural land baseline...")
baseline_natural_px = count_pixels(PROCESSED_DIR / 'conversion_layer_2006.tif', 2)
baseline_nat_ha     = baseline_natural_px * PIXEL_AREA_HA
print(f"  Total natural land in 2006: {baseline_nat_ha:,.0f} ha")

# ── Per-year cumulative conversions ───────────────────────────────────────────
# conversion_layer_YEAR counts value-1 = all pixels that were natural in 2006
# and are corn in YEAR. This is the cumulative unique conversion count.
print("\nCounting cumulative conversions per year...")
cumulative = {}
for year in YEARS:
    px = count_pixels(PROCESSED_DIR / f'conversion_layer_{year}.tif', 1)
    cumulative[year] = px * PIXEL_AREA_HA
    print(f"  {year}: {cumulative[year]:,.0f} ha converted since 2006")

# ── Build interval table ──────────────────────────────────────────────────────
# Natural at start of interval = baseline - cumulative converted through prev year
# Converted in interval        = cumulative[year] - cumulative[prev_year]
# % converted                  = converted_in_interval / natural_at_start

print("\n── Incremental natural land converted to corn by interval ──\n")
print(f"{'Interval':<14} {'Converted (ha)':>16} {'Natural at Start (ha)':>22} "
      f"{'% Converted':>12} {'Cumulative (ha)':>16}")
print("-" * 84)

rows = []
for i, year in enumerate(YEARS[1:], 1):
    prev_year        = YEARS[i - 1]
    converted_ha     = cumulative[year] - cumulative[prev_year]
    natural_start_ha = baseline_nat_ha  - cumulative[prev_year]
    pct              = converted_ha / natural_start_ha * 100 if natural_start_ha > 0 else 0

    rows.append({
        'interval':          f'{prev_year}→{year}',
        'converted_ha':      converted_ha,
        'natural_start_ha':  natural_start_ha,
        'pct':               pct,
        'cumulative_ha':     cumulative[year],
    })

    print(f"{f'{prev_year}→{year}':<14} {converted_ha:>16,.0f} {natural_start_ha:>22,.0f} "
          f"{pct:>11.2f}% {cumulative[year]:>16,.0f}")

df = pd.DataFrame(rows)
print("-" * 84)
print(f"{'Total':<14} {df['converted_ha'].sum():>16,.0f}")
print(f"\nBaseline natural land (2006):  {baseline_nat_ha:,.0f} ha")
print(f"Total converted by 2012:       {cumulative[2012]:,.0f} ha  "
      f"({cumulative[2012] / baseline_nat_ha * 100:.1f}% of 2006 natural land)")