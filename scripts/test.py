import rasterio
import rasterio.windows
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path('../data/processed-data')
YEARS         = [2006, 2007, 2008, 2009, 2010, 2011, 2012]
PIXEL_AREA_HA = (30 ** 2) / 10_000
CHUNK_SIZE    = 2048
CORN_CODE     = 1
NATURAL_CODES = [176, 190, 195]

rows = []

for i, year in enumerate(YEARS[1:], 1):
    prev_year = YEARS[i - 1]
    prev_path = PROCESSED_DIR / f'mosaic_{prev_year}.tif'
    curr_path = PROCESSED_DIR / f'mosaic_{year}.tif'

    new_conversions   = 0   # natural → corn
    reversions        = 0   # corn → natural
    remaining_natural = 0   # natural pixels at start of interval
    corn_at_start     = 0   # corn pixels at start of interval

    with rasterio.open(prev_path) as prev_src, \
         rasterio.open(curr_path) as curr_src:

        for row_off in range(0, prev_src.height, CHUNK_SIZE):
            actual_height = min(CHUNK_SIZE, prev_src.height - row_off)
            window = rasterio.windows.Window(
                col_off=0, row_off=row_off,
                width=prev_src.width, height=actual_height
            )

            prev_chunk = prev_src.read(1, window=window)
            curr_chunk = curr_src.read(
                1, window=window,
                out_shape=(actual_height, prev_src.width),
                resampling=rasterio.enums.Resampling.nearest
            )

            was_natural = np.isin(prev_chunk, NATURAL_CODES)
            was_corn    = (prev_chunk == CORN_CODE)
            is_natural  = np.isin(curr_chunk, NATURAL_CODES)
            is_corn     = (curr_chunk == CORN_CODE)

            # Natural → corn (conversion)
            new_conversions   += int(np.sum(was_natural & is_corn))
            # Corn → natural (reversion)
            reversions        += int(np.sum(was_corn & is_natural))
            # Natural land at start of interval
            remaining_natural += int(np.sum(was_natural))
            # Corn at start of interval (denominator for reversion %)
            corn_at_start     += int(np.sum(was_corn))

            del prev_chunk, curr_chunk

    converted_ha     = new_conversions * PIXEL_AREA_HA
    reverted_ha      = reversions      * PIXEL_AREA_HA
    nat_start_ha     = remaining_natural * PIXEL_AREA_HA
    corn_start_ha    = corn_at_start     * PIXEL_AREA_HA
    net_ha           = converted_ha - reverted_ha

    pct_converted = converted_ha / nat_start_ha  * 100 if nat_start_ha  > 0 else 0
    pct_reverted  = reverted_ha  / corn_start_ha * 100 if corn_start_ha > 0 else 0

    rows.append({
        'interval':       f'{prev_year}→{year}',
        'nat_start_ha':   nat_start_ha,
        'corn_start_ha':  corn_start_ha,
        'converted_ha':   converted_ha,
        'pct_converted':  pct_converted,
        'reverted_ha':    reverted_ha,
        'pct_reverted':   pct_reverted,
        'net_ha':         net_ha,
    })

    print(f"  {prev_year}→{year}: "
          f"converted {converted_ha:,.0f} ha, "
          f"reverted {reverted_ha:,.0f} ha, "
          f"net {net_ha:,.0f} ha")

df = pd.DataFrame(rows)

print("\n── Year-over-year natural land flows ──\n")
print(f"{'Interval':<14} {'Nat→Corn (ha)':>14} {'% of Nat':>10} "
      f"{'Corn→Nat (ha)':>14} {'% of Corn':>10} {'Net (ha)':>12}")
print("-" * 78)
for _, r in df.iterrows():
    print(f"{r['interval']:<14} "
          f"{r['converted_ha']:>14,.0f} {r['pct_converted']:>9.2f}% "
          f"{r['reverted_ha']:>14,.0f} {r['pct_reverted']:>9.2f}% "
          f"{r['net_ha']:>12,.0f}")
print("-" * 78)
print(f"{'Total':<14} {df['converted_ha'].sum():>14,.0f} {'':>10} "
      f"{df['reverted_ha'].sum():>14,.0f} {'':>10} "
      f"{df['net_ha'].sum():>12,.0f}")