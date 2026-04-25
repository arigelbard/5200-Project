import rasterio
import numpy as np

pixel_area_ha = (30**2) / 10_000

for year in [2006, 2007, 2008, 2009, 2010, 2011, 2012]:
    path = f'../data/processed-data/conversion_layer_{year}.tif'
    with rasterio.open(path) as src:
        data = src.read(1)
    converted = np.sum(data == 1) * pixel_area_ha
    print(f"{year}: {converted:,.0f} ha converted since 2006 baseline")