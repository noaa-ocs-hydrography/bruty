import glob
import re

from osgeo import gdal

from xipe_dev.xipe.raster import attribute_tables

institutions = {0:{}, 1:{}}
all_fnames = [fname for fname in glob.glob("v:\\bruty_tile_exports\\**\\*.tif", recursive=True) if "original" not in fname]
use_files = {14:{}, 15:{}, 16:{}}
for fname in all_fnames:
    m = re.search(r"Tile(\d+)_PBG(\d+)_(\d+)m", fname)
    tile, zone, res = map(int, m.groups())
    try:
        if res < use_files[zone][tile][0]:
            use_files[zone][tile] = [res, fname]
    except KeyError:
        use_files[zone][tile] = [res, fname]

fnames = []
for zone in use_files:
    for tile in use_files[zone]:
        fnames.append(use_files[zone][tile][1])
for fname in fnames:
    print(fname)
    rat = attribute_tables.read_rat(fname)
    res = gdal.Open(fname).GetGeoTransform()[1]
    for survey in rat:
        cov = survey['coverage']
        cnt = survey['count']
        inst = survey['institution']
        if not inst:
            print(survey)
        institutions[cov][inst] = cnt * res * res + institutions[cov].setdefault(inst, 0)

for cov, data in institutions.items():
    raw = [(val, inst) for inst, val in data.items()]
    raw.sort()
    raw.reverse()
    print("\n\nCoverage = ", bool(cov), "\n")
    for val, inst in raw:
        print(inst, "%.1f"%(val/(1000*1000)))
