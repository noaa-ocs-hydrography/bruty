import pathlib
import os
import shutil
from functools import partial

import pytest
import numpy
from osgeo import gdal, osr

from nbs.scripts.tile_specs import TileInfo
from nbs.bruty import tile_export

from xipe_dev.xipe.raster import CONTRIBUTOR_BAND_NAME, ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME
from test_data import master_data, make_clean_dir, data_dir, SW_5x5, NW_5x5, SE_5x5, MID_5x5
from nbs.configs import get_logger, iter_configs, set_file_logging, log_config, parse_multiple_values

LOGGER = get_logger("nbs.bruty.tests")

nan = numpy.nan
os.makedirs(data_dir, exist_ok=True)

norfolk_spherical = (-8494906.0, 4419895.0)
norfolk_utm_18 = (383135.613, 4080370.689)
norfolk_ll = (-76.311039, 36.862043)


def make_1m_tile(self, tx, ty, tile_history):
    dx = int(tile_history.max_x - tile_history.min_x) + 1
    dy = int(tile_history.max_y - tile_history.min_y) + 1
    # return number of rows, cols
    return dy, dx


simple_utm_dir = data_dir.joinpath('simple_utm3_db')
custom_scoring_dir = data_dir.joinpath('custom_scoring_db')
aoi_dir = data_dir.joinpath('aoi_db')

def test_complete_export_tiled():
    use_block_size = 1024  # tile_export.DEFAULT_BLOCK_SIZE
    rows = cols = use_block_size * 3
    bands = 3
    res = 16
    epsg = 26919
    tile_info = TileInfo(production_branch="PBB", utm=19, tile=22, datum='MLLW', build=True, locality='test',
                         resolution="16", closing_distance=[use_block_size*16], out_of_date="", change_summary="",
                         st_srid=26919, geometry=None, combine_public=True, combine_internal=True, combine_navigation=True)
    nav_export = tile_export.RasterExport(data_dir, tile_info, tile_export.NAVIGATION, '2022_', '1117_c')
    driver = gdal.GetDriverByName("GTiff")
    os.makedirs(nav_export.extracted_filename.parent, exist_ok=True)
    try:
        os.remove(nav_export.extracted_filename)
    except:
        pass
    new_ds = driver.Create(str(nav_export.extracted_filename), xsize=cols, ysize=rows, bands=bands, eType=gdal.GDT_Float32,
                           options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
    # Get raster projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    # Set projection
    new_ds.SetProjection(dest_wkt)
    new_ds.SetGeoTransform((0, res, 0, rows * res, 0, -res))
    # save memory and use the same integer values for depth, uncertainty and contributor
    data = numpy.zeros([rows, cols])
    data.fill(1000000)
    point_size = int(use_block_size/100)  # just to make the data easily visible
    for r, c in [(0, 0), (int(rows/2), 0), (int(rows/2), int(cols/2)), (rows-point_size-1, cols-point_size-1), (rows-point_size-1, 0)]:
        for j in range(point_size):
            data[r+j, c:c+point_size] = 1 + j
    # make a fake contributor list.  We made integers but now have to pretend they are ints encoded as floats - so convert them
    contribs = numpy.unique(data)
    int_contribs = numpy.sort(numpy.frombuffer(contribs.astype(numpy.float32).tobytes(), numpy.int32)).tolist()
    fake_contributors = {n: {'from_filename': "test"} for n in int_contribs}
    names = ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME, CONTRIBUTOR_BAND_NAME
    for n in range(new_ds.RasterCount):
        band = new_ds.GetRasterBand(n + 1)
        band.WriteArray(data)
        band.SetNoDataValue(1000000)
        band.SetDescription(names[n])
        band.ComputeStatistics(0)
        del band

    del new_ds

    tile_export.complete_export_tiled(nav_export, fake_contributors, use_block_size * res, epsg=epsg, decimals=2, block_size=use_block_size, debug_plots=True)


