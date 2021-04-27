import pathlib
import os
import shutil
from functools import partial
import time

import pytest
import numpy
from osgeo import gdal

from nbs.bruty.history import DiskHistory, MemoryHistory, RasterHistory
from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from nbs.bruty.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, \
    TMSMercatorTileBackend, merge_arrays
from nbs.bruty.world_raster_database import WorldDatabase, onerr, get_geotransformer, UTMTileBackendExactRes, CustomArea
from nbs.bruty.vr_utils import upsample_vr, vr_to_points_and_mask

from test_data import master_data, make_clean_dir, data_dir, SW_5x5, NW_5x5, SE_5x5, MID_5x5

nan = numpy.nan
os.makedirs(data_dir, exist_ok=True)

def test_upsample():
    vr_path = data_dir.parent.joinpath("H-10771.bag")
    vr_path = r"G:\Data\NBS\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag"  # r"G:\Data\survey_outlines\bugs\H13212_MB_VR_MLLW.bag"
    res = 5
    t1 = time.time()
    use_dir = make_clean_dir('test_just_mask2numba')
    vr_to_points_and_mask(vr_path, use_dir.joinpath("upsampled.tif"), use_dir.joinpath("quick_mask.tif"), res, block_size=512, nbs_mask=False)
    t2 = time.time()
    use_dir = make_clean_dir('test_full_nbs2numba')
    upsample_vr(vr_path, use_dir.joinpath("interpolated.tif"), res, block_size=512)
    t3 = time.time()
    print("just mask", t2-t1)
    print("nbs", t3-t2)

if __name__ == '__main__':
    test_upsample()