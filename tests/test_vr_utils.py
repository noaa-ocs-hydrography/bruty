import pathlib
import os
import shutil
from functools import partial

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
    use_dir = make_clean_dir('test_upsampling5')
    vr_path = data_dir.parent.joinpath("H-10771.bag")
    # data ranges from 2 to 7m so choosing 3 for initial test
    res = 3
    # vr_to_points_and_mask(vr_path, use_dir.joinpath("upsampled.tif"), use_dir.joinpath("mask.tif"), res)
    upsample_vr(vr_path, use_dir.joinpath("interpolated.tif"), res)
