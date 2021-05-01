import pathlib
import os
import shutil
from functools import partial
import time

import pytest
import numpy
from osgeo import gdal, osr

from HSTB.drivers import bag
from nbs.bruty.history import DiskHistory, MemoryHistory, RasterHistory
from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from nbs.bruty.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, \
    TMSMercatorTileBackend, merge_arrays
from nbs.bruty.world_raster_database import WorldDatabase, onerr, get_geotransformer, UTMTileBackendExactRes, CustomArea
from nbs.bruty import vr_utils

from test_data import master_data, make_clean_dir, data_dir, SW_5x5, NW_5x5, SE_5x5, MID_5x5

nan = numpy.nan
os.makedirs(data_dir, exist_ok=True)


def test_draw_right_triangle():
    output = numpy.zeros((5, 5), dtype=numpy.int32)
    pts = numpy.array([[1, 1, 1], [4, 1, 1], [4, 4, 1]], dtype=numpy.int32)
    vr_utils.draw_triangle(output, pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1]], dtype=numpy.int32)
    assert (result == output).all()


def test_draw_triangle():
    output = numpy.zeros((5, 5), dtype=numpy.int32)
    pts = numpy.array([[2, 0, 1], [4, 2, 1], [0, 4, 1]], dtype=numpy.int32)
    vr_utils.draw_triangle(output, pts, 1)
    result = numpy.array([[0, 0, 0, 0, 1],
                          [0, 0, 1, 1, 0],
                          [1, 1, 1, 1, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 1, 0, 0]], dtype=numpy.int32)
    assert (result == output).all()


def test_draw_rect():
    output = numpy.zeros((5, 5), dtype=numpy.int32)
    pts = numpy.array([[1, 1, 1], [4, 1, 1], [4, 4, 1], [1, 4, 1]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=numpy.int32)
    assert (result == output).all()


def test_partial_draw_rect():
    output = numpy.zeros((5, 5), dtype=numpy.int32)
    # turn off a corner (make bool in third position = 0) to make sure it draws triangles when only three points exist
    # Lower left
    pts = numpy.array([[1, 1, 1], [4, 1, 1], [4, 4, 1], [1, 4, 0]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 1, 1]], dtype=numpy.int32)
    assert (result == output).all()

    # Lower right
    output *= 0  # clear the output
    pts = numpy.array([[1, 1, 0], [4, 1, 1], [4, 4, 1], [1, 4, 1]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [0, 0, 0, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 1, 1, 1, 1]], dtype=numpy.int32)
    assert (result == output).all()
    # upper left
    output *= 0  # clear the output
    pts = numpy.array([[1, 1, 1], [4, 1, 1], [4, 4, 0], [1, 4, 1]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 1, 1, 0, 0],
                          [0, 1, 0, 0, 0]], dtype=numpy.int32)
    assert (result == output).all()
    # upper right
    output *= 0  # clear the output
    pts = numpy.array([[1, 1, 1], [4, 1, 0], [4, 4, 1], [1, 4, 1]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [0, 1, 1, 1, 1],
                          [0, 0, 1, 1, 1],
                          [0, 0, 0, 1, 1],
                          [0, 0, 0, 0, 1]], dtype=numpy.int32)
    assert (result == output).all()


def test_draw_parallelogram():
    output = numpy.zeros((5, 5), dtype=numpy.int32)
    pts = numpy.array([[1, 0, 1], [4, 2, 1], [4, 4, 1], [1, 2, 1]], dtype=numpy.int32)
    vr_utils.vr_close_quad(output, *pts, 1)
    result = numpy.array([[0, 0, 0, 0, 0],
                          [1, 1, 1, 0, 0],
                          [0, 1, 1, 0, 0],
                          [0, 0, 1, 1, 0],
                          [0, 0, 1, 1, 1]], dtype=numpy.int32)
    assert (result == output).all()


def test_inside_refinement():
    # make a 4x4 refinement
    # map it to a 12x12 output
    # could use numpy.linspace for laying out the r,c pattern
    r = numpy.array([[1, 1, 1, 1],
                     [4, 4, 4, 4],
                     [7, 7, 7, 7],
                     [10,10, 10,10]], dtype=numpy.int32)
    c = numpy.array([[1, 4, 7, 10,],
                     [1, 4, 7, 10, ],
                     [1, 4, 7, 10, ],
                     [1, 4, 7, 10, ]], dtype=numpy.int32)
    b = numpy.array([[1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1],
                     [1, 1, 1, 1]], dtype=numpy.int32)
    ref = numpy.array([r,c,b], dtype=numpy.int32)
    output = numpy.zeros((12, 12), dtype=numpy.int32)
    result = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          ])
    vr_utils.vr_triangle_closing(output, ref, 1)
    assert (result == output).all()
    # add a hole in the middle
    b[2,2] = 0
    ref = numpy.array([r, c, b], dtype=numpy.int32)
    output *= 0  # clear the output
    result = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,0,1,1,1,0],
                          [0,1,1,1,1,1,0,0,0,1,1,0],
                          [0,1,1,1,1,0,0,0,0,0,1,0],
                          [0,1,1,1,1,1,0,0,0,1,1,0],
                          [0,1,1,1,1,1,1,0,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          ])
    vr_utils.vr_triangle_closing(output, ref, 1)
    assert (result == output).all()

def test_close_gap():
    # close the gap between two refinements, this does not add the points from the refinements just the gap between them
    # (and the last rows as a side effect)
    r = numpy.array([[1, 1, 1, 1],
                     [4, 4, 4, 4],], dtype=numpy.int32)
    c = numpy.array([[1, 4, 7, 10,],
                     [1, 4, 7, 10, ]], dtype=numpy.int32)
    b = numpy.array([[1, 1, 1, 1],
                     [1, 1, 1, 1]], dtype=numpy.int32)
    ref_top = numpy.array([r,c,b], dtype=numpy.int32)
    ref_top_xyz = ref_top
    r2 = numpy.array([[7, 7, 7, 7],
                     [10,10, 10,10]], dtype=numpy.int32)
    ref_bottom = numpy.array([r2,c,b], dtype=numpy.int32)
    ref_bottom_xyz = ref_bottom

    output = numpy.zeros((12, 12), dtype=numpy.int32)
    result = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          ])
    vr_utils.vr_close_refinements(output, ref_top, ref_top_xyz, ref_bottom, ref_bottom_xyz, 100, 1, horz=False)
    assert (result == output).all()

    # with a hole on the edge
    b2 = numpy.array([[1, 1, 0, 1],
                     [1, 1, 1, 1]], dtype=numpy.int32)
    ref_bottom = numpy.array([r2,c,b2], dtype=numpy.int32)
    ref_bottom_xyz = ref_bottom

    output = numpy.zeros((12, 12), dtype=numpy.int32)
    result = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,1,1,1,1,0],
                          [0,1,1,1,1,1,1,0,0,1,1,0],
                          [0,1,1,1,1,0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          ])
    vr_utils.vr_close_refinements(output, ref_top, ref_top_xyz, ref_bottom, ref_bottom_xyz, 100, 1, horz=False)
    assert (result == output).all()

    # vertical gap closing
    r = numpy.array([[1, 1,],
                     [4, 4, ],
                     [7, 7, ],
                     [10, 10,],], dtype=numpy.int32)
    c = numpy.array([[1, 4,],
                     [1, 4, ],
                     [1, 4, ],
                     [1, 4,]], dtype=numpy.int32)
    b = numpy.array([[1, 1,],
                     [1, 1,],
                     [1, 1,],
                     [1, 1,]], dtype=numpy.int32)
    ref_top = numpy.array([r,c,b], dtype=numpy.int32)
    ref_top_xyz = ref_top
    c2 = c + 6
    ref_bottom = numpy.array([r,c2,b], dtype=numpy.int32)
    ref_bottom_xyz = ref_bottom
    output = numpy.zeros((12, 12), dtype=numpy.int32)
    result = numpy.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,1,1,1,1,0,0,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0],
                          ])
    vr_utils.vr_close_refinements(output, ref_top, ref_top_xyz, ref_bottom, ref_bottom_xyz, 100, 1, horz=True)
    assert (result == output).all()


def test_vr_to_points():
    # make a VR that covers 48m x 48m to export at 1m.
    # Each supergrid refinement covers 24x24, so 2x2 super grids.
    # Make the refinements roughly be 24m (per cell), 8m (3x3), 2m (12x12) and 1m (24x24)
    use_dir = make_clean_dir("vr_tests12")
    vr = bag.VRBag.new_bag(use_dir.joinpath("fake_vr.bag"), mode="w")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26916)  # UTM 16 - so working in meters
    vr.horizontal_crs_wkt = srs.ExportToWkt()
    vr.set_res((24,24))
    vr.set_origin((12,12))
    vr.set_refinements([[None, None], [None, None]])  # initialize a 2x2 VR

    # lower right quadrant - 8m spacing - inset by 4m so only 2x2 data
    depths=numpy.ones([2,2], dtype=numpy.float32)
    uncertainties = depths
    ref_8m = bag.Refinement(depths, uncertainties, 8, 8, 4, 4)
    vr.set_refinement(0,1, ref_8m)
    ref = vr.get_refinement(0, 1)
    assert ref.geotransform == (28, 8, 0, 4, 0, 8)

    # lower left quadrant - 1m spacing - inset by .25m so only 23x23 data
    depths=numpy.ones([23,23], dtype=numpy.float32)
    uncertainties = depths
    ref_1m = bag.Refinement(depths, uncertainties, 1, 1, .25, .25)
    vr.set_refinement(0,0, ref_1m)
    ref = vr.get_refinement(0, 0)
    assert ref.geotransform == (.25, 1, 0, .25, 0, 1)

    # upper left quadrant - 2m spacing - inset by 1.75m (1m normnal + .75 extra) so only 11x11 data
    depths=numpy.ones([11,11], dtype=numpy.float32)
    uncertainties = depths
    ref_2m = bag.Refinement(depths, uncertainties, 2, 2, .75, 24.75)
    vr.set_refinement(1,0, ref_2m, local_offset=False)
    ref = vr.get_refinement(1, 0)
    assert ref.geotransform == (.75, 2, 0, 24.75, 0, 2)

    # upper right quadrant - 22m spacing - inset by 1m so only 1 data point
    depths=numpy.ones([1,1], dtype=numpy.float32)
    uncertainties = depths
    ref_22m = bag.Refinement(depths, uncertainties, 20, 20, 1, 1)
    vr.set_refinement(1,1, ref_22m)
    ref = vr.get_refinement(1, 1)
    assert ref.geotransform == (25, 20, 0, 25, 0, 20)

    sr_path = use_dir.joinpath('sr.tif')
    vr_utils.vr_to_sr_points(vr, sr_path, 1)
    ds = gdal.Open(str(sr_path))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    assert data.shape[0] == 48
    assert data.shape[1] == 48
    data_ud = numpy.flipud(data)  # flip since the tif is written top down
    # the lower left quad is fully filled
    assert (data_ud[0:23, 0:23]==1).all()
    # the upper left quad has 11x11  2m data, so every other cell is filled or nan
    assert numpy.count_nonzero(~numpy.isnan(data_ud[24:48, 0:24])) == 121  # 11x11 = 121 data points should be filled in the quadrant
    assert (data_ud[25:46:2, 1:22:2] == 1).all()
    assert numpy.isnan(data_ud[24:48:2, 0:24:2]).all()
    # the lower right has 2x2 at 8 meter spacing that was inset by 4 meters.
    assert numpy.count_nonzero(~numpy.isnan(data_ud[0:24, 24:48])) == 4  # 2x2 = 4 data points should be filled in the quadrant
    # make sure the data points are where they should be, 4m inset + 8m spacing on the lower row
    assert data_ud[4+4, 24+4+4] == 1  # row = 0 + 4m offset + half res (8/2),  col = 24 + 4m offset + half res (8/2)
    assert data_ud[4+4+8, 24+4+4] == 1
    assert data_ud[4+4, 24+4+4+8] == 1
    assert data_ud[4+4+8, 24+4+4+8] == 1
    # the upper right has 1x1 at 20 meter spacing that was inset by 1 meter.
    assert numpy.count_nonzero(~numpy.isnan(data_ud[25:48, 25:48])) == 1  # 1x1 = 1 data points should be filled in the quadrant
    # make sure the data points are where they should be, 1m inset + 20m res
    assert data_ud[24+1+10, 24+1+10] == 1  # row/col 1 --> 24 plus 1m offset plus half 20m res --> 10


def test_upsample():
    vr_path = data_dir.parent.joinpath("H-10771.bag")
    # vr_path = r"G:\Data\NBS\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag"  # r"G:\Data\survey_outlines\bugs\H13212_MB_VR_MLLW.bag"
    res = 5
    t1 = time.time()
    use_dir = make_clean_dir('test_just_mask2basic')
    vr_to_points_and_mask(vr_path, use_dir.joinpath("upsampled.tif"), use_dir.joinpath("quick_mask.tif"), res, block_size=512, nbs_mask=False)
    t2 = time.time()
    use_dir = make_clean_dir('test_full_nbs2basic')
    upsample_vr(vr_path, use_dir.joinpath("interpolated.tif"), res, block_size=512)
    t3 = time.time()
    print("just mask", t2 - t1)
    print("nbs", t3 - t2)


if __name__ == '__main__':
    test_upsample()
