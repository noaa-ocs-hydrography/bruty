import pathlib
import os
import shutil
from functools import partial

import pytest
import numpy
from osgeo import gdal, osr

from nbs.bruty.history import DiskHistory, MemoryHistory, RasterHistory
from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from nbs.bruty.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, \
    TMSMercatorTileBackend, merge_arrays
from nbs.bruty.world_raster_database import WorldDatabase, onerr, get_crs_transformer, UTMTileBackendExactRes, CustomArea

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


def test_merge_arrays():
    # basic test, merge 5x5 matrices and store the Z, uncertainty and flags.  Sort on score and Z
    r, c, z, uncertainty, score, flags = SW_5x5
    # store z, uncertainty, flags in the data (omit r,c, score)
    output_data = numpy.full([3, *(numpy.array(r.shape) * 2)], numpy.nan)
    # sort on score and depth so create storage space for those
    output_sort_values = numpy.full([2, *output_data.shape[1:]], numpy.nan)
    merge_arrays(r, c, (score, z), (z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[0, :5, :5] == SW_5x5[2]).all()

    r, c, z, uncertainty, score, flags = NW_5x5
    merge_arrays(r, c, (score, z), (z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[0, 5:10, :5] == NW_5x5[2]).all()

    r, c, z, uncertainty, score, flags = SE_5x5
    merge_arrays(r, c, (score, z), (z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[0, :5, 5:10] == SE_5x5[2]).all()

    r, c, z, uncertainty, score, flags = MID_5x5
    merge_arrays(r, c, (score, z), (z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[0, 2:5, 2:5] == 999).all()
    assert (output_data[0, 5:7, 5:7] == 999).all()
    assert (output_data[0, 5:10, :5] == NW_5x5[2]).all()
    assert (output_data[0, :5, 5:10] == SE_5x5[2]).all()


def test_sort_arrays():
    # merge 5x5 matrices and store the Z, uncertainty and flags.  Triple key sort on Z then row and col (X then Y)
    r0, c0, z, uncertainty, score, flags = SW_5x5
    uncertainty = uncertainty * 0 + 1
    # store x,y,z, uncertainty, flags in the data (omit score)
    output_data = numpy.full([5, *numpy.array(r0.shape)], numpy.nan)
    # sort on z,x,y so create storage space for those
    output_sort_values = numpy.full([3, *output_data.shape[1:]], numpy.nan)

    r = r0
    c = c0
    merge_arrays(r, c, (z, r, c), (r, c, z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[2, :5, :5] == SW_5x5[2]).all()

    # change the row which wins the tiebreaker when z is the same
    r = r0 + .3
    c = c0 + .3
    uncertainty = uncertainty * 0 + 2
    merge_arrays(r, c, (z, r, c), (r, c, z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[3, :5, :5] == 2).all()

    # change the col which wins the tiebreaker when z is the same
    r = r0 + .3
    c = c0 + .6
    uncertainty = uncertainty * 0 + 3
    merge_arrays(r, c, (z, r, c), (r, c, z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[3, :5, :5] == 3).all()

    # zero out elevation so nothing should be selected
    r = r0
    c = c0
    uncertainty = uncertainty * 0 + 4
    merge_arrays(r, c, (z * 0, r, c), (r, c, z * 0, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[3, :5, :5] == 3).all()

    # change uncertainty but none should be selected since all the Z, x, y are the same as what's in the merged array already
    r = r0 + .3
    c = c0 + .6
    uncertainty = uncertainty * 0 + 4
    merge_arrays(r, c, (z, r, c), (r, c, z, uncertainty, flags), output_data, output_sort_values)
    assert (output_data[3, :5, :5] == 3).all()


def test_sortkey_bounds():
    # merge 5x5 matrices and store the Z, uncertainty and flags.  Triple key sort on Z then row and col (X then Y)
    r0, c0, z, uncertainty, score, flags = SW_5x5
    uncertainty = uncertainty * 0 + 1
    # store x,y,z, uncertainty, flags in the data (omit score)
    output_data = numpy.full([5, *numpy.array(r0.shape)], numpy.nan)
    # sort on z,x,y so create storage space for those
    output_sort_values = numpy.full([3, *output_data.shape[1:]], numpy.nan)

    r = r0
    c = c0
    # test a filled Z bound, None for row bounds and partial (None, 3) for column bounds
    merge_arrays(r, c, (z, r, c), (r, c, z, uncertainty, flags), output_data, output_sort_values, key_bounds=[(10, 999), None, (None, 3)])
    # removed by the z key_bounds
    assert (numpy.isnan(output_data[2, :, 0])).all()
    assert (output_data[2, :, 1:4] == SW_5x5[2][:, 1:4]).all()
    # removed by the column key_bounds
    assert (numpy.isnan(output_data[2, :, 4])).all()


def test_db_json():
    use_dir = make_clean_dir("json_db")
    # @todo parameterize this
    # db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    db = CustomArea(None, 395813.20000000007, 3350563.9800000004, 406818.20000000007, 3343878.9800000004, 4, 4, use_dir)
    db2 = WorldDatabase.open(use_dir)
    assert db2.db.storage_class.__name__ == db.db.storage_class.__name__
    assert db2.db.history_class.__name__ == db.db.history_class.__name__
    assert db2.db.data_class.__name__ == db.db.data_class.__name__
    assert db2.db.tile_scheme.min_x == db.db.tile_scheme.min_x
    assert db2.db.tile_scheme.max_y == db.db.tile_scheme.max_y
    assert db2.db.tile_scheme.epsg == db.db.tile_scheme.epsg
    assert db2.db.tile_scheme.zoom == db.db.tile_scheme.zoom


def test_make_db():
    use_dir = simple_utm_dir
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # override the init_tile with a function to make approx 1m square resolution
    db.init_tile = partial(make_1m_tile, db)
    # NOTE: when using this coordinate tile the 0,0 position ends up in a cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the cells will end up in the -1,-1 index on the export

    r, c, z, uncertainty, score, flags = SW_5x5  # flip the r,c order since insert wants x,y
    db.insert_survey_array((c, r, z, uncertainty, score, flags), "origin")
    r, c, z, uncertainty, score, flags = SE_5x5
    db.insert_survey_array((c, r, z, uncertainty, score, flags), "east")
    # this +50 array should not show up as the score is the same but depth is deeper
    r, c, z, uncertainty, score, flags = SW_5x5
    db.insert_survey_array((c + 5, r, z + 50, uncertainty, score, flags), "east2")
    r, c, z, uncertainty, score, flags = NW_5x5
    db.insert_survey_array((c, r, z, uncertainty, score, flags), "north east")
    # overwrite some of the origin grid but keep score below east and northeast
    r, c, z, uncertainty, score, flags = MID_5x5
    db.insert_survey_array((c, r, z, uncertainty, score, flags), "overwrite origin")

    tx, ty = db.db.get_tiles_indices(0, 0, 0, 0)[0]
    tile = db.db.get_tile_history_by_index(tx, ty)
    raster_data = tile[-1]
    arr = raster_data.get_arrays()
    r0, c0 = raster_data.xy_to_rc(0, 0)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 2, c0:c0 + 2] == SW_5x5[2][:2, :2])
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)
    assert numpy.all(arr[0, r0:r0 + 5, c0 + 5:c0 + 10] == SE_5x5[2])
    assert numpy.all(arr[0, r0 + 5:r0 + 10, c0:c0 + 5] == NW_5x5[2])


def test_export_area():
    use_dir = simple_utm_dir
    # create the database if not existing
    if not os.path.exists(use_dir):
        test_make_db()

    # NOTE: when using this coordinate tile the 0,0 position ends up in a cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the cells will end up in the -1,-1 index on the export
    # db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    db = WorldDatabase.open(use_dir)
    # override the init_tile with a function to make approx 1m square resolution
    db.init_tile = partial(make_1m_tile, db)

    # NOTE: when using this coordinate tile the 0,0 x,y position ends up in a database cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the exported 0,0 will end up in the -1,-1 cell on the export
    db.export_area(use_dir.joinpath("new0.5.tif"), -1, -1, 11, 11, (.5, .5))
    db.export_area(use_dir.joinpath("new1.tif"), -1, -1, 11, 11, (1, 1))
    ds = gdal.Open(str(use_dir.joinpath("new1.tif")))
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray()
    r, c, z, uncertainty, score, flags = SW_5x5
    # r0, c0 = (-1, -1)
    # the database holds things in a positive Y convention so rows are flipped.
    # # @todo Could account for that here by checking the geotransform for DY sign
    assert numpy.all(arr[-1:-3:-1, 0: 2] == z[:2, :2])
    assert numpy.all(arr[-3:-6:-1, 2: 5] == 999)
    assert numpy.all(arr[-1:-6:-1, 5:10] == z + 100)
    assert numpy.all(arr[-6:-11:-1, 0: 5] == z + 200)
    assert numpy.isnan(arr[5, 7])

    # check the downsampling of export, this should put 4x4 cell blocks into each exported cell
    """index 111111111111111111  222222222222222222  33333333 
        1   [  0,  10,  20,  30,  40, 100, 110, 120, 130, 140],
        1   [  1,  11,  21,  31,  41, 101, 111, 121, 131, 141],
        1   [  2,  12, 999, 999, 999, 102, 112, 122, 132, 142],
        1   [  3,  13, 999, 999, 999, 103, 113, 123, 133, 143],
        2   [  4,  14, 999, 999, 999, 104, 114, 124, 134, 144],
        2   [200, 210, 220, 230, 240, 999, 999, nan, nan, nan],
        2   [201, 211, 221, 231, 241, 999, 999, nan, nan, nan],
        2   [202, 212, 222, 232, 242, nan, nan, nan, nan, nan],
        3   [203, 213, 223, 233, 243, nan, nan, nan, nan, nan],
        3   [204, 214, 224, 234, 244, nan, nan, nan, nan, nan]],

    Should yield (remembering scores of 999 was lower than 100 and 200 blocks:
    999  123  143
    232  242  144
    234  244  nan
    """
    db.export_area(use_dir.joinpath("new4.tif"), -1, -1, 11, 11, (4, 4), align=False)
    ds4 = gdal.Open(str(use_dir.joinpath("new4.tif")))
    b4 = ds4.GetRasterBand(1)
    a4 = b4.ReadAsArray()
    # can't use == because the nan (empty cell) fails
    assert a4[0, 0] == 234 and a4[0, 1] == 244
    assert numpy.isnan(a4[0, 2])
    assert numpy.all(a4[1:] == [[232, 242, 144], [999, 123, 143]])


# importlib.reload(tile_calculations)
# g = tile_calculations.GlobalGeodetic(zoom=2); f=g.xy_to_tile; print(f(-100, 40)); print(f(100, -40)); print(f(182, -91))
# g.tile_to_xy(0,0, 2)
# tx, ty = g.xy_to_tile_index(45, 45); print(tx,ty); print(g.tile_index_to_xy(tx, ty))
# from bruty.tile_calculations import TMSTilesMercator, GoogleTilesMercator
# merc = TMSTilesMercator(13)
# tx, ty = merc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(merc.tile_index_to_xy(tx, ty))
# gmerc = GoogleTilesMercator(13)
# tx, ty = gmerc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(gmerc.tile_index_to_xy(tx, ty))


@pytest.fixture(scope="module", params=list(master_data.values()), ids=list(master_data.keys()))
def data_lists(request):
    yield request.param


@pytest.fixture(scope="module")
def data_arrays(data_lists):
    npd = [numpy.array(data) for data in data_lists]
    for i, survey in enumerate(npd):
        while survey.ndim < 3:  # make sure it's 4 dimensional - first is survey, then LayerEnum then x, y
            survey = numpy.expand_dims(survey, axis=-1)
            npd[i] = survey
    return npd


@pytest.fixture(scope="module")
def data_rasters(data_arrays):
    return [RasterData.from_arrays(data) for data in data_arrays]


@pytest.fixture(scope="module", params=[
    (RasterHistory, MemoryHistory, MemoryStorage, ""),
    (RasterHistory, DiskHistory, TiffStorage, data_dir.joinpath('tiff_db'))
],
                ids=['memory', 'disk_tiff'])
def db_params(request):
    history_class, storage_class, data_class, data_path = request.param
    yield history_class, storage_class, data_class, data_path


@pytest.fixture(scope="module", params=[GoogleMercatorTileBackend, TMSMercatorTileBackend, LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend],
                ids=['google_merc', 'tms_mercator', 'lat_lon_tiles', 'google_spherical_tiles', 'utm_tiles'])
def history_db(request, db_params):
    tile_backend_class = request.param
    history_class, storage_class, data_class, data_path = db_params
    if data_path:
        data_path = data_path.joinpath(tile_backend_class.__name__)

    if issubclass(tile_backend_class, UTMTileBackend):
        world_db = tile_backend_class(32618, history_class, storage_class, data_class, data_path)
    else:
        world_db = tile_backend_class(history_class, storage_class, data_class, data_path)
    yield world_db


# @pytest.mark.parametrize("raster", [RasterData(MemoryStorage()), RasterData(TiffStorage(data_dir.joinpath("layer_order.tif")))], ids=["mem", "tiff"])
def test_indices(history_db):
    # test at 0,0 degrees and norfolk, va
    tms_origin_idx = ((2 ** history_db.tile_scheme.zoom) / 2, (2 ** history_db.tile_scheme.zoom) / 2)
    google_origin_idx = ((2 ** history_db.tile_scheme.zoom) / 2, (2 ** history_db.tile_scheme.zoom) / 2 - 1)  # y is flipped

    if isinstance(history_db, UTMTileBackend):
        norfolk_coord = norfolk_utm_18
        norfolk_idx = 3776, 3783
        # since range is -10mil to 20mil and -1mil to 10mil, the origin indx is about 1/3 of num tiles in x and 10% num tiles of Y
        origin_idx = 2730, 744
    elif isinstance(history_db, GoogleMercatorTileBackend):
        norfolk_coord = norfolk_spherical
        norfolk_idx = (2359, 3192)
        origin_idx = google_origin_idx
    elif isinstance(history_db, TMSMercatorTileBackend):
        norfolk_coord = norfolk_spherical
        norfolk_idx = (2359, 4999)
        origin_idx = tms_origin_idx
    elif isinstance(history_db, GoogleLatLonTileBackend):
        norfolk_coord = norfolk_ll
        norfolk_idx = (2359, 2418)  # this runs -90 to 90 in Y unlike spherical which ends at 85 degrees.  X tile is same since both go -180 to 180.
        origin_idx = google_origin_idx
    elif isinstance(history_db, LatLonBackend):
        norfolk_coord = norfolk_ll
        norfolk_idx = (2359, 5773)  # this runs -90 to 90 in Y unlike spherical which ends at 85 degrees.  X tile is same since both go -180 to 180.
        origin_idx = tms_origin_idx

    xx, yy = history_db.get_tiles_index_sparse(0, 0, *norfolk_coord)
    assert xx[0] == origin_idx[0] or xx[-1] == origin_idx[0]
    assert yy[0] == origin_idx[1] or yy[-1] == origin_idx[1]
    assert xx[0] == norfolk_idx[0] or xx[-1] == norfolk_idx[0]
    assert yy[0] == norfolk_idx[1] or yy[-1] == norfolk_idx[1]
    x1, y1, x2, y2 = history_db.tile_scheme.tile_index_to_xy(*norfolk_idx)
    assert norfolk_coord[0] >= x1 and norfolk_coord[0] <= x2
    assert norfolk_coord[1] >= y1 and norfolk_coord[1] <= y2


arr1 = numpy.zeros((5, 3, 5))
arr2 = arr1 + 1
arr3 = arr2 + 1
arr1[:, 0] = numpy.nan
arr2[:, 1] = numpy.nan
r0 = RasterData.from_arrays(arr1)
r1 = RasterData.from_arrays(arr2)
r2 = RasterData.from_arrays(arr3)


def fill_tile_history(history):
    # if issubclass(history_db.storage_class, DiskHistory):
    history.clear()
    history.append(r0)
    history.append(r1)
    history.append(r2)

    # # Try a reload from disk if applicable
    # if issubclass(history_db.storage_class, DiskHistory):
    #     del history
    #     history = history_db.get_tile_history(x1, y1)
    assert numpy.all(arrays_match(history[0].get_arrays(), r0.get_arrays()))
    assert numpy.all(arrays_match(history[2].get_arrays(), r2.get_arrays()))


def test_add_data(history_db):
    if isinstance(history_db, UTMTileBackend):
        norfolk_coord = norfolk_utm_18
    elif isinstance(history_db, GoogleMercatorTileBackend):
        norfolk_coord = norfolk_spherical
    elif isinstance(history_db, TMSMercatorTileBackend):
        norfolk_coord = norfolk_spherical
    elif isinstance(history_db, GoogleLatLonTileBackend):
        norfolk_coord = norfolk_ll
    elif isinstance(history_db, LatLonBackend):
        norfolk_coord = norfolk_ll
    x1 = norfolk_coord[0]
    y1 = norfolk_coord[1]
    # try getting and filling a tile based on the x,y
    history = history_db.get_tile_history(x1, y1)
    fill_tile_history(history)

    # try getting a 2x2 range of tiles around the x,y and see if they work
    x2 = x1 + history_db.tile_scheme.width() / history_db.tile_scheme.num_tiles()
    y2 = y1 + history_db.tile_scheme.height() / history_db.tile_scheme.num_tiles()
    indices = history_db.get_tiles_indices(x1, y1, x2, y2)

    for tx, ty in indices:
        history = history_db.get_tile_history_by_index(tx, ty)
        fill_tile_history(history)


# def test_pbc19_tile_4():
#     use_dir = make_clean_dir('tile4_metadata_utm_db')
#
#     db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
#
#     if False:
#         w, n = 288757.22, 4561186.99
#         e, s = 297025.15, 4547858.16
#         db.insert_survey_array(numpy.array(([w, e], [n, s], [45, 33], [2, 2], [10, 10], [0, 0])),
#                                r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_2m_MLLW_2of3.bag_corners")
#         # # should be NW in tile 3519,4141 and SE in 3541,4131 for UTM 19
#         raise Exception("stopping")
#     for bag_file in [r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_2m_MLLW_2of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_4m_MLLW_3of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\RI_16_BHR_20190417_BD_2019_023_FULL_4m_interp.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\RI_17_GSP_20190418_BD_2019_022_FULL_4m_interp.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_1of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_2of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_3of3.bag",
#                      # r"C:\Data\nbs\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_MB_2m_MLLW_2of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_MB_50cm_MLLW_1of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_VB_4m_MLLW_3of3.bag",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_1m_MLLW_1of3.bag", ]:
#         print('processsing grid', bag_file)
#         db.insert_survey_gdal(bag_file)
#
#     for txt_file in [r"C:\Data\nbs\PBC19_Tile4_surveys\D00111.csar.du.txt",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H06443.csar.du.txt",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H08615.csar.du.txt",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\F00363.csar.du.txt",
#                      r"C:\Data\nbs\PBC19_Tile4_surveys\H06442.csar.du.txt", ]:
#         print('processsing txt', txt_file)
#         db.insert_txt_survey(txt_file, override_epsg=4326)
#
#
# def test_pbc19_vr():
#     use_dir = make_clean_dir('tile4_VR_utm_db')
#     db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
#     db.insert_survey_vr(r"C:\Data\nbs\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag")
#     print("processed_vr")


def test_export_area_full_db():
    use_dir = make_clean_dir('tile4_vr_utm_db')
    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    db.export_area(use_dir.joinpath("export_tile_new.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)


def test_exact_res_db():
    use_dir = make_clean_dir('tile4_exact')
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)
    db = WorldDatabase(UTMTileBackendExactRes(4, 4, 26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    x = y = depth = uncertainty = score = flags = numpy.arange(10)
    db.insert_survey_array((x, y * 300, depth, uncertainty, score, flags), "test")


def test_custom_area():
    use_dir = make_clean_dir('custom_area_0.5_c')
    x1, y1, x2, y2 = extents = (641430.0, 4542174.0, 646498.0, 4547326.0)
    resolution = 0.5
    epsg = 26918
    cust_db = CustomArea(epsg, *extents, resolution, resolution, use_dir)
    # make an edge
    x = numpy.arange(x1, x2, resolution)
    y = numpy.arange(y1, y2, resolution)
    # only the lower left corner
    # x = numpy.arange(x1, x1+50, resolution)
    # y = numpy.arange(y1, y1+50, resolution)
    valy = numpy.full(y.shape, 1.0)
    left_edge = (numpy.full(y.shape, x1), y, valy, valy, valy, valy)
    valx = numpy.full(x.shape, 2.0)
    bottom_edge = (x, numpy.full(x.shape, y1), valx, valx, valx, valx)
    cust_db.insert_survey_array(left_edge, "left")
    cust_db.insert_survey_array(bottom_edge, "bottom")
    cust_db.export(use_dir.joinpath("export_all.tif"))
    cust_db.export(use_dir.joinpath("export_elev.tif"), layers=[LayersEnum.ELEVATION])


def custom_scoring(pts1, pts2):
    ret = []
    for pts in (pts1, pts2):
        contrib = pts[LayersEnum.CONTRIBUTOR].copy()
        elev = pts[LayersEnum.ELEVATION].copy()
        contrib[pts[LayersEnum.CONTRIBUTOR] == 2] = 10  # make the +50 array show up which was suppressed in the original test
        score = numpy.array([contrib, elev])
        ret.append(score)
    ret.append((False, False))  # reverse the elevation comparison
    return ret

def rev_custom_scoring(pts1, pts2):
    # flip all the flags for reverse sort
    v1, v2, rev = custom_scoring(pts1, pts2)
    rev = [not b for b in rev]  
    return v1, v2, rev

def test_custom_scoring():
    """  Make a database using the custom_score for a compare function and make sure it retains the correct data

        Based on original scores the following matrix was made
       [  0,  10,  20,  30,  40, 100, 110, 120, 130, 140],
       [  1,  11,  21,  31,  41, 101, 111, 121, 131, 141],
       [  2,  12, 999, 999, 999, 102, 112, 122, 132, 142],
       [  3,  13, 999, 999, 999, 103, 113, 123, 133, 143],
       [  4,  14, 999, 999, 999, 104, 114, 124, 134, 144],
       [200, 210, 220, 230, 240, 999, 999, nan, nan, nan],
       [201, 211, 221, 231, 241, 999, 999, nan, nan, nan],
       [202, 212, 222, 232, 242, nan, nan, nan, nan, nan],
       [203, 213, 223, 233, 243, nan, nan, nan, nan, nan],
       [204, 214, 224, 234, 244, nan, nan, nan, nan, nan],

    Now revising the scoring with a custom function we should get the depths below
      0, 10, 20, 30, 40, 50, 60, 70, 80
      1, 11, 21, 31, 41, 51, 61, 71, 81
      2, 12,999,999,999, 52, 62, 72, 82
      3, 13,999,999,999, 53, 63, 73, 83
      4, 14,999,999,999, 54, 64, 74, 84
    200,210,220,230,240,999,999,nan,nan
    201,211,221,231,241,999,999,nan,nan
    202,212,222,232,242,nan,nan,nan,nan
    203,213,223,233,243,nan,nan,nan,nan
    204,214,224,234,244,nan,nan,nan,nan
    nan,nan,nan,nan,nan,nan,nan,nan,nan
    nan,nan,nan,nan,nan,nan,nan,nan,nan
    nan,nan,nan,nan,nan,nan,nan,nan,nan

    """
    # use the SW, SE, NW, mid arrays from test data but write them to disk as a text file and a raster then insert.
    # also change the scoring priority using a custom function so that the result is different than if the original scores were used.

    use_dir = custom_scoring_dir
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)
    x1, y1, x2, y2 = extents = (-.5, -.5, 11.5, 11.5)
    resolution = 1
    epsg = 26918
    db = CustomArea(epsg, *extents, resolution, resolution, use_dir, history_class=RasterHistory)

    # test the insert text function by writing the data out with numpy first
    r, c, z, uncertainty, score, flags = SW_5x5  # flip the r,c order since insert wants x,y
    txt_filename = use_dir.joinpath("SW.txt")
    numpy.savetxt(txt_filename, numpy.array([z.reshape(-1), c.reshape(-1), r.reshape(-1), uncertainty.reshape(-1)]).T)
    # db.insert_survey_array((c, r, z, uncertainty, score, flags), "origin", contrib_id=1)
    db.insert_txt_survey(txt_filename, contrib_id=1, dformat=[('depth', 'f4'), ('x', 'f8'), ('y', 'f8'), ('uncertainty', 'f4')],
                         compare_callback=custom_scoring)

    # this SE with values 100-140 is retained in the original test, this time it should get overwritten due to custom scoring
    r, c, z, uncertainty, score, flags = SE_5x5
    tiles = db.insert_survey_array((c, r, z, uncertainty, score, flags), "east", contrib_id=3, compare_callback=custom_scoring)
    # add the metadata info so we can remove/reinsert this in other tests
    db.finished_survey_insertion("SE orig", tiles, 3)

    # this should overwrite the SE data with values of 50-90 based on the custom_scoring callback making the contrib==2 have a higher score
    r, c, z, uncertainty, score, flags = SW_5x5
    tiles = db.insert_survey_array((c + 5, r, z + 50, uncertainty, score, flags), "east2", contrib_id=2, compare_callback=custom_scoring)
    # add the metadata info so we can remove/reinsert this in other tests
    db.finished_survey_insertion("SE+50", tiles, 2)

    # this NW should be kept over the next NW, which allows to test that it disappears on reinsert since it doesn't have a filename
    r, c, z, uncertainty, score, flags = NW_5x5
    tiles = db.insert_survey_array((c, r, z, uncertainty, score, flags), "east", contrib_id=6, compare_callback=custom_scoring)
    # add the metadata info so we can remove/reinsert this in other tests
    db.finished_survey_insertion("SE orig again", tiles, 6)

    # test an input geotif via the insert_gdal function
    r, c, z, uncertainty, score, flags = NW_5x5
    gdal_filename = use_dir.joinpath("NW.tif")
    driver = gdal.GetDriverByName('GTiff')
    # dataset = driver.CreateDataSource(self.path)
    dataset = driver.Create(str(gdal_filename), xsize=z.shape[1], ysize=z.shape[0], bands=2, eType=gdal.GDT_Float32,
                            options=['COMPRESS=LZW', "TILED=YES"])
    gt = [-0.5, 1, 0, 4.5, 0, 1]  # note this geotransform is positive Y
    dataset.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()
    dataset.SetProjection(dest_wkt)
    for b, val in enumerate([z, uncertainty]):
        band = dataset.GetRasterBand(b + 1)
        band.WriteArray(val)
    del dataset

    db.insert_survey_gdal(gdal_filename, contrib_id=5, compare_callback=custom_scoring)

    # overwrite some of the origin grid but keep score (based on contrib) below NW
    # use a file so the reinsert after removal will work for contributor #4
    r, c, z, uncertainty, score, flags = MID_5x5
    txt_filename = use_dir.joinpath("middle.txt")
    numpy.savetxt(txt_filename, numpy.array([z.reshape(-1), c.reshape(-1), r.reshape(-1), uncertainty.reshape(-1)]).T)
    # db.insert_survey_array((c, r, z, uncertainty, score, flags), "overwrite origin", contrib_id=4, compare_callback=custom_scoring)
    db.insert_txt_survey(txt_filename, contrib_id=4, dformat=[('depth', 'f4'), ('x', 'f8'), ('y', 'f8'), ('uncertainty', 'f4')],
                         compare_callback=custom_scoring)

    # check that the data was processed correctly
    tx, ty = db.db.get_tiles_indices(0, 0, 0, 0)[0]
    tile = db.db.get_tile_history_by_index(tx, ty)
    # db.db.get_tile_history_by_index(0, 0)[-1].get_array(LayersEnum.CONTRIBUTOR)
    raster_data = tile[-1]
    arr = raster_data.get_arrays()
    r0, c0 = raster_data.xy_to_rc(0, 0)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 2, c0:c0 + 2] == SW_5x5[2][:2, :2])  # data from text file
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)  # overwrite of SW data
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 5:r0 + 7, c0 + 5:c0 + 7] == 999)  # the overwrite of the empty space and
    # instead of SE_5x5 this is the +50 array because of the custom scoring function
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 5, c0 + 5:c0 + 10] == SW_5x5[2] + 50)  # data that was only retained based on the custom_scoring
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 5:r0 + 10, c0:c0 + 5] == NW_5x5[2])  # data from tif
    assert numpy.all(arr[LayersEnum.CONTRIBUTOR, r0 + 5:r0 + 10, c0:c0 + 5] == 6)  # data from tif


def test_export_custom_scoring():
    """  The depths, contributors and scores from the prior test
      0, 10, 20, 30, 40, 50, 60, 70, 80, 90
      1, 11, 21, 31, 41, 51, 61, 71, 81, 91
      2, 12,999,999,999, 52, 62, 72, 82, 92
      3, 13,999,999,999, 53, 63, 73, 83, 93
      4, 14,999,999,999, 54, 64, 74, 84, 94
    200,210,220,230,240,999,999,nan,nan,nan
    201,211,221,231,241,999,999,nan,nan,nan
    202,212,222,232,242,nan,nan,nan,nan,nan
    203,213,223,233,243,nan,nan,nan,nan,nan
    204,214,224,234,244,nan,nan,nan,nan,nan

    1,  1,  1,  1,  1,  2,  2,  2,  2,  2
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    6,  6,  6,  6,  6,  4,  4,nan,nan,nan
    6,  6,  6,  6,  6,  4,  4,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan

    100,100,100,100,100,  2,  2,  2,  2,  2
    100,100,100,100,100,  2,  2,  2,  2,  2
    100,100,2.5,2.5,2.5,  2,  2,  2,  2,  2
    100,100,2.5,2.5,2.5,  2,  2,  2,  2,  2
    100,100,2.5,2.5,2.5,  2,  2,  2,  2,  2
    100,100,100,100,100,2.5,2.5,nan,nan,nan
    100,100,100,100,100,2.5,2.5,nan,nan,nan
    100,100,100,100,100,nan,nan,nan,nan,nan
    100,100,100,100,100,nan,nan,nan,nan,nan
    100,100,100,100,100,nan,nan,nan,nan,nan

    Now we will export the data at a lower res and test the combine logic.
    a 3m export on the original data and score (highest score then largest elevation) should get
     21  41  82  92
    220 240 999  94
    223 243 999 nan
    224 244 nan nan

    using the custom compare function should yield (based on highest contributor number and highest elevation)
      - note we made the +50 (contrib=2) get a custom score higher than other contributors in the custom_score function
    999  52  82  92
    220  54  84  94
    223 243 999 nan
    224 244 nan nan

    reversing the using the custom compare function should yield (based on smallest contributor number and lowest elevation)
      0  30  60  90
      3 999 999  93
    201 999 999 nan
    204 234 nan nan
    """
    use_dir = custom_scoring_dir
    # create the database if not existing
    if not os.path.exists(use_dir):
        test_custom_scoring()
    db = WorldDatabase.open(use_dir)

    # export at the original resolution and make sure the original data is retained in the right positions.
    export_fname = use_dir.joinpath("export.tif")
    db.export(export_fname)
    ds1 = gdal.Open(str(export_fname))
    # geotransform will have negative Y which make the indices in opposite order as the ones we passed in originally
    assert numpy.all(ds1.GetGeoTransform() == (-0.5, 1.0, 0.0, 11.5, 0.0, -1.0))
    arr = ds1.ReadAsArray()
    assert numpy.all(arr[LayersEnum.ELEVATION, -1:-3:-1, 0:2:] == SW_5x5[2][:2, :2])  # last two rows, first two columns
    assert numpy.all(arr[LayersEnum.ELEVATION, -3:-5:-1, 2:5] == 999)
    assert numpy.all(arr[LayersEnum.ELEVATION, -6:-8:-1, 5:7] == 999)
    # instead of SE_5x5 this is the +50 array because of the custom scoring function
    assert numpy.all(arr[0, -1:-6:-1, 5:10] == SW_5x5[2] + 50)
    assert numpy.all(arr[0, -6:-11:-1, 0:5] == NW_5x5[2])

    # test the export of a lower resolution so export has to combine data
    export_fname3 = use_dir.joinpath("export3.tif")
    db.export_area(export_fname3, -0.5, 11.5, 11.5, -0.5, (3, 3))
    ds3 = gdal.Open(str(export_fname3))
    arr3 = ds3.ReadAsArray()
    numpy.testing.assert_equal(arr3[0][-1], (21, 41, 82, 92))
    numpy.testing.assert_equal(arr3[0][-2], (220, 240, 999, 94))
    numpy.testing.assert_equal(arr3[0][-3], (223, 243, 999, numpy.nan))
    numpy.testing.assert_equal(arr3[0][-4], (224, 244, numpy.nan, numpy.nan))

    # test a custom scoring method (use contributor number instead of score)
    export_fname3cc = use_dir.joinpath("export3cc.tif")
    db.export_area(export_fname3cc, -0.5, 11.5, 11.5, -0.5, (3, 3), compare_callback=custom_scoring)
    ds3cc = gdal.Open(str(export_fname3cc))
    arr3cc = ds3cc.ReadAsArray()
    numpy.testing.assert_equal(arr3cc[0][-1], (999, 52, 82, 92))
    numpy.testing.assert_equal(arr3cc[0][-2], (220, 54, 84, 94))
    numpy.testing.assert_equal(arr3cc[0][-3], (223, 243, 999, numpy.nan))
    numpy.testing.assert_equal(arr3cc[0][-4], (224, 244, numpy.nan, numpy.nan))

    # test the reverse flags of the custom scoring
    export_fname3ccrev = use_dir.joinpath("export3cc.tif")
    db.export_area(export_fname3ccrev, -0.5, 11.5, 11.5, -0.5, (3, 3), compare_callback=rev_custom_scoring)
    ds3ccrev = gdal.Open(str(export_fname3cc))
    arr3ccrev = ds3ccrev.ReadAsArray()
    numpy.testing.assert_equal(arr3ccrev[0][-1], (0, 30, 60, 90))
    numpy.testing.assert_equal(arr3ccrev[0][-2], (3, 999, 999, 93))
    numpy.testing.assert_equal(arr3ccrev[0][-3], (201, 999, 999, numpy.nan))
    numpy.testing.assert_equal(arr3ccrev[0][-4], (204, 234, numpy.nan, numpy.nan))

def test_remove_survey():
    """  The depths, contributors and scores from the prior test
      0, 10, 20, 30, 40, 50, 60, 70, 80, 90
      1, 11, 21, 31, 41, 51, 61, 71, 81, 91
      2, 12,999,999,999, 52, 62, 72, 82, 92
      3, 13,999,999,999, 53, 63, 73, 83, 93
      4, 14,999,999,999, 54, 64, 74, 84, 94
    200,210,220,230,240,999,999,nan,nan,nan
    201,211,221,231,241,999,999,nan,nan,nan
    202,212,222,232,242,nan,nan,nan,nan,nan
    203,213,223,233,243,nan,nan,nan,nan,nan
    204,214,224,234,244,nan,nan,nan,nan,nan

    1,  1,  1,  1,  1,  2,  2,  2,  2,  2
    1,  1,  1,  1,  1,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    1,  1,  4,  4,  4,  2,  2,  2,  2,  2
    6,  6,  6,  6,  6,  4,  4,nan,nan,nan
    6,  6,  6,  6,  6,  4,  4,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan
    6,  6,  6,  6,  6,nan,nan,nan,nan,nan
    
    Then we remove contributor #2 
    which causes a remove and re-insert of contributors 4,5,6
    but #6 was a memory insert so gets passed over so only 4,5 are restored
    leaving the following values: 
        1   [  0,  10,  20,  30,  40, 100, 110, 120, 130, 140],
        1   [  1,  11,  21,  31,  41, 101, 111, 121, 131, 141],
        1   [  2,  12, 999, 999, 999, 102, 112, 122, 132, 142],
        1   [  3,  13, 999, 999, 999, 103, 113, 123, 133, 143],
        2   [  4,  14, 999, 999, 999, 104, 114, 124, 134, 144],

      0, 10, 20, 30, 40,100,110,120,130,140
      1, 11, 21, 31, 41,101,111,121,131,141
      2, 12,999,999,999,999,999,122,132,142
      3, 13,999,999,999,999,999,123,133,143
      4, 14,999,999,999,999,999,124,134,144
    200,210,220,230,240,999,999,nan,nan,nan
    201,211,221,231,241,999,999,nan,nan,nan
    202,212,222,232,242,nan,nan,nan,nan,nan
    203,213,223,233,243,nan,nan,nan,nan,nan
    204,214,224,234,244,nan,nan,nan,nan,nan

    1,  1,  1,  1,  1,  3,  3,  3,  3,  3
    1,  1,  1,  1,  1,  3,  3,  3,  3,  3
    1,  1,  4,  4,  4,  4,  4,  3,  3,  3
    1,  1,  4,  4,  4,  4,  4,  3,  3,  3
    1,  1,  4,  4,  4,  4,  4,  3,  3,  3
    5,  5,  5,  5,  5,  4,  4,nan,nan,nan
    5,  5,  5,  5,  5,  4,  4,nan,nan,nan
    5,  5,  5,  5,  5,nan,nan,nan,nan,nan
    5,  5,  5,  5,  5,nan,nan,nan,nan,nan
    5,  5,  5,  5,  5,nan,nan,nan,nan,nan

    """
    use_dir = custom_scoring_dir
    # create the database as we modify it in this function, running this test twice in a row would fail otherwise.
    test_custom_scoring()
    db = WorldDatabase.open(use_dir)
    # Removing contributor #2 should cause contributors 2, 4, 5 to be removed
    # (note that contributor insertion order was 1,3,2,6,4,5 to make sure 2 overwrote 3)
    # The recompute should then try to re-insert contributors 4, 5, 6 but only #4+5 came from a file; reinsert of #6 should then fail gracefully
    db.remove_and_recompute(2, compare_callback=custom_scoring)

    # check that the data was processed correctly
    tx, ty = db.db.get_tiles_indices(0, 0, 0, 0)[0]
    tile = db.db.get_tile_history_by_index(tx, ty)
    raster_data = tile[-1]
    arr = raster_data.get_arrays()
    r0, c0 = raster_data.xy_to_rc(0, 0)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 2, c0:c0 + 2] == SW_5x5[2][:2, :2])  # data from text file
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)  # overwrite of SW data
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 5:r0 + 7, c0 + 5:c0 + 7] == 999)  # the overwrite of the empty space and
    # no +50 anymore as that is the contributor that was removed
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 5, c0 + 7:c0 + 10] == SE_5x5[2][:, 2:])  # the original contrib=3 data
    assert numpy.all(arr[LayersEnum.CONTRIBUTOR, r0:r0 + 5, c0 + 7:c0 + 10] == 3)  # the original contrib=3 data
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 5:r0 + 10, c0:c0 + 5] == NW_5x5[2])  # data from tif
    # data from tif replaced the memory array (contrib=6) that was put in
    assert numpy.all(arr[LayersEnum.CONTRIBUTOR, r0 + 5:r0 + 10, c0:c0 + 5] == 5)


def test_fast_export():
    use_dir = data_dir.joinpath('fast_export_db')
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)
    resolution = 1
    epsg = 26918
    db = WorldDatabase(
        UTMTileBackendExactRes(resolution, resolution, epsg, RasterHistory, DiskHistory, TiffStorage, use_dir))

    # test the insert text function by writing the data out with numpy first
    r, c, z, uncertainty, score, flags = SW_5x5  # flip the r,c order since insert wants x,y
    db.insert_survey_array((c, r, z, uncertainty, score, flags), "origin", contrib_id=1)

    # find the tile that was filled then move a copy of the data to adjacent tiles
    tx, ty = db.db.tile_scheme.xy_to_tile_index(c[0,0], r[0,0])
    x1, y1, x2, y2 = db.db.tile_scheme.tile_index_to_xy(tx, ty)

    # add a copy of the data to the tile to the right and above the original data.  Add 20 to the the right and fifty above
    db.insert_survey_array((c+(x2-x1), r, z+20, uncertainty, score, flags), "west", contrib_id=3)
    db.insert_survey_array((c, r+(y2-y1), z + 50, uncertainty, score, flags), "north", contrib_id=2)

    # extract the entire area that we just added data to, the original tile plus adjacent
    # x2, y2 are the edge of the first tile which will actually return the second tile when queried for where that position would fall
    # but we'll add one cell resolution to make sure we are in the north and east tiles
    arr, txs, tys, indices = db.fast_extract(x1, y1, x2+resolution, y2+resolution)

    assert arr


