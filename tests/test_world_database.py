import pathlib
import os
import shutil
from functools import partial

import pytest
import numpy
from osgeo import gdal

from bruty.history import DiskHistory, MemoryHistory, RasterHistory
from bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from bruty.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, \
    TMSMercatorTileBackend, merge_arrays
from bruty.world_raster_database import WorldDatabase, onerr, get_geotransformer

from tests.test_data import master_data, data_dir, SW_5x5, NW_5x5, SE_5x5, MID_5x5

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


def test_make_db():
    use_dir = simple_utm_dir
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # override the init_tile with a function to make approx 1m square resolution
    db.init_tile = partial(make_1m_tile, db)
    # NOTE: when using this coordinate tile the 0,0 position ends up in a cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the cells will end up in the -1,-1 index on the export

    db.insert_survey_array(SW_5x5, "origin")
    db.insert_survey_array(SE_5x5, "east")
    # this +50 array should not show up as the score is the same but depth is deeper
    r, c, z, uncertainty, score, flags = SE_5x5
    db.insert_survey_array((c + 5, r, z + 50, uncertainty, score, flags), "east")
    db.insert_survey_array(NW_5x5, "north east")
    # overwrite some of the origin grid but keep score below east and northeast
    db.insert_survey_array(MID_5x5, "overwrite origin")
    tx, ty = db.db.get_tiles_indices(0, 0, 0, 0)[0]
    tile = db.db.get_tile_history_by_index(tx, ty)
    raster_data = tile[-1]
    arr = raster_data.get_arrays()
    r0, c0 = raster_data.xy_to_rc(0, 0)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0:r0 + 2, c0:c0 + 2] == z[:2, :2])
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)
    assert numpy.all(arr[LayersEnum.ELEVATION, r0 + 2:r0 + 5, c0 + 2:c0 + 5] == 999)
    assert numpy.all(arr[0, r0:r0 + 5, c0 + 5:c0 + 10] == z + 100)
    assert numpy.all(arr[0, r0 + 5:r0 + 10, c0:c0 + 5] == z + 200)


def test_export_area():
    use_dir = simple_utm_dir
    # create the database if not existing
    if not os.path.exists(use_dir):
        test_make_db()

    # NOTE: when using this coordinate tile the 0,0 position ends up in a cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the cells will end up in the -1,-1 index on the export
    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # override the init_tile with a function to make approx 1m square resolution
    db.init_tile = partial(make_1m_tile, db)

    # NOTE: when using this coordinate tile the 0,0 x,y position ends up in a database cell that spans from -0.66 < x < 0.34 and -.72 < y < 0.27
    # so the exported 0,0 will end up in the -1,-1 cell on the export
    db.export_area_old(use_dir.joinpath("old.tif"), -1, -1, 11, 11, (.5, .5))
    db.export_area(use_dir.joinpath("new0.5.tif"), -1, -1, 11, 11, (.5, .5))
    db.export_area(use_dir.joinpath("new1.tif"), -1, -1, 11, 11, (1, 1))
    ds = gdal.Open(str(use_dir.joinpath("new1.tif")))
    b = ds.GetRasterBand(1)
    arr = b.ReadAsArray()
    r, c, z, uncertainty, score, flags = simple_5x5
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
    db.export_area(use_dir.joinpath("new4.tif"), -1, -1, 11, 11, (4, 4))
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


def test_pbc19_tile_4():
    use_dir = data_dir.joinpath('tile4_metadata_utm_db')
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619

    if False:
        w, n = 288757.22, 4561186.99
        e, s = 297025.15, 4547858.16
        db.insert_survey_array(numpy.array(([w, e], [n, s], [45, 33], [2, 2], [10, 10], [0, 0])),
                               r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_2m_MLLW_2of3.bag_corners")
        # # should be NW in tile 3519,4141 and SE in 3541,4131 for UTM 19
        raise Exception("stopping")
    for bag_file in [r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_2m_MLLW_2of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_4m_MLLW_3of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\RI_16_BHR_20190417_BD_2019_023_FULL_4m_interp.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\RI_17_GSP_20190418_BD_2019_022_FULL_4m_interp.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_1of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_2of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12009_MB_2m_MLLW_3of3.bag",
                     # r"C:\Data\nbs\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_MB_2m_MLLW_2of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_MB_50cm_MLLW_1of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12023_VB_4m_MLLW_3of3.bag",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_1m_MLLW_1of3.bag", ]:
        print('processsing grid', bag_file)
        db.insert_survey_gdal(bag_file)

    georef_transformer = get_geotransformer(4326, 26919)
    for txt_file in [r"C:\Data\nbs\PBC19_Tile4_surveys\D00111.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H06443.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H08615.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\F00363.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H06442.csar.du.txt", ]:
        print('processsing txt', txt_file)
        db.insert_txt_survey(txt_file, transformer=georef_transformer)


def test_pbc19_vr():
    use_dir = data_dir.joinpath('tile4_VR_utm_db')
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    db.insert_survey_vr(r"C:\Data\nbs\PBC19_Tile4_surveys\H12010_MB_VR_MLLW.bag")
    print("processed_vr")


def test_export_area_full_db():
    use_dir = data_dir.joinpath('tile4_vr_utm_db')
    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    db.export_area_old(use_dir.joinpath("export_tile_old.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)
    db.export_area(use_dir.joinpath("export_tile_new.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)
