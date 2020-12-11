import pathlib
import os
import shutil

import pytest
import numpy

from xipe_dev.xipe2.history import DiskHistory, MemoryHistory, RasterHistory
from xipe_dev.xipe2.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from xipe_dev.xipe2.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, TMSMercatorTileBackend
from xipe_dev.xipe2.world_raster_database import WorldDatabase, onerr, get_geotransform

from xipe_dev.xipe2.test_data import master_data, data_dir

nan = numpy.nan
os.makedirs(data_dir, exist_ok=True)

norfolk_spherical = (-8494906.0, 4419895.0)
norfolk_utm_18 = (383135.613, 4080370.689)
norfolk_ll = (-76.311039, 36.862043)


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
    tms_origin_idx =  ((2**history_db.tile_scheme.zoom)/2, (2**history_db.tile_scheme.zoom)/2)
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
    use_dir = data_dir.joinpath('tile4_utm_db')
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619

    if False:
        w, n = 288757.22, 4561186.99
        e, s = 297025.15, 4547858.16
        db.insert_survey_array(numpy.array(([w, e], [n, s], [45, 33], [2, 2], [10, 10], [0, 0])), r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_2m_MLLW_2of3.bag_corners")
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
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H12700_MB_1m_MLLW_1of3.bag",]:
        print('processsing grid', bag_file)
        db.insert_survey_gdal(bag_file)

    georef_transformer = get_geotransform(4326, 26919)
    for txt_file in [r"C:\Data\nbs\PBC19_Tile4_surveys\D00111.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H06443.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H08615.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\F00363.csar.du.txt",
                     r"C:\Data\nbs\PBC19_Tile4_surveys\H06442.csar.du.txt",]:
        print('processsing txt', txt_file)
        db.insert_txt_survey(txt_file, transformer=georef_transformer)

# importlib.reload(tile_calculations)
# g = tile_calculations.GlobalGeodetic(zoom=2); f=g.xy_to_tile; print(f(-100, 40)); print(f(100, -40)); print(f(182, -91))
# g.tile_to_xy(0,0, 2)
# tx, ty = g.xy_to_tile_index(45, 45); print(tx,ty); print(g.tile_index_to_xy(tx, ty))
# from xipe_dev.xipe2.tile_calculations import TMSTilesMercator, GoogleTilesMercator
# merc = TMSTilesMercator(13)
# tx, ty = merc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(merc.tile_index_to_xy(tx, ty))
# gmerc = GoogleTilesMercator(13)
# tx, ty = gmerc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(gmerc.tile_index_to_xy(tx, ty))