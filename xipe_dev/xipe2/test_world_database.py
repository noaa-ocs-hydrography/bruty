import pathlib
import os

import pytest
import numpy

from xipe_dev.xipe2.history import DiskHistory, MemoryHistory, RasterHistory
from xipe_dev.xipe2.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum
from xipe_dev.xipe2.world_raster_database import LatLonBackend, GoogleLatLonTileBackend, UTMTileBackend, GoogleMercatorTileBackend, TMSMercatorTileBackend

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

arr1 = numpy.zeros((4, 3, 5))
arr2 = arr1 + 1
arr3 = arr2 + 1
arr1[:,0] = numpy.nan
arr2[:,1] = numpy.nan
r0 = RasterData.from_arrays(arr1)
r1 = RasterData.from_arrays(arr2)
r2 = RasterData.from_arrays(arr3)

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
    x2 = x1 + history_db.tile_scheme.width() / history_db.tile_scheme.num_tiles()
    y2 = y1 + history_db.tile_scheme.height() / history_db.tile_scheme.num_tiles()
    indices = history_db.get_tiles_indices(x1, y1, x2, y2)
    history = history_db.get_tile_history(x1, y1)
    if issubclass(history_db.storage_class, DiskHistory):
        history.clear()
    history.append(r0)
    history.append(r1)
    history.append(r2)

    # Try a reload from disk if applicable
    if issubclass(history_db.storage_class, DiskHistory):
        del history
        history = history_db.get_tile_history(x1, y1)
    assert numpy.all(history[0].get_arrays() == r0)
    assert numpy.all(history[2].get_arrays() == r2)



# importlib.reload(tile_calculations)
# g = tile_calculations.GlobalGeodetic(zoom=2); f=g.xy_to_tile; print(f(-100, 40)); print(f(100, -40)); print(f(182, -91))
# g.tile_to_xy(0,0, 2)
# tx, ty = g.xy_to_tile_index(45, 45); print(tx,ty); print(g.tile_index_to_xy(tx, ty))
# from xipe_dev.xipe2.tile_calculations import TMSTilesMercator, GoogleTilesMercator
# merc = TMSTilesMercator(13)
# tx, ty = merc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(merc.tile_index_to_xy(tx, ty))
# gmerc = GoogleTilesMercator(13)
# tx, ty = gmerc.xy_to_tile_index(-8494906.0, 4419895.0); print(tx,ty); print(gmerc.tile_index_to_xy(tx, ty))