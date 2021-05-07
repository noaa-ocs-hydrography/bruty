import json

from nbs.bruty.tile_calculations import TilingScheme, ExactTilingScheme, ExactUTMTiles, UTMTiles, LatLonTiles
from test_data import master_data, make_clean_dir, data_dir, SW_5x5, NW_5x5, SE_5x5, MID_5x5


def test_json():
    # dir = make_clean_dir("TilingScheme")
    ts = TilingScheme(1, 2, 3, 4, 5, 26919)
    orig_json_dict = ts.for_json()
    json_str = json.dumps(orig_json_dict)
    json_dict = json.loads(json_str)
    ts2 = TilingScheme.create_from_json(json_dict)
    assert ts2.__class__.__name__ == 'TilingScheme'
    assert ts2.min_x == ts.min_x
    assert ts2.min_y == ts.min_y
    assert ts2.max_x == ts.max_x
    assert ts2.max_y == ts.max_y
    assert ts2.zoom == ts.zoom
    assert ts2.epsg == ts.epsg

def test_UTM():
    # dir = make_clean_dir("TilingScheme")
    ts = UTMTiles(5, 26919)
    orig_json_dict = ts.for_json()
    json_str = json.dumps(orig_json_dict)
    json_dict = json.loads(json_str)
    ts2 = TilingScheme.create_from_json(json_dict)
    assert ts2.__class__.__name__ == 'UTMTiles'
    assert ts2.min_x == ts.min_x
    assert ts2.max_y == ts.max_y
    assert ts2.zoom == ts.zoom
    assert ts2.epsg == ts.epsg

def test_exact_scheme():
    ts = ExactTilingScheme(5, 10, 0, 2, 100, 200, 1, 26919)
    orig_json_dict = ts.for_json()
    json_str = json.dumps(orig_json_dict)
    json_dict = json.loads(json_str)
    ts2 = TilingScheme.create_from_json(json_dict)
    assert ts2.__class__.__name__ == 'ExactTilingScheme'
    assert ts2.min_x == ts.min_x
    assert ts2.max_y == ts.max_y
    assert ts2.zoom == ts.zoom
    assert ts2.epsg == ts.epsg
    assert all(ts2.edges_x == ts.edges_x)
    assert all(ts2.edges_y == ts.edges_y)
    assert ts2.res_x == ts.res_x
    assert ts2.res_y == ts.res_y

