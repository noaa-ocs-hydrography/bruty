import pathlib
import os

import pytest
import numpy

from nbs.bruty.history import DiskHistory, MemoryHistory, RasterHistory
from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
from test_data import master_data, data_dir


@pytest.fixture(scope="module", params=list(master_data.values()), ids=list(master_data.keys()))
def data_lists(request):
    yield request.param


@pytest.fixture(scope="module")
def data_arrays(data_lists):
    npd = [numpy.array(data) for data in data_lists]
    for i, survey in enumerate(npd):
        while survey.ndim < 3:
            survey = numpy.expand_dims(survey, axis=-1)
            npd[i] = survey
    return npd


@pytest.fixture(scope="module")
def data_rasters(data_arrays):
    return [RasterData.from_arrays(data) for data in data_arrays]


@pytest.fixture(scope="module", params=[MemoryHistory(MemoryStorage), DiskHistory(TiffStorage, data_dir.joinpath("history"))],
                ids=['memHist', 'tiffHist'])
def raster_history(request):
    raster = RasterHistory(request.param)
    raster.clear()  # delete files if they exist
    return raster


@pytest.mark.parametrize("raster", [RasterData(MemoryStorage()), RasterData(TiffStorage(data_dir.joinpath("layer_order.tif")))], ids=["mem", "tiff"])
def test_layer_ordering(raster):
    elev = numpy.array([[0, 0], [0, 0]])
    uncert = numpy.array([[1, 1], [1, 1]])
    contr = numpy.array([[2, 2], [2, 2]])
    score = numpy.array([[3, 3], [3, 3]])
    flags = numpy.array([[4, 4], [4, 4]])
    mask = numpy.array([[0, 0], [0, 0]])

    data = numpy.array((elev, uncert, contr, score, flags, mask))

    raster.set_arrays(data[:5])
    assert numpy.all(raster.get_array(LayersEnum.ELEVATION) == raster.get_array("ELEVATION"))
    assert numpy.all(raster.get_array(LayersEnum.ELEVATION) == elev)
    assert numpy.all(raster.get_array(LayersEnum.UNCERTAINTY) == uncert)
    assert numpy.all(raster.get_array(LayersEnum.CONTRIBUTOR) == contr)
    assert numpy.all(raster.get_array(LayersEnum.SCORE) == score)

    raster.set_arrays(data[:4], [3, 2, 1, 0])  # pass them in backwards and make sure they sort correctly
    assert numpy.all(raster.get_array(LayersEnum.ELEVATION) == score)
    assert numpy.all(raster.get_array(LayersEnum.UNCERTAINTY) == contr)
    assert numpy.all(raster.get_array(LayersEnum.CONTRIBUTOR) == uncert)
    assert numpy.all(raster.get_array(LayersEnum.SCORE) == elev)

    raster.set_arrays(data[:4], ["UNCERTAINTY", LayersEnum.ELEVATION, 3, "CONTRIBUTOR"])  # pass them in using strings, numbers and enum values
    assert numpy.all(raster.get_array(LayersEnum.ELEVATION) == uncert)
    assert numpy.all(raster.get_array(LayersEnum.UNCERTAINTY) == elev)
    assert numpy.all(raster.get_array(LayersEnum.CONTRIBUTOR) == score)
    assert numpy.all(raster.get_array(LayersEnum.SCORE) == contr)


def test_fill(raster_history, data_rasters):
    for data in data_rasters:
        raster_history.append(data)
    for i, data in enumerate(data_rasters):
        assert numpy.all(arrays_match(raster_history[i].get_arrays(), data.get_arrays()))
        # assert numpy.all(raster_history[i].get_arrays() == data.get_arrays())
        assert raster_history[i].get_metadata() == data.get_metadata()


