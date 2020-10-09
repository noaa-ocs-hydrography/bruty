import os
import enum
import numpy
from collections.abc import Sequence

from osgeo import gdal

from xipe_dev.xipe2.abstract import VABC, abstractmethod

# @todo - investigate the two choices: store full data arrays which should be faster and deltas could be created vs storing deltas which may be smaller but slower
# @todo - also deltas need a mask (could use a mask or contributor value of -1) so we can tell the difference between no-change and was empty  (both are a nan right now that doesn't work)
LayersEnum = enum.IntEnum('Layers', (("ELEVATION", 0), ("UNCERTAINTY", 1), ("CONTRIBUTOR", 2), ("SCORE", 3)))  # , ("MASK", 4)


class Storage(VABC):
    @staticmethod
    def _layers_as_ints(layers):
        int_layers = []
        if layers is None:
            int_layers = [int(lyr) for lyr in LayersEnum]
        elif isinstance(layers, int):
            int_layers = [layers]
        elif isinstance(layers, str):
            int_layers = [LayersEnum[layers]]
        elif isinstance(layers, Sequence):
            for lyr in layers:
                if isinstance(lyr, str):
                    int_layers.append(LayersEnum[lyr])
                else:
                    int_layers.append(int(lyr))
        else:
            raise TypeError(f"layers {layers} not understood for accessing raster data")
        return int_layers

    @abstractmethod
    def get_arrays(self, layers=None):
        raise NotImplementedError()
    @abstractmethod
    def get_metadata(self):
        raise NotImplementedError()
    @abstractmethod
    def set_arrays(self, layers=None):
        raise NotImplementedError()
    @abstractmethod
    def set_metadata(self):
        raise NotImplementedError()


class BagStorage(Storage):
    extension = ".bag"
    pass
class TiffStorage(Storage):
    """This might be usable as any gdal raster by making the driver/extension a parameter"""
    extension = ".tif"
    def __init__(self, path):
        self.path = path
        self._version = 1
    def get_arrays(self, layers=None):
        layer_nums = self._layers_as_ints(layers)
        # @todo check the version number and update for additional layers if needed
        dataset = gdal.Open(str(self.path))  # str in case it's a pathlib.Path
        array_list = []
        for lyr in layer_nums:
            band = dataset.GetRasterBand(lyr + 1)
            array_list.append(band.ReadAsArray())
            del band
        del dataset
        return numpy.array(array_list)
    def set(self, data):
        self.set_arrays(data.get_arrays())
        self.set_metadata(data.get_metadata())
    def get_metadata(self):
        # @todo implement metadata
        print("metadata not supported yet")
        return {}
        # raise NotImplementedError()
    def set_arrays(self, arrays, layers=None):
        layer_nums = self._layers_as_ints(layers)
        if os.path.exists(self.path):
            # note: this must be `gdal.OpenEx()` or it will raise an error. Why? I dunno.
            dataset = gdal.Open(str(self.path), gdal.GA_Update)
        else:
            driver = gdal.GetDriverByName('GTiff')
            # dataset = driver.CreateDataSource(self.path)
            dataset = driver.Create(str(self.path), xsize=arrays.shape[2], ysize=arrays.shape[1], bands=len(LayersEnum), eType=gdal.GDT_Float32)
            for band_index in LayersEnum:
                band = dataset.GetRasterBand(band_index + 1)
                if band_index == 0:  # tiff only supports one value of nodata, supress the warning
                    band.SetNoDataValue(float(numpy.nan))
                band.SetDescription(LayersEnum(band_index).name)
                del band
            # dataset.SetProjection(crs.wkt)
            # dataset.SetGeoTransform(transform.to_gdal())

        for index, lyr in enumerate(layer_nums):
            band = dataset.GetRasterBand(lyr + 1)
            band.WriteArray(arrays[index])
            del band
        del dataset

    def set_metadata(self, metadata):
        print("set is not NotImplemented")
        # raise NotImplementedError()

class DatabaseStorage(Storage):
    pass
class MemoryStorage(Storage):
    extension = ""
    def __init__(self):  # , shape):
        self._version = 1
        self.metadata = {}
        self.arrays = None  # numpy.full([self._layers_as_ints(None), *shape], numpy.nan)
    def get_arrays(self, layers=None):
        return self.arrays[self._layers_as_ints(layers)].copy()
    def get_metadata(self):
        return self.metadata.copy()
    def set_arrays(self, arrays, layers=None):
        if self.arrays is None:
            self.arrays = numpy.full(arrays.shape, numpy.nan)
        layer_nums = self._layers_as_ints(layers)
        # this should in theory reorder the arrays to match the LayersEnum ordering
        self.arrays[layer_nums] = arrays
    def set_metadata(self, metadata):
        self.metadata = metadata.copy()

class XarrayStorage(Storage):
    pass

class RasterData(VABC):
    def __init__(self, storage):
        self.storage = storage
        self._version = 1
    def __repr__(self):
        return str(self.get_arrays())

    @staticmethod
    def from_arrays(arrays, metadata={}, layers=None):
        r = RasterData(MemoryStorage())
        r.set_arrays(arrays, layers)
        r.set_metadata(metadata)
        return r
    def get_metadata(self):
        return self.storage.get_metadata()
    def set_metadata(self, metadata):
        self.storage.set_metadata(metadata)
    def get_array(self, layer):
        return self.storage.get_arrays(layer)[0]
    def get_arrays(self, layers=None):
        return self.storage.get_arrays(layers)
    def get_array_at_res(self, res, layers=None):
        """ Return the array at a different resolution than intrinsically stored.
        This could be to change the density of the tile (higher res data being added) or
        to create visualizations that could need higher or lower numbers of pixels.

        Parameters
        ----------
        res
        layers
            list of layer names desired.  If None then get all available,

        Returns
        -------

        """
        pass
    def set_array(self, array, layer):
        self.storage.set_arrays(array, layer)
    def set_arrays(self, arrays, layers=None):
        self.storage.set_arrays(arrays, layers)
    def get_elevation(self):
        return self.storage.get_array(LayersEnum.ELEVATION)
    def get_flags(self):  # include index to survey score parameters (year?)
        pass
    def computed_score(self):
        # @todo look up the score parameters in the database table and then compute a score based on that
        return self.get_array(LayersEnum.SCORE)
    def apply_delta(self, delta):
        current_data = self.get_arrays()
        d = delta.get_arrays()
        indices = ~numpy.isnan(d)
        current_data[indices] = d[indices]
        r = RasterDelta(MemoryStorage())
        r.set_arrays(current_data)
        return r


class RasterDelta(RasterData):
    def __init__(self, storage):
        super().__init__(storage)
        self._ver = 1

    @staticmethod
    def from_rasters(raster_old, raster_new):
        new_data = raster_new.get_arrays()
        old_data = raster_old.get_arrays()
        diff_indices = new_data != old_data
        # numpy.logical_or.reduce(i)
        indices = numpy.any(diff_indices, axis=0)  # if any of the layers had a change then save them all the layers at that location, not strictly necessary
        delta_array = numpy.full(new_data.shape, numpy.nan)
        delta_array[:, indices] = old_data[:, indices]
        r = RasterDelta(MemoryStorage())
        r.set_arrays(delta_array)
        return r

