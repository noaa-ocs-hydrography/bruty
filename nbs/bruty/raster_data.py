import os
import enum
import numpy
from collections.abc import Sequence

from osgeo import gdal, osr

from nbs.bruty.abstract import VABC, abstractmethod
from nbs.bruty.utils import affine, affine_center, inv_affine

# @todo - investigate the two choices: store full data arrays which should be faster and deltas could be created vs storing deltas which may be smaller but slower
# @todo - also deltas need a mask (could use a mask or contributor value of -1) so we can tell the difference between no-change and was empty  (both are a nan right now that doesn't work)
LayersEnum = enum.IntEnum('Layers', (("ELEVATION", 0), ("UNCERTAINTY", 1), ("CONTRIBUTOR", 2), ("SCORE", 3), ("FLAGS", 4), ("MASK", 5)))
ALL_LAYERS = tuple(range(LayersEnum.MASK+1))
INFO_LAYERS = tuple(range(LayersEnum.MASK))

class Storage(VABC):
    @staticmethod
    def _layers_as_ints(layers):
        int_layers = []
        if layers is None:
            int_layers = INFO_LAYERS
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
    def __init__(self, path, arrays=None, layers=None):
        self.path = path
        self._version = 1
        if arrays is not None:
            self.set_arrays(arrays, layers)
        self.metadata = {}
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
        self.set_metadata(data.get_metadata())  # @todo set metadata first, it gets used in set_arrays for writing tiff, maybe should move this to meta!
        self.set_arrays(data.get_arrays(ALL_LAYERS), ALL_LAYERS)
    def get_metadata(self):
        # @todo implement metadata
        print("metadata not supported yet")
        return self.metadata.copy()
        # raise NotImplementedError()
    def set_arrays(self, arrays, layers=None):
        layer_nums = self._layers_as_ints(layers)
        if os.path.exists(self.path):
            # note: this must be `gdal.OpenEx()` or it will raise an error. Why? I dunno.
            dataset = gdal.Open(str(self.path), gdal.GA_Update)
        else:
            driver = gdal.GetDriverByName('GTiff')
            # dataset = driver.CreateDataSource(self.path)
            dataset = driver.Create(str(self.path), xsize=arrays.shape[2], ysize=arrays.shape[1], bands=len(LayersEnum), eType=gdal.GDT_Float32,
                                    options=['COMPRESS=LZW'])
            meta = self.get_metadata()
            try:
                min_x = meta['min_x']
                min_y = meta['min_y']
                max_x = meta['max_x']
                max_y = meta['max_y']
                dx = (max_x - min_x) / arrays.shape[2]
                dy = (max_y - min_y) / arrays.shape[1]
                epsg = meta['epsg']
                gt = [min_x, dx, 0, min_y, 0, dy]

                # Set location
                dataset.SetGeoTransform(gt)

                # Get raster projection
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(epsg)
                dest_wkt = srs.ExportToWkt()

                # Set projection
                dataset.SetProjection(dest_wkt)
            except KeyError:
                pass  # doesn't have full georeferencing

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
        # @todo save metadata to a json file or something
        print("set is not NotImplemented (stored to disk)")
        self.metadata = metadata

class DatabaseStorage(Storage):
    pass
class MemoryStorage(Storage):
    extension = ""
    def __init__(self, arrays=None, layers=None):  # , shape):
        self._version = 1
        self.metadata = {}
        self.arrays = None  # numpy.full([self._layers_as_ints(None), *shape], numpy.nan)
        if arrays is not None:
            self.set_arrays(arrays, layers)
    def get_arrays(self, layers=None):
        return self.arrays[self._layers_as_ints(layers), :].copy()
    def get_metadata(self):
        return self.metadata.copy()
    def set_arrays(self, arrays, layers=None):
        if self.arrays is None:
            # make sure the data is allocated for all layers even if not supplied at this time (MASK layer in particular)
            self.arrays = numpy.full([len(LayersEnum)] + list(arrays.shape[1:]), numpy.nan)
        layer_nums = self._layers_as_ints(layers)
        # this should in theory reorder the arrays to match the LayersEnum ordering
        self.arrays[layer_nums, :] = arrays
    def set_metadata(self, metadata):
        self.metadata = metadata.copy()

class XarrayStorage(Storage):
    pass

class RasterData(VABC):
    def __init__(self, storage, arrays=None, layers=None):
        self.storage = storage
        self._version = 1
        if arrays is not None:
            self.set_arrays(arrays, layers)
    def __repr__(self):
        return str(self.get_arrays())

    @staticmethod
    def from_arrays(arrays, metadata={}, layers=None):
        r = RasterData(MemoryStorage())
        r.set_arrays(arrays, layers)
        r.set_metadata(metadata)
        return r

    def set_metadata_element(self, key, val):
        meta = self.get_metadata()
        meta[key] = val
        self.set_metadata(meta)

    def set_corners(self, min_x, min_y, max_x, max_y):
        meta = self.get_metadata()
        meta['min_x'] = min_x
        meta['min_y'] = min_y
        meta['max_x'] = max_x
        meta['max_y'] = max_y
        self.set_metadata(meta)

    def set_epsg(self, val):
        self.set_metadata_element('epsg', val)
    def get_epse(self):
        return self.get_metadata()['epsg']

    def get_corners(self):
        meta = self.get_metadata()
        return meta['min_x'], meta['min_y'], meta['max_x'], meta['max_y']
    # @property
    # def min_x(self):
    #     return self.get_metadata()['min_x']
    # @min_x.setter
    # def min_x(self, val):
    #     self.set_metadata_element('min_x', val)

    @property
    def width(self):
        min_x, min_y, max_x, max_y = self.get_corners()
        return max_x - min_x
    @property
    def height(self):
        min_x, min_y, max_x, max_y = self.get_corners()
        return max_y - min_y

    def xy_to_rc_using_dims(self, nrows, ncols, x, y):
        """Convert from real world x,y to row, col indices given the shape of the array or tile to be used"""
        min_x, min_y, max_x, max_y = self.get_corners()
        col = numpy.array(ncols * (x - min_x) / self.width, numpy.int32)
        row = numpy.array(nrows * (y - min_y) / self.height, numpy.int32)
        return row, col

    def xy_to_rc(self, x, y):
        """Convert from real world x,y to raster row, col indices"""
        array = self.get_array(0)
        return self.xy_to_rc_using_dims(array.shape[0], array.shape[1], x, y)

    def rc_to_xy_using_dims(self, nrows, ncols, r, c, center=False):
        """Convert from real world x,y to raster row, col indices"""
        min_x, min_y, max_x, max_y = self.get_corners()
        res_x = (max_x - min_x) / ncols
        res_y = (max_y - min_y) / nrows
        if center:
            fn = affine_center
        else:
            fn = affine
        return fn(r, c, min_x, res_x, 0, min_y, 0, res_y)

    def rc_to_xy(self, r, c):
        """Convert from real world x,y to raster row, col indices"""
        array = self.get_array(0)
        return self.xy_to_rc_using_dims(array.shape[0], array.shape[1], r, c)


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
        current_data = self.get_arrays(ALL_LAYERS)
        d = delta.get_arrays(ALL_LAYERS)
        indices = d[LayersEnum.MASK] == 1  # ~numpy.isnan(d)
        current_data[:, indices] = d[:, indices]
        r = RasterDelta(MemoryStorage(current_data, ALL_LAYERS))
        # r.set_arrays(current_data)
        return r


def arrays_dont_match(new_data, old_data):
    """If both arrays have a nan value treat that as equal even though numpy threats them as not equal"""
    not_both_nan = ~numpy.logical_and(numpy.isnan(new_data[:LayersEnum.MASK]), numpy.isnan(old_data[:LayersEnum.MASK]))
    diff_indices = numpy.logical_and(new_data[:LayersEnum.MASK] != old_data[:LayersEnum.MASK], not_both_nan)
    return diff_indices


def arrays_match(new_data, old_data):
    """If both arrays have a nan value treat that as equal even though numpy threats them as not equal"""
    return ~arrays_dont_match(new_data, old_data)


class RasterDelta(RasterData):
    def __init__(self, storage, arrays=None, layers=None):
        super().__init__(storage, arrays, layers)
        self._ver = 1

    @staticmethod
    def from_rasters(raster_old, raster_new):
        new_data = raster_new.get_arrays(ALL_LAYERS)
        old_data = raster_old.get_arrays(ALL_LAYERS)
        # not_both_nan = ~numpy.logical_and(numpy.isnan(new_data[:LayersEnum.MASK]), numpy.isnan(old_data[:LayersEnum.MASK]))
        # diff_indices = numpy.logical_and(new_data[:LayersEnum.MASK] != old_data[:LayersEnum.MASK], not_both_nan)
        diff_indices = arrays_dont_match(new_data, old_data)
        # numpy.logical_or.reduce(i)
        indices = numpy.any(diff_indices, axis=0)  # if any of the layers had a change then save them all the layers at that location, not strictly necessary
        delta_array = numpy.full(new_data.shape, numpy.nan)
        delta_array[:LayersEnum.MASK, indices] = old_data[:LayersEnum.MASK, indices]
        delta_array[LayersEnum.MASK, indices] = 1
        r = RasterDelta(MemoryStorage(delta_array, ALL_LAYERS))
        # r.set_arrays(delta_array)
        return r

