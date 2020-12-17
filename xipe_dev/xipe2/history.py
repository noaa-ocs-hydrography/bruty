import json
from collections.abc import MutableSequence
import pathlib
import os
import re
from abc import ABC, abstractmethod

import numpy

from xipe_dev.xipe2.abstract import VABC
from xipe_dev.xipe2.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum

class History(VABC, MutableSequence):
    """ Base class for things that want to act like a list.  Intent is to maintain something that acts like a list and returns/stores
    instances of the data_class.  Could be derived from to store in memory or on disk or in a database.
    """
    def __init__(self, data_class, data_path="", prefix="", postfix=""):
        self.data_path = pathlib.Path(data_path)
        self.postfix = postfix
        self.prefix = prefix
        self.data_class = data_class

    def _abs_index(self, key):
        """ Returns a positive integer for and index.  If key is a negative number then it computes what the equivalent positive index would be.
        Parameters
        ----------
        key
            integer index to look up.
        Returns
        -------

        """
        if key < 0:
            key = len(self) + key
        return key

    def current(self):
        """ Gets the current object at the top of the stack (i.e. index [-1]
        Return
        -------
        Object at the last position
        """
        return self[-1]

    # @abstractmethod
    # def at_index(self, index):
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def at_commit(self, commit_id):
    #     raise NotImplementedError()

    def reverse(self) -> None:
        raise NotImplementedError("Don't use reverse to change data in place, use __reversed__() -- "
                                  "otherwise this has permanent effects on disk storage that are probably unintentional")

# using xarray and zarr might remove this DiskHistory class -- just keep the whole world as a huge zarr array?
# n x m tiles with  i x j cells with nd layers amd then t for the various commits which would be indexed as [n,m,t,i,j,nd].
# Would this be safe for operations that 'fail' and need to revert and how to lock so operations don't read from a dataset that is being written to?
# essentially we need a sparse matrix representation which could be scipy.sparse or zarr
class DiskHistory(History):
    """ Store 'data_class' objects to disk located at 'data_path' and allow access to them like a list.
    """
    def __init__(self, data_class, data_path, prefix="", postfix=""):
        """ Setup to store data_class objects to disk using the prefix and postfix in the filenames and store them under the directory 'data_path'.
        Files will be named something like <prefix>_0001_<postfix>.tif

        Parameters
        ----------
        data_class
            Type of data to store, probably RasterData.
        data_path
            Location to store files.  Files will be named numerically.
        prefix
            prefix the data files stored under data_path.
        postfix
            postfix for the files storyed under data_path.
        """
        super().__init__(data_class, data_path, prefix, postfix)
        os.makedirs(self.data_path, exist_ok=True)
    @property
    def filename_pattern(self):
        """ Gets the pattern to use to find valid files inside the self.data_path directory.

        Returns
        -------
        string to use with regular expression search
        """
        return f'{self.prefix}_\\d+_{self.postfix}{self.data_class.extension}'

    def filename_from_index(self, key):
        """ Create a filename based on the number 'key'

        Parameters
        ----------
        key
            integer to use to make the file name

        Returns
        -------
            string of the full path to the file
        """
        fname = self.filename_pattern.replace("\\d+", "%06d"%self._abs_index(key))
        return self.data_path.joinpath(fname)

    def get_metadata(self):
        try:
            metadata = json.load(open(self.data_path.joinpath("metadata.json")))
        except FileNotFoundError:
            metadata = {}
        return metadata

    def set_metadata(self, meta):
        json.dump(meta, open(self.data_path.joinpath("metadata.json"), "w"))

    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            return [self.__getitem__(index) for index in key]
        elif isinstance(key, slice):
            return [self.__getitem__(index) for index in range(*key.indices(len(self)))]
        else:
            fname = self.filename_from_index(key)
            return RasterData(self.data_class(fname))

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            for i, col_index in enumerate(key):
                self.__setitem__(col_index, value[i])
        elif isinstance(key, slice):
            for i, col_index in enumerate(range(*key.indices(len(self)))):
                self.__setitem__(col_index, value[i])
        else:
            fname = self.filename_from_index(key)
            data = self.data_class(fname)
            data.set(value)

    def __delitem__(self, key):
        if isinstance(key, (list, tuple)):
            for index in sorted(key):
                self.__delitem__(index)
        elif isinstance(key, slice):
            remove_indices = range(*key.indices(len(self)))
            for index in enumerate(remove_indices.__reversed__()):
                self.__delitem__(index)
        else:
            key = self._abs_index(key)
            fname = self.filename_from_index(key)
            os.remove(fname)
            for index in range(key+1, len(self)):
                os.rename(self.filename_from_index(index), self.filename_from_index(index-1))

    def data_files(self):
        """ Finds all the files in the data_path directory that would match the naming convention

        Returns
        -------
        list of string filepaths
        """
        file_list = os.listdir(self.data_path)
        return [fname for fname in file_list if re.match(self.filename_pattern, fname)]

    def __len__(self):
        return len(self.data_files())

    def insert(self, key, value):
        """ Insert an instance into the list of data_class objects.
        This will rename the existing files and then create a new file at the desired index.

        Parameters
        ----------
        key
            position in the list to insert at
        value
            instance to put into the list.
        Returns
        -------
        None

        """
        key = self._abs_index(key)
        # fname = self.filename_from_index(key)
        # Rename the existing files from the end to the location desired and then create the new data at the position that was opened up.
        for index in range(len(self) - 1, key - 1, -1):
            os.rename(self.filename_from_index(index), self.filename_from_index(index+1))
        # put the new data in place
        self.__setitem__(key, value)


class MemoryHistory(History):
    """ Stores data_class objects in memory.  This is basically a thin wrapper around a list.
    """
    def __init__(self, data_class=None, data_path="", prefix="", postfix=""):
        """
        Parameters
        ----------
        data_class
            instance class expected to store.
        data_path
            unused
        prefix
            unused
        postfix
            unused
        """
        super().__init__(data_class, data_path, prefix, postfix)
        self.data = []
        self.metadata = {}
    def get_metadata(self):
        return self.metadata

    def set_metadata(self, meta):
        self.metadata = meta
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    def __setitem__(self, key, value):
        return self.data.__setitem__(key, value)
    def __delitem__(self, key):
        return self.data.__delitem__(key)
    def __len__(self):
        return self.data.__len__()
    def insert(self, key, value):
        return self.data.insert(key, value)


class RasterHistory(History):
    """ This class works on top of a History (passed in via 'history_data') instance to return the actual surface at a certain time
    as opposed to the deltas that the 'history_data' holds
    """
    def __init__(self, history_data, x1=None, y1=None, x2=None, y2=None):
        """ Acts like a list of RasterData while actually storing the current raster as a full dataset and the others as deltas vs the current data.

        Parameters
        ----------
        history_data
            A MemoryHistory or DiskHistory to use for storage
        min_x
            an X coordinate for the area being kept
        min_y
            a Y coordinate for the area being kept
        max_x
            an X coordinate for the area being kept
        max_y
            a Y coordinate for the area being kept
        """
        self.set_corners(x1, y1, x2, y2)
        self.history = history_data
        self.sources = []

    def set_corners(self, min_x, min_y, max_x, max_y):
        if min_x is not None and max_y is not None:
            self.max_y = max(min_y, max_y)
            self.max_x = max(min_x, max_x)
            self.min_y = min(min_y, max_y)
            self.min_x = min(min_x, max_x)
            meta = self.get_metadata()
            meta["min_x"] = self.min_x
            meta['min_y'] = self.min_y
            meta["max_x"] = self.max_x
            meta["max_y"] = self.max_y
            self.set_metadata(meta)
        else:
            self.max_x = self.max_y = self.min_x = self.min_y = None

    def get_metadata(self):
        return self.history.get_metadata()

    def set_metadata(self, meta):
        self.history.set_metadata(meta)

    def get_corners(self):
        return self.min_x, self.min_y, self.max_x, self.max_y

    def set_epsg(self, epsg):
        self.epsg = epsg

    def __setitem__(self, key, value):
        # @todo have to reform the deltas in the history like insert does.  Maybe a "reform" function is needed.
        print("setitem doesn't work yet, only append (not insert)")

    def __delitem__(self, key):
        # @todo have to reform the deltas in the history like insert does.  Maybe a "reform" function is needed.
        self.history.__delitem__(key)
        print("history delete doesn't work yet")

    def __len__(self):
        return len(self.history)

    def make_empty_data(self, rows, cols):
        arr = numpy.full((len(LayersEnum)-1, rows, cols), numpy.nan)  # don't supply mask
        raster_val = RasterData(MemoryStorage(), arr)
        raster_val.set_corners(self.min_x, self.min_y, self.max_x, self.max_y)
        raster_val.set_epsg(self.epsg)
        return raster_val

    def __getitem__(self, key):
        """Return full raster data at the index"""
        if isinstance(key, (list, tuple)):
            return [self.__getitem__(index) for index in key]
        elif isinstance(key, slice):
            if key.stop > len(self):
                key = slice(key.start, len(self), key.step)
            return [self.__getitem__(index) for index in range(*key.indices(len(self)))]
        else:
            if key < 0:
                key = len(self.history) + key
            if key >= len(self.history) or key < 0:
                raise IndexError("history index out of range")
            current_val = self.history[-1]
            for delta_idx in range(len(self.history) - 2, key-1, -1):
                current_val = current_val.apply_delta(self.history[delta_idx])
            current_val.set_corners(self.min_x, self.min_y, self.max_x, self.max_y)
            current_val.set_epsg(self.epsg)
            return current_val

    def insert(self, key, value):
        # First make sure the data is a RasterData instance
        if not isinstance(value, RasterData):
            raster_val = RasterData(MemoryStorage())
            raster_val.set_arrays(value)
        else:
            raster_val = value
        # find out where the data should go in the existing list
        key = self._abs_index(key)

        if len(self) > 0:
            # we need to make room on disk and also recompute the raster deltas that are stored after the insertion point
            # create a list of the rasters back to the point being inserted at
            current_val = self.history[-1]  # get the last item which is a full raster then work our way back with the deltas
            raster_stack = [current_val]
            for delta_idx in range(len(self.history) - 2, key - 1, -1):
                current_val = current_val.apply_delta(self.history[delta_idx])
                raster_stack.insert(0, current_val)
            # we either have gone back one point past the spot we want (insert at 1) or if being added to the beginning then use 0
            if key > 0:
                insert_index = 1
            else:
                insert_index = 0
            raster_stack.insert(insert_index, raster_val)  # this will be at the end if the key was equal to the len(self)
            # create new deltas from the insertion point to the end.
            for stack_index, history_index in enumerate(range(key - 1, len(self))):
                raster_delta = RasterDelta.from_rasters(raster_stack[stack_index], raster_stack[stack_index + 1])
                self.history[history_index] = raster_delta  # put the new delta into the history
            self.history.append(raster_stack[-1])  # and finally add a full raster data as the end of the history
        else:
            # if we are empty then just put the full array in
            self.history.append(raster_val)

class AccumulationHistory(RasterHistory):
    """ This class acts like a full history but really only keeps one raster, the current data.
    Insert is disabled except for at the last index, which is essentially [0] or [-1]
    """
    def insert(self, key, value):
        if len(self) > 0:
            # if we aren't empty then delete the existing file as it's about to be replaced
            del self.history[0]
        super().insert(0, value)


class PointsHistory(History):
    pass