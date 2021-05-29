import json
from collections.abc import MutableSequence
import pathlib
import os
import re
from abc import ABC, abstractmethod

import numpy

from nbs.bruty.abstract import VABC
from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum

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

    @classmethod
    def exists(cls, data_path):
        pth = pathlib.Path(data_path)
        metaname = cls.make_meta_filename(pth)
        return metaname.exists()

    @staticmethod
    def make_meta_filename(data_path):
        pth = pathlib.Path(data_path)
        return pth.joinpath("metadata.json")

    @property
    def metadata_filename(self):
        return self.make_meta_filename(self.data_path)
        # return self.data_path.joinpath("metadata.json")

    def validate(self):
        meta = self.get_metadata()
        expected = set(meta['contributors'].keys())
        commits = {}
        for idx in range(len(self)):
            contrib = str(self[idx].get_metadata()['contrib_id'])  # use str since json returns it as strings
            commits.setdefault(contrib, 0)
            commits[contrib] += 1
        actual = set(commits.keys())
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        duplicates = [contrib for contrib, cnt in commits.items() if cnt > 1]
        return missing, extra, duplicates, expected, actual

    def get_metadata(self):
        """ Returns a copy of the metadata, so editing the returned dictionary will have no effect on the history itself.

        Returns
        -------
        dict
            keys should include 'min_x', 'min_y', 'max_x', 'max_y', 'contributors' and optionally 'epsg'
        """
        try:
            metadata = json.load(open(self.metadata_filename))
        except FileNotFoundError:
            metadata = {}
        except Exception as e:
            print(self.metadata_filename)
            raise e

        return metadata

    def set_metadata(self, meta):
        """ Writes a metadata.json file that is global to the history.  This is separate from the json files that may occur with the TiffStorage.

        Parameters
        ----------
        meta

        Returns
        -------

        """
        json.dump(meta, open(self.metadata_filename, "w"))

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
            os.remove(self.data_class.build_metapath(fname))
            for index in range(key+1, len(self)):
                from_filename, to_filename = self.filename_from_index(index), self.filename_from_index(index-1)
                os.rename(from_filename, to_filename)
                from_meta = self.data_class.build_metapath(from_filename)
                to_meta = self.data_class.build_metapath(to_filename)
                os.rename(from_meta, to_meta)

    def data_files(self):
        """ Finds all the files in the data_path directory that would match the naming convention

        Returns
        -------
        list of string filepaths
        """
        file_list = os.listdir(self.data_path)
        # note - caret is beginning of string and the dollar sign at the end means end of string
        # which is needed since arc will put an "_0000_.tif.aux.xml" in the directory which will match without the $
        return [fname for fname in file_list if re.match("^" + self.filename_pattern + "$", fname)]

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
            from_filename, to_filename = self.filename_from_index(index), self.filename_from_index(index + 1)
            os.rename(from_filename, to_filename)
            from_meta = self.data_class.build_metapath(from_filename)
            to_meta = self.data_class.build_metapath(to_filename)
            os.rename(from_meta, to_meta)
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
        # @todo fix this to be better serialization, this is goofy with get, set metadata
        self.history = history_data
        meta = self.get_metadata()
        if not meta or x1 is not None or x2 is not None:
            self.set_corners(x1, y1, x2, y2)
        else:
            self.min_x = meta["min_x"]
            self.min_y = meta['min_y']
            self.max_x = meta["max_x"]
            self.max_y = meta["max_y"]
            try:
                self.epsg = meta["epsg"]
            except KeyError:  # epsg was never set
                pass
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

    def validate(self):
        return self.history.validate()

    def get_metadata(self):
        return self.history.get_metadata()

    def set_metadata(self, meta):
        self.history.set_metadata(meta)

    def get_corners(self):
        return self.min_x, self.min_y, self.max_x, self.max_y

    def set_epsg(self, epsg):
        self.epsg = epsg
        meta = self.get_metadata()
        meta["epsg"] = self.epsg
        self.set_metadata(meta)

    def __setitem__(self, key, value):
        # @todo have to reform the deltas in the history like insert does.  Maybe a "reform" function is needed.
        print("setitem doesn't work yet, only append (not insert)")

    def __delitem__(self, key):
        # @todo have to reform the deltas in the history like insert does.  Maybe a "reform" function is needed.
        if isinstance(key, (list, tuple)):
            indices = sorted(key)  # remove from the end to beginning or the indices will remove the wrong items
            indices.reverse()
            return [self.__delitem__(index) for index in indices]
        elif isinstance(key, slice):
            if key.stop is not None and key.stop > len(self):
                key = slice(key.start, len(self), key.step)
            indices = list(range(*key.indices(len(self))))
            if indices[0] < indices[-1]:  # go from the end to the beginning as you can only remove the last item
                indices.reverse()
            return self.__delitem__(indices)
        else:
            key = self._abs_index(key)
            if key >= len(self) or key < 0:
                raise IndexError("history index out of range")
            if key != len(self) - 1:
                # @todo allow removing in the middle?
                #    figure out the raster at the last place before delete then
                #    make a delta for what changed on the last commit then
                #    add all the deltas from the raster we computed plus the remaining deltas without the one being deleted
                raise IndexError("Can only remove the last item, not something before since this is a stack of differences (this could change if needed)")
            rd = self[key-1]  # this retrieves a RasterData at the index specified, which can then be used to replace the RasterDelta that was there
            # metadata should be loaded with the raster data, right?
            # saved_meta = self.history[key-1].get_metadata()  # save this in case for when it gets removed by the delete below
            # rd.set_metadata(saved_meta)
            del self.history[key]  # remove the last thing which is a RasterData
            full_meta = self.get_metadata()
            # @todo get rid of these key strings and either use class variables or make a wrapper class with slots as these strings are not good design choice long term
            full_meta['contributors'] = {}
            if key > 0:
                del self.history[key - 1]  # remove the next to last delta and replace it with a RasterData
                self.history.append(rd)
                # since we can't trust that the ID was only added once in a tile
                # (when insert processing breaks the survey may have an multiple entries in the tiles but only one in the global database metadata)
                # we will rebuild the contributors by reading the histories and recreate the contributor dictionary
                for h in range(len(self)):
                    meta = self.history[h].get_metadata()
                    full_meta['contributors'][str(meta['contrib_id'])] = meta['contrib_path']
            self.set_metadata(full_meta)



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
            key = self._abs_index(key)
            if key >= len(self) or key < 0:
                raise IndexError("history index out of range")
            current_val = self.history[-1]
            for delta_idx in range(len(self) - 2, key-1, -1):
                rd = self.history[delta_idx]
                current_val = current_val.apply_delta(rd)
                current_val.set_metadata(rd.get_metadata())
            # current_val.set_corners(self.min_x, self.min_y, self.max_x, self.max_y)
            # current_val.set_epsg(self.epsg)
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
        meta = self.get_metadata()
        raster_meta = raster_val.get_metadata()
        meta.setdefault("contributors", {}).update(raster_meta.get('contributors', {}))
        self.set_metadata(meta)

class AccumulationHistory(RasterHistory):
    """ This class acts like a full history but really only keeps one raster, the current data.
    Insert is disabled except for at the last index, which is essentially [0] or [-1]
    """
    def insert(self, key, value):
        if len(self) > 0:
            # if we aren't empty then delete the existing file as it's about to be replaced
            del self.history[0]
        super().insert(0, value)

