import pathlib

import numpy

from xipe_dev.xipe2.abstract import VABC, abstractmethod
from xipe_dev.xipe2.tile_calculations import TMSTilesMercator, GoogleTilesMercator, GoogleTilesLatLon, UTMTiles, LatLonTiles

class WorldTilesBackend(VABC):
    """ Class to control Tile addressing.
    It should know what the projection of the tiles is and how they are split.
    Access should then be provided by returning a Tile object.
    """
    def __init__(self, tile_scheme, history_class, storage_class, data_class, data_path):
        """

        Parameters
        ----------
        tile_scheme
            an instance of a TilingScheme derived class which defines what the coordinate to tile index will be
        history_class
            A History derived class that will store the data like a mini-repo based on the tile scheme supplied - probably a RasterHistory
        storage_class
            The data storage for the history_class to use.  Probably a MemoryHistory or DiskHistory
        data_class
            Defines how to store the data, probably dervied from raster_data.Storage, like TiffStorage or MemoryStorage or BagStorage
        data_path
            Root directory to store file structure under, if applicable.
        """
        self._version = 1
        self.tile_scheme = tile_scheme
        self.data_path = pathlib.Path(data_path)
        self.data_class = data_class
        self.history_class = history_class
        self.storage_class = storage_class

    @property
    def __version__(self) -> int:
        return 1

    @property
    def epsg(self) -> int:
        pass

    def get_crs(self):
        return self.epsg

    def get_tile_history(self, x, y):
        tx, ty = self.tile_scheme.xy_to_tile_index(x, y)
        return self.get_tile_history_by_index(tx, ty)

    def tile_index_to_str(self, tx, ty):
        """ A function that can be overridden to change how directories are created.  The default is just to return the index as a string.
        An overridden version could add north/south or special named (like 2359, 4999 in TMS_mercator is Atlantic_Marine_Center).

        Parameters
        ----------
        tx
            tile index in x direction
        ty
            tile index in y direction

        Returns
        -------
        (str, str) of the x and y strings respectively.

        """
        return str(tx), str(ty)

    def get_tile_history_by_index(self, tx, ty):
        tx_str, ty_str = self.tile_index_to_str(tx, ty)
        history = self.history_class(self.storage_class(self.data_class, self.data_path.joinpath(tx_str).joinpath(ty_str)))
        return history

    def iter_tiles(self, x, y, x2, y2):
        for tx, ty in self.get_tiles_indices(x, y, x2, y2):
            yield self.get_tile_history_by_index(tx, ty)

    def get_tiles_indices(self, x, y, x2, y2):
        """ Get the indices of tiles that fall within rectangle specified by x,y to x2,y2 as a numpy array of tuples.
        Each entry of the returned array is the tx,ty index for a tile.
        Note that Tile indices are x -> tx which means it is essentially (column, row).

        e.g. if the tiles being returned were from tx = 1,2 and ty = 3,4 then the returned value would be
        array([[1, 3], [2, 3], [1, 4], [2, 4]])

        Parameters
        ----------
        x
        y
        x2
        y2

        Returns
        -------
        numpy array of shape (n,2)

        """
        xx, yy = self.get_tiles_index_matrix(x, y, x2, y2)
        return numpy.reshape(numpy.stack([xx, yy], axis=-1), (-1, 2))

    def get_tiles_index_matrix(self, x, y, x2, y2):
        """ Get the indices of tiles that fall within rectangle specified by x,y to x2,y2 as a pair of numpy arrays.
        Each entry of the returned array is the tx,ty index for a tile.
        Note that Tile indices are x -> tx which means it is essentially (column, row).

        e.g. if the tiles being returned were from tx = 1,2 and ty = 3,4 then the returned value would be
        (array([[1, 2], [1, 2]]) ,   array([[3, 3], [4, 4]]))

        Parameters
        ----------
        x
        y
        x2
        y2

        Returns
        -------
        Tile X and Tile Y numpy arrays of shape (n, m)

        """
        xs, ys = self.get_tiles_index_sparse(x, y, x2, y2)
        xx, yy = numpy.meshgrid(xs, ys)
        return xx, yy


    def get_tiles_index_sparse(self, x, y, x2, y2):
        """ Get the indices of tiles that fall within rectangle specified by x,y to x2,y2 as a sparse list.
        Each entry of the returned list is the tx or ty index for a tile.
        Note that Tile indices are x -> tx which means it is essentially (column, row).

        e.g. if the tiles being returned were from tx = 1,2 and ty = 3,4 then the returned value would be
        [1,2], [3,4]

        Parameters
        ----------
        x
        y
        x2
        y2

        Returns
        -------
        List for tile x of length n and list for tile y of length m

        """
        tx, ty = self.tile_scheme.xy_to_tile_index(x, y)
        tx2, ty2 = self.tile_scheme.xy_to_tile_index(x2, y2)
        xs = list(range(min(tx, tx2), max(tx, tx2) +1))
        ys = list(range(min(ty, ty2), max(ty, ty2) +1))
        return xs, ys

class LatLonBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = LatLonTiles(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)

class GoogleLatLonTileBackend(WorldTilesBackend):
    # https://gist.githubusercontent.com/maptiler/fddb5ce33ba995d5523de9afdf8ef118/raw/d7565390d2480bfed3c439df5826f1d9e4b41761/globalmaptiles.py
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = GoogleTilesLatLon(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)

class UTMTileBackend(WorldTilesBackend):
    def __init__(self, utm_epsg, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = UTMTiles(zoom=zoom_level)
        self.utm_epsg = utm_epsg
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)

class GoogleMercatorTileBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = GoogleTilesMercator(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)

class TMSMercatorTileBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = TMSTilesMercator(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)

class WMTileBackend(WorldTilesBackend):
    def __init__(self, storage, zoom_level=13):
        super().__init__(storage)


class Lock:
    def __init__(self, storage, id):
        pass
    def aquire(self, timeout=0):
        pass
    def release(self):
        pass
    def is_active(self):
        pass
    def __del__(self):
        self.release()
    def notify(self):
        pass

class ReadLock(Lock):  # actively being read
    pass
class WriteLock(Lock):  # actively being modified
    pass
class PendingReadLock(WriteLock):  # something wants to read but there are active/pending writes
    pass
class PendingWriteLock(WriteLock):  # something wants to modify but there are active reads
    pass


class Storage(VABC):
    """ Class to control the metadata associated with a WorldDatabaseBackend.
    This would store things like read/write/pending_write locks and hashes etc per Tile id.
    """
    @abstractmethod
    def readable_tile(self, id):
        # lock the file for reading
        pass

    @abstractmethod
    def writeable_tile(self, tile):
        # request lock for writing
        pass
    def get_read_lock(self, id):
        pass
    def get_write_lock(self, id):
        pass

class PostgresStorage(Storage):
    pass
class SQLite(Storage):
    pass
class PickleStorage(Storage):  # this would only work for single instances
    pass

class WorldDatabase:
    """ Class to control Tiles that cover the Earth.
    Also supplies locking of tiles for read/write.
    All access to the underlying data should go through this class to ensure data on disk is not corrupted.

    There are read and write locks.
    Multiple read requests can be honored at once.
    While a read is happening then a write will have to wait until reads are finished, but a write request lock is created.
    While there is a write request pending no more read requests will be allowed.

    A survey being inserted must have all its write requests accepted before writing to any of the Tiles.
    Otherwise a read could get an inconsistent state of some tiles having a survey applied and some not.
    """
    def __init__(self, backend):
        pass
    def insert_survey(self, survey_data):
        pass
    def export_area(self, area):
        pass
    def export_at_date(self, area, date):
        pass
    def remove_survey(self, survey_id):
        pass
    def revise_survey(self, survey_id, survey_data):
        pass
    def find_contributing_surveys(self, area):
        pass
    def find_area_affected_by_survey(self, survey_data):
        pass
    def change_survey_score(self, survey_id, new_score):
        pass
    def __cleanup_disk(self):
        pass


