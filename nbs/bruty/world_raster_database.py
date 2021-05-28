import json
import os
import pathlib
import shutil
import tempfile
import time
import gc
from datetime import datetime

import numpy

import rasterio.crs
from osgeo import gdal, osr, ogr

from HSTB.drivers import bag
from nbs.bruty.utils import merge_arrays, merge_array, get_crs_transformer, onerr, tqdm, make_gdal_dataset_size, make_gdal_dataset_area, \
            calc_area_array_params, compute_delta_coord, iterate_gdal_image, add_uncertainty_layer, transform_rect
from nbs.bruty.raster_data import TiffStorage, LayersEnum, RasterData, affine, inv_affine, affine_center, arrays_dont_match
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.bruty.abstract import VABC, abstractmethod
from nbs.bruty.tile_calculations import TMSTilesMercator, GoogleTilesMercator, GoogleTilesLatLon, UTMTiles, LatLonTiles, TilingScheme, \
            ExactUTMTiles, ExactTilingScheme
from nbs.bruty import morton
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, Lock, EXCLUSIVE, SHARED, NON_BLOCKING


geo_debug = False
_debug = False

no_lock = False

if no_lock:  # too many file locks for windows is preventing some surveys from processing.  Use this when I know only one process is running.
    class AreaLock(AreaLock):
        def __init__(self, *args, **kywds):
            pass
        def acquire(self):
            return True
        def release(self):
            pass


NO_OVERRIDE = -1

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
            Defines how to store the data, probably derived from raster_data.Storage, like TiffStorage or MemoryStorage or BagStorage
        data_path
            Root directory to store file structure under, if applicable.
        """
        self._version = 1
        self._loaded_from_version = None
        self.tile_scheme = tile_scheme
        self.data_path = pathlib.Path(data_path)
        self.data_class = data_class
        self.history_class = history_class
        self.storage_class = storage_class
        if data_path:
            os.makedirs(self.data_path, exist_ok=True)
        self.to_file()  # store parameters so it can be loaded back from disk

    @staticmethod
    def from_file(data_dir, filename="backend_metadata.json"):
        data_path = pathlib.Path(data_dir).joinpath(filename)
        infile = open(data_path, 'r')
        data = json.load(infile)
        data['data_path'] = pathlib.Path(data_dir)  # overridde in case the user copied the data to a different path
        return WorldTilesBackend.create_from_json(data)

    def to_file(self, data_path=None):
        if not data_path:
            data_path = self.data_path.joinpath("backend_metadata.json")
        outfile = open(data_path, 'w')
        json.dump(self.for_json(), outfile)

    def for_json(self):
        json_dict = {'class': self.__class__.__name__,
                     'module': self.__class__.__module__,
                     'data_path': str(self.data_path),
                     'version': self._version,
                     'tile_scheme': self.tile_scheme.for_json(),
                     'data_class': self.data_class.__name__,
                     'history_class': self.history_class.__name__,
                     'storage_class': self.storage_class.__name__,
                     }
        return json_dict

    @staticmethod
    def create_from_json(json_dict):
        cls = eval(json_dict['class'])
        # bypasses the init function as we will use 'from_json' to initialize - like what pickle does
        obj = cls.__new__(cls)
        obj.from_json(json_dict)
        return obj

    def from_json(self, json_dict):
        self.tile_scheme = TilingScheme.create_from_json(json_dict['tile_scheme'])
        self.data_class = eval(json_dict['data_class'])
        self.history_class = eval(json_dict['history_class'])
        self.storage_class = eval(json_dict['storage_class'])
        self.data_path = pathlib.Path(json_dict['data_path'])
        # do any version updates here
        self._loaded_from_version = json_dict['version']
        self._version = json_dict['version']

    def iterate_filled_tiles(self):
        for tx_dir in os.scandir(self.data_path):
            if tx_dir.is_dir():
                for ty_dir in os.scandir(tx_dir):
                    if ty_dir.is_dir():
                        tx, ty = self.str_to_tile_index(tx_dir.name, ty_dir.name)
                        tile_history = self.get_tile_history_by_index(tx, ty)
                        try:
                            raster = tile_history[-1]
                            yield tx, ty, raster, tile_history.get_metadata()
                        except IndexError:
                            print("accumulation db made directory but not filled?", ty_dir.path)

    def append_accumulation_db(self, accumulation_db):
        # iterate the acculation_db and append the last rasters from that into this db
        for tx, ty, raster, meta in accumulation_db.iterate_filled_tiles():
            tile_history = self.get_tile_history_by_index(tx, ty)
            tile_history.append(raster)

    def make_accumulation_db(self, data_path):
        """ Make a database that has the same layout and types, probably a temporary copy while computing tiles.

        Parameters
        ----------
        data_path
            Place to store the data.  A local temporary directory or subdirectory inside of this directory would make sense if deleted later

        Returns
        -------
            WorldTilesBackend instance

        """
        # Use the same tile_scheme (which is just geographic parameters) with an AccumulationHistory to allow multiple passes on the same dataset to be added to the main database at one time
        use_history = AccumulationHistory
        new_db = WorldTilesBackend(self.tile_scheme, use_history, self.storage_class, self.data_class, data_path)
        return new_db

    @property
    def __version__(self) -> int:
        return self._version

    @property
    def epsg(self) -> int:
        return self.tile_scheme.epsg

    def get_crs(self):
        return self.epsg

    def get_tile_history(self, x, y):
        tx, ty = self.tile_scheme.xy_to_tile_index(x, y)
        return self.get_tile_history_by_index(tx, ty)

    def str_to_tile_index(self, strx, stry):
        """Inverses the tile_index_to_str naming"""
        return int(strx), int(stry)

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

    def get_history_path_by_index(self, tx, ty):
        tx_str, ty_str = self.tile_index_to_str(tx, ty)
        return self.data_path.joinpath(tx_str).joinpath(ty_str)

    def get_tile_history_by_index(self, tx, ty):
        hist_path = self.get_history_path_by_index(tx, ty)
        history = self.history_class(self.storage_class(self.data_class, hist_path))
        lx, ly, ux, uy = self.tile_scheme.tile_index_to_xy(tx, ty)
        history.set_corners(lx, ly, ux, uy)
        history.set_epsg(self.epsg)
        return history

    def iter_tiles(self, x, y, x2, y2):
        for tx, ty in self.get_tiles_indices(x, y, x2, y2):
            yield tx, ty, self.get_tile_history_by_index(tx, ty)

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
        xs = list(range(min(tx, tx2), max(tx, tx2) + 1))
        ys = list(range(min(ty, ty2), max(ty, ty2) + 1))
        return xs, ys


class LatLonBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13, epsg=None):
        tile_scheme = LatLonTiles(zoom=zoom_level, epsg=epsg)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class GoogleLatLonTileBackend(WorldTilesBackend):
    # https://gist.githubusercontent.com/maptiler/fddb5ce33ba995d5523de9afdf8ef118/raw/d7565390d2480bfed3c439df5826f1d9e4b41761/globalmaptiles.py
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = GoogleTilesLatLon(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class UTMTileBackend(WorldTilesBackend):
    def __init__(self, utm_epsg, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = UTMTiles(zoom=zoom_level, epsg=utm_epsg)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class UTMTileBackendExactRes(WorldTilesBackend):
    def __init__(self, res_x, res_y, utm_epsg, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = ExactUTMTiles(res_x, res_y, zoom=zoom_level, epsg=utm_epsg)
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


class WorldDatabase(VABC):
    """ Class to control Tiles that cover the Earth.
    Also supplies locking of tiles for read/write.
    All access to the underlying data should go through this class to ensure data on disk is not corrupted.

    There are (will be) read and write locks.
    Multiple read requests can be honored at once.
    While a read is happening then a write will have to wait until reads are finished, but a write request lock is created.
    While there is a write request pending no more read requests will be allowed.

    A survey being inserted must have all its write requests accepted before writing to any of the Tiles.
    Otherwise a read could get an inconsistent state of some tiles having a survey applied and some not.
    """

    def __init__(self, backend):
        self.db = backend
        # @todo, should the survey lists be stored in a sqlite database?
        self.included_surveys = {}
        self.included_ids = {}
        self.to_file()

    @staticmethod
    def open(data_dir):
        data_path = WorldDatabase.metadata_filename_in_dir(data_dir)
        with Lock(data_path, 'r', SHARED) as infile:  # this will wait for the file to be available
            data = json.load(infile)
        data['data_path'] = pathlib.Path(data_dir)  # overridde in case the user copied the data to a different path
        cls = eval(data['class'])
        obj = cls.__new__(cls)
        obj.from_json(data)
        return obj

    @staticmethod
    def metadata_filename_in_dir(data_dir):
        filename = "wdb_metadata.json"
        data_path = pathlib.Path(data_dir).joinpath(filename)
        return data_path

    def metadata_filename(self):
        return WorldDatabase.metadata_filename_in_dir(self.db.data_path)

    def update_metadata_from_disk(self, data_path=None, locked_file=None):
        local_lock = None
        if data_path is None:
            data_path = self.metadata_filename()
        # Lock the file so no other processes change it at the same time as reading
        if locked_file is None:
            print('locking metadata (shared) at ', datetime.now().isoformat())
            local_lock = Lock(data_path, 'r', SHARED)  # this will wait for the file to be available
            local_lock.acquire()
            infile = local_lock.openfile()
        else:
            infile = locked_file
        data = json.load(infile)
        self.update_metadata_from_json(data)
        if local_lock is not None:
            local_lock.release()
            print('UNlocking metadata at ', datetime.now().isoformat())

    def to_file(self, data_path=None, locked_file=None):
        local_lock = None
        if locked_file is None:
            if not data_path:
                data_path = self.db.data_path.joinpath("wdb_metadata.json")
            # outfile = open(data_path, 'w')
            local_lock = Lock(data_path, 'w', EXCLUSIVE)  # this will wait for the file to be available
            local_lock.acquire()
            outfile = local_lock.openfile()
        else:
            outfile = locked_file
        json.dump(self.for_json(), outfile, indent=0)
        if local_lock is not None:
            local_lock.release()

    def for_json(self):
        # @todo make a base class or mixin that auto converts to json,
        #    maybe list the attributes desired with a callback that handles anything not auto converted to json
        json_dict = {'class': self.__class__.__name__,
                     'module': self.__class__.__module__,
                     'survey_paths': self.included_surveys,
                     'survey_ids': self.included_ids,
                     }
        return json_dict

    def from_json(self, json_dict):
        """Build the entire object from json"""
        self.db = WorldTilesBackend.from_file(json_dict['data_path'])
        self.update_metadata_from_json(json_dict)

    def update_metadata_from_json(self, json_dict):
        """Just build the survey metadata lists (another process may have written) rather than all objects"""
        # json stores all dictionary keys as strings, so have to convert back to integers
        self.included_ids = {int(key): val for key, val in json_dict['survey_ids'].items()}
        self.included_surveys = json_dict['survey_paths']

    def insert_survey(self, path_to_survey_data, override_epsg=NO_OVERRIDE, contrib_id=numpy.nan, compare_callback=None, reverse_z=False):
        done = False
        extension = pathlib.Path(str(path_to_survey_data).lower()).suffix
        if extension == '.bag':
            try:
                vr = bag.VRBag(path_to_survey_data, mode='r')
            except (bag.BAGError, ValueError):  # value error for single res of version < 1.6.  Maybe should change to BAGError in bag.py
                pass
            else:
                self.insert_survey_vr(vr, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback, reverse_z=reverse_z)
                done = True
        if not done:
            if extension in ['.bag', '.tif', '.tiff']:
                self.insert_survey_gdal(path_to_survey_data, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback, reverse_z=reverse_z)
                done = True
        if not done:
            if extension in ['.csar',]:
                # export to xyz
                # FIXME -- export points from csar and support LAS or whatever points file is decided on.
                # self.insert_txt_survey(path_to_survey_data, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback, reverse_z=reverse_z)
                done = True

    def insert_txt_survey(self, path_to_survey_data, survey_score=100, flags=0, format=None, override_epsg=NO_OVERRIDE,
                          contrib_id=numpy.nan, compare_callback=None, reverse_z=False):
        """ Reads a text file and inserts into the tiled database.
        The format parameter is passed to numpy.loadtxt and needs to have names of x, y, depth, uncertainty.

        Parameters
        ----------
        path_to_survey_data
            full path filename to read using numpy
        survey_score
            score to apply to data, if not a column in the data
        flags
            flags to apply to data, if not a column in the data
        format
            numpy dtype format to pass to numpy.loadtxt, default is [('y', 'f8'), ('x', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')]
        transformer
            Optional function used to transform from x,y in the file to the coordinate system of the database.
            It will be called as new_x, new_y = func( x, y ).

        Returns
        -------
        None
        """
        if override_epsg == NO_OVERRIDE:
            epsg = self.db.epsg
        else:
            epsg = override_epsg
        transformer = get_crs_transformer(epsg, self.db.epsg)

        if not format:
            format = [('y', 'f8'), ('x', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')]
        data = numpy.loadtxt(path_to_survey_data, dtype=format)
        x = data['x']
        y = data['y']
        if transformer:
            x, y = transformer.transform(x, y)
        depth = data['depth']
        if reverse_z:
            depth *= -1
        uncertainty = data['uncertainty']
        score = numpy.full(x.shape, survey_score)
        flags = numpy.full(x.shape, flags)
        txs, tys = self.db.tile_scheme.xy_to_tile_index(x, y)
        tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
        with AreaLock(tile_list, EXCLUSIVE|NON_BLOCKING, self.db.get_history_path_by_index) as lock:
            if contrib_id is None or contrib_id not in self.included_ids:
                tiles = self.insert_survey_array(numpy.array((x, y, depth, uncertainty, score, flags)), path_to_survey_data,
                                                 contrib_id=contrib_id, compare_callback=compare_callback)
                self.finished_survey_insertion(path_to_survey_data, tiles, contrib_id)
            else:
                raise Exception(f"Survey Exists already in database {contrib_id}")

    # @todo - make the survey_ids and survey_paths into properties that load from disk when called by user so that they stay in sync.
    #    Otherwise use postgres to hold that info so queries are current.
    def finished_survey_insertion(self, path_to_survey_data, tiles, contrib_id=numpy.nan):
        # store the tiles filled by this survey as a convenience lookup for the future when removing or querying.
        print('locking metadata for exclusive at ', datetime.now().isoformat())
        with Lock(self.metadata_filename(), 'r+', EXCLUSIVE) as metadata_file:
            backup_path1 = self.metadata_filename().parent.joinpath("wdb_metadata.bak1")
            backup_path2 = self.metadata_filename().parent.joinpath("wdb_metadata.bak2")
            self.to_file(backup_path1)
            # update the metadata in case another process wrote to it since the last time we updated/loaded
            self.update_metadata_from_disk(locked_file=metadata_file)
            # make a backup of the metadata in case something breaks here
            self.to_file(backup_path2)
            # add the new survey to the metadata and store to disk
            # json doesn't like pathlib.Paths to be stored -- convert to strings
            self.included_surveys[str(path_to_survey_data)] = (contrib_id, list(tiles))  # json doesn't like sets, convert to list
            if contrib_id is not None:
                self.included_ids[contrib_id] = (str(path_to_survey_data), list(tiles))
            # rather than overwrite, since we have it locked, truncate the file and then write new data to it
            metadata_file.seek(0)
            metadata_file.truncate(0)
            self.to_file(locked_file=metadata_file)
            print('unlocking metadata at ', datetime.now().isoformat())

    def insert_survey_array(self, input_survey_data, contrib_name, accumulation_db=None, contrib_id=numpy.nan, compare_callback=None):
        """ Insert a numpy array (or list of lists) of data into the database.

        Parameters
        ----------
        input_survey_data
            numpy array or list of lists in this configuration (x, y, depth, uncertainty, score, flags)
        contrib_name
            pathname or contributor name to associate with this data.
        accumulation_db
            If multiple calls will be made from the same survey then an accumulation database can be supplied.
            This will keep the contributor from having multiple records in the history.
            A subsequent call to db.append_accumulation_db would be needed to transfer the data into this database.
        contrib_id
            An integer id for the dataset being supplied
        compare_callback
            A function which if supplied (also requires contrib_id) will be called with the data being inserted and the existing raster data.
            The function(pts, new_arrays) should return arrays and if sorts should be reversed (sort_vals, new_sort_values, reverse).
            NOTE: pts is ordered as 'raster_data.LayersEnum' with X, Y prepended (so raster_data.Layers(Elevation) is index=2)
                new_arrays is just in default raster_data order

            As an example, here is what happens if compare_callback is None  (Score then depth)

            sort_vals = (pts[LayersEnum.SCORE + 2], pts[LayersEnum.ELEVATION + 2])
            new_sort_values = numpy.array((new_arrays[LayersEnum.SCORE], new_arrays[LayersEnum.ELEVATION]))
            reverse = (False, False)


        Returns
        -------
        None
        """
        # fixme for survey_data - use pandas?  structured arrays?

        # @todo Allow for pixel sizes, right now treating as point rather than say 8m or 4m coverages
        if not isinstance(contrib_name, str):
            contrib_name = str(contrib_name)
        if not accumulation_db:
            accumulation_db = self.db
        if contrib_id is None:
            contrib_id = numpy.nan
        contributor = numpy.full(input_survey_data[0].shape, contrib_id)
        survey_data = numpy.array((input_survey_data[0], input_survey_data[1], input_survey_data[2],
                                   input_survey_data[3], contributor, input_survey_data[4], input_survey_data[5]))
        # if a multidemensional array was passed in then reshape it to be one dimenion.
        # ex: the input_survey_data is 4x4 data with the 6 required data fields (x,y,z, uncertainty, score, flag),
        #   so 6 x 4 x 5 -- becomes a 6 x 20 array instead
        if len(survey_data.shape) > 2:
            survey_data = survey_data.reshape(survey_data.shape[0], -1)
        # Compute the tile indices for each point
        txs, tys = accumulation_db.tile_scheme.xy_to_tile_index(survey_data[0], survey_data[1])
        tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
        # itererate each tile that was found to have data
        # @todo figure out the contributor - should be the unique id from the database of surveys
        # @FIXME - confirm that points outside bounds (less than lower left and greater than upper right) don't crash this
        for i_tile, (tx, ty) in enumerate(tile_list):
            if _debug:
                pass
                # print("debug skipping tiles")
                # if tx != 3325 or ty != 3207:  # utm 16, US5PLQII_utm, H13196 -- gaps in DB and exported enc cells
                # if tx != 3325 or ty != 3207:  # utm 16, US5MSYAF_utm, H13193 (raw bag is in utm15 though) -- gaps in DB and exported enc cells  217849.73 (m), 3307249.86 (m)
                # if tx != 4614 or ty != 3227:  # utm 15 h13190 -- area with res = 4.15m (larger than the 4m output)
                # if tx != 4615 or ty != 3227:  # utm 15 h13190 -- area with res = 4.15m (larger than the 4m output)
                # if tx != 4148 or ty != 4370:
                #     continue
            print(f'processing tile {i_tile + 1} of {len(tile_list)}')
            tile_history = accumulation_db.get_tile_history_by_index(tx, ty)
            try:
                raster_data = tile_history[-1]
            except IndexError:
                # if the accumulation_db is empty then get the last tile from the main db
                tile_history_main = self.db.get_tile_history_by_index(tx, ty)
                try:
                    raster_data = tile_history_main[-1]
                except IndexError:
                    # empty tile, allocate one
                    rows, cols = self.init_tile(tx, ty, tile_history)
                    raster_data = tile_history.make_empty_data(rows, cols)
                    if geo_debug and True:  # draw a box around the tile and put a notch in the 0,0 corner
                        make_outlines = raster_data.get_arrays()
                        make_outlines[:, :, 0] = 99  # outline around box
                        make_outlines[:, :, -1] = 99
                        make_outlines[:, 0, :] = 99
                        make_outlines[:, -1, :] = 99
                        make_outlines[:, 0:15, 0:15] = 44  # registration notch at 0,0
                        raster_data.set_arrays(make_outlines)

            new_arrays = raster_data.get_arrays()
            # just operate on data that falls in this tile
            pts = survey_data[:, numpy.logical_and(txs == tx, tys == ty)]

            # replace x,y with row, col for the points
            i, j = raster_data.xy_to_rc_using_dims(new_arrays.shape[1], new_arrays.shape[2], pts[0], pts[1])
            # if there is a contributor ID and callback,
            # pass the existing contributors to the callback function and genenerate as many comparison matrices as needed
            # also allow for depth (elevation) to be placed into the comparison matrices.
            # @fixme confirm this is working
            if compare_callback is not None and contrib_id is not None:
                # pts has x,y then columns in RasterData order, so slice off the first two columns
                sort_vals, new_sort_values, reverse = compare_callback(pts[2:], new_arrays)
            else:  # default to score, depth with no reversals
                new_sort_values = numpy.array((new_arrays[LayersEnum.SCORE], new_arrays[LayersEnum.ELEVATION]))
                sort_vals = (pts[LayersEnum.SCORE + 2], pts[LayersEnum.ELEVATION + 2])
                reverse = (False, False)

            # combine the existing data ('new_arrays') with newly supplied data (pts derived from survey_data)
            # negative depths, so don't reverse the sort of the second key (depth)
            merge_arrays(i, j, sort_vals, pts[2:], new_arrays, new_sort_values, reverse_sort=reverse)
            # create a new raster data set to either replace the old one in the database or add a new history item
            # depending on if there is an accumulation buffer (i.e. don't save history) or regular database (save history of additions)
            rd = RasterData.from_arrays(new_arrays)
            rd.set_metadata(raster_data.get_metadata())  # copy the metadata to the new raster
            rd.set_last_contributor(contrib_id, contrib_name)
            tile_history.append(rd)

            # for x, y, depth, uncertainty, score, flag in sorted_pts:
            #     # fixme: score should be a lookup into the database so that data/decay is considered correctly
            #
            #     # @todo If scores are all computed then we could overwrite rapidly based on index,
            #     #   but has to be sorted by depth so the shallowest sounding in a cell is retained in case there were multiple
            #     #   We are not trying to implement CUBE or CHRT here, just pick a value and shoalest is safest
            #     # raster_data = raster_data[]
        return [tuple((int(tx), int(ty))) for tx, ty in tile_list]  # convert to a vanilla python int for compatibility with json

    def init_tile(self, tx, ty, tile_history):
        # @todo lookup the resolution to use by default.
        #   Probably will be a lookup based on the ENC cell the tile covers and then twice the resolution needed for that cell
        # @todo once resolution is determined then convert to the right size in the coordinate system to represent that sizing in meters

        # this should be ~2m when zoom 13 is the zoom level used (zoom 13 = 20m at 256 pix, so 8 times finer)
        # return rows, cols
        if isinstance(self.db.tile_scheme, ExactTilingScheme):
            # rows and columns
            # get the x,y bounds and figure out how many pixels (cells) would fit
            lx, ly, ux, uy = self.db.tile_scheme.tile_index_to_xy(tx, ty)
            #  -- since it is supposed to be an exact fit round up any numerical errors and truncate to an int
            return int(0.00001 + (uy-ly)/self.db.tile_scheme.res_y), int(0.00001+(ux-lx)/self.db.tile_scheme.res_x)
        else:
            return 512, 512

    def insert_survey_vr(self, vr, survey_score=100, flag=0, override_epsg=NO_OVERRIDE, contrib_id=numpy.nan, compare_callback=None, reverse_z=False):
        """
        Parameters
        ----------
        vr
        survey_score
        flag

        Returns
        -------

        """
        if not isinstance(vr, bag.VRBag):
            vr = bag.VRBag(vr, mode='r')

        print("@todo - do transforms correctly with proj/vdatum etc")
        if override_epsg == NO_OVERRIDE:
            epsg = rasterio.crs.CRS.from_string(vr.srs.ExportToWkt()).to_epsg()
        else:
            epsg = override_epsg
        crs_transformer = get_crs_transformer(epsg, self.db.epsg)

        # adjust for the VR returning center of supercells and not the edge.
        supercell_half_x = vr.cell_size_x / 2.0
        supercell_half_y = vr.cell_size_y / 2.0
        x1 = vr.minx - supercell_half_x
        y1 = vr.miny - supercell_half_y
        x2 = vr.maxx + supercell_half_x
        y2 = vr.maxy + supercell_half_y
        # convert to target reference system if needed
        if crs_transformer is not None:
            ((x1, y1), (x2, y2)) = crs_transformer.transform((x1, x2), (y1, y2))
        tile_list = self.db.get_tiles_indices(x1, y1, x2, y2)
        with AreaLock(tile_list, EXCLUSIVE|NON_BLOCKING, self.db.get_history_path_by_index) as lock:
            if contrib_id is None or contrib_id not in self.included_ids:
                refinement_list = numpy.argwhere(vr.get_valid_refinements())
                # in order to speed up the vr processing, which would have narrow strips being processed
                # use a morton ordering on the tile indices so they are more closely processed in geolocation
                # and fewer loads/writes are requested of the db tiles (which are slow tiff read/writes)
                mort = morton.interleave2d_64(refinement_list.T)
                sorted_refinement_indices = refinement_list[numpy.lexsort([mort])]

                temp_path = tempfile.mkdtemp(dir=self.db.data_path)
                storage_db = self.db.make_accumulation_db(temp_path)
                x_accum, y_accum, depth_accum, uncertainty_accum, scores_accum, flags_accum = None, None, None, None, None, None
                max_len = 500000
                all_tiles = set()
                for iref, (ti, tj) in enumerate(sorted_refinement_indices):
                    # get an individual refinement and convert it to x,y from the row column system it was in.
                    refinement = vr.read_refinement(ti, tj)
                    # todo replace this with
                    r, c = numpy.indices(refinement.depth.shape)  # make indices into array elements that can be converted to x,y coordinates
                    pts = numpy.array([r, c, refinement.depth, refinement.uncertainty]).reshape(4, -1)
                    pts = pts[:, pts[2] != vr.fill_value]  # remove nodata points

                    x, y = affine_center(pts[0], pts[1], *refinement.geotransform)  # refinement_llx, resolution_x, 0, refinement_lly, 0, resolution_y)
                    ptsnew = refinement.get_xy_pts_arrays()
                    xnew, ynew = ptsnew[:2]
                    ptsnew = ptsnew[2:]
                    if not ((x==xnew).all() and (y==ynew).all() and (pts==ptsnew).all()):
                        raise Exception("mismatch")

                    if _debug:
                        print("debugging")
                        # inspect_x, inspect_y = 690134.03, 3333177.81  # ti, tj = (655, 265) in H13190
                        # if min(x) <  inspect_x and max(x) >inspect_x and min(y)< inspect_y and max(y) > inspect_y:
                        #     mdata = vr.varres_metadata[ti, tj]
                        #     resolution_x = mdata["resolution_x"]
                        #     resolution_y = mdata["resolution_y"]
                        #     sw_corner_x = mdata["sw_corner_x"]
                        #     sw_corner_y = mdata["sw_corner_y"]
                        #     bag_supergrid_dy = vr.cell_size_y
                        #     bag_llx = vr.minx - bag_supergrid_dx / 2.0  # @todo seems the llx is center of the supergridd cel?????
                        #     bag_lly = vr.miny - bag_supergrid_dy / 2.0
                        #     supergrid_x = tj * bag_supergrid_dx
                        #     supergrid_y = ti * bag_supergrid_dy
                        #     refinement_llx = bag_llx + supergrid_x + sw_corner_x - resolution_x / 2.0  # @TODO implies swcorner is to the center and not the exterior
                        #     refinement_lly = bag_lly + supergrid_y + sw_corner_y - resolution_y / 2.0
                        # else:
                        #     continue
                    if crs_transformer:
                        x, y = crs_transformer.transform(x, y)
                    depth = pts[2]
                    if reverse_z:
                        depth *= -1
                    uncertainty = pts[3]
                    scores = numpy.full(x.shape, survey_score)
                    flags = numpy.full(x.shape, flag)
                    # it's really slow to add each refinement to the db, so store up points until it's bigger and write at once
                    if x_accum is None:  # initialize arrays here to get the correct types
                        x_accum = numpy.zeros([max_len], dtype=x.dtype)
                        y_accum = numpy.zeros([max_len], dtype=y.dtype)
                        depth_accum = numpy.zeros([max_len], dtype=depth.dtype)
                        uncertainty_accum = numpy.zeros([max_len], dtype=uncertainty.dtype)
                        scores_accum = numpy.zeros([max_len], dtype=scores.dtype)
                        flags_accum = numpy.zeros([max_len], dtype=flags.dtype)
                        last_index = 0
                    # dump the accumulated arrays to the database if they are about to overflow the accumulation arrays
                    if last_index + len(x) > max_len:
                        tiles = self.insert_survey_array(numpy.array((x_accum[:last_index], y_accum[:last_index], depth_accum[:last_index],
                                                              uncertainty_accum[:last_index], scores_accum[:last_index], flags_accum[:last_index])),
                                                 vr.filename, accumulation_db=storage_db, contrib_id=contrib_id, compare_callback=compare_callback)
                        all_tiles.update(tiles)
                        last_index = 0
                    # append the new data to the end of the accumulation arrays
                    prev_index = last_index
                    last_index += len(x)
                    x_accum[prev_index:last_index] = x
                    y_accum[prev_index:last_index] = y
                    depth_accum[prev_index:last_index] = depth
                    uncertainty_accum[prev_index:last_index] = uncertainty
                    scores_accum[prev_index:last_index] = scores
                    flags_accum[prev_index:last_index] = flags

                if last_index > 0:
                    tiles = self.insert_survey_array(numpy.array((x_accum[:last_index], y_accum[:last_index], depth_accum[:last_index],
                                                          uncertainty_accum[:last_index], scores_accum[:last_index], flags_accum[:last_index])),
                                             vr.filename, accumulation_db=storage_db, contrib_id=contrib_id, compare_callback=compare_callback)
                    all_tiles.update(tiles)
                self.db.append_accumulation_db(storage_db)
                shutil.rmtree(storage_db.data_path, onerror=onerr)
                # @fixme make sure vr.name is correct for full path
                # fixme need to use a blocking locker on writing the metadata to the world_database
                self.finished_survey_insertion(vr.name, all_tiles, contrib_id)
            else:
                raise Exception(f"Survey Exists already in database {contrib_id}")

    def insert_survey_gdal(self, path_to_survey_data, survey_score=100, flag=0, override_epsg=NO_OVERRIDE, data_band=1, uncert_band=2,
                           contrib_id=numpy.nan, compare_callback=None, reverse_z=False):
        """ Insert a gdal readable dataset into the database.
        Currently works for BAG and probably geotiff.
        Parameters
        ----------
        path_to_survey_data
            full path to the gdal readable file
        survey_score
            score to use with the survey when combining into the database
        flag
            flag to apply when inserting the survey into the database

        Returns
        -------
        None

        """
        # @fixme rasterdata on disk is not storing the survey_ids?  Forgot to update/write metadata?

        # metadata = {**dataset.GetMetadata()}
        # pixel_is_area = True if 'AREA_OR_POINT' in metadata and metadata['AREA_OR_POINT'][:-2] == 'Area' else False

        ds = gdal.Open(str(path_to_survey_data))
        drv = ds.GetDriver()
        driver_name = drv.ShortName
        geotransform = ds.GetGeoTransform()
        if override_epsg == NO_OVERRIDE:
            epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
        else:
            epsg = override_epsg
        # fixme - do coordinate transforms correctly
        print("@todo - do transforms correctly with proj/vdatum etc")
        crs_transformer = get_crs_transformer(epsg, self.db.epsg)

        x1, y1 = affine_center(0, 0, *geotransform)
        x2, y2 = affine_center(ds.RasterYSize, ds.RasterXSize, *geotransform)
        if crs_transformer is not None:
            ((x1, y1), (x2, y2)) = crs_transformer.transform((x1, x2), (y1,y2))
        tile_list = self.db.get_tiles_indices(x1, y1, x2, y2)
        with AreaLock(tile_list, EXCLUSIVE|NON_BLOCKING, self.db.get_history_path_by_index) as lock:
            if contrib_id is None or contrib_id not in self.included_ids:
                temp_path = tempfile.mkdtemp(dir=self.db.data_path)
                storage_db = self.db.make_accumulation_db(temp_path)
                all_tiles = set()
                # read the data array in blocks
                # if geo_debug and False:
                #     col_block_size = col_size
                #     row_block_size = row_size

                # @fixme -- bands 1,2 means bag works but single band will fail
                for ic, ir, nodata, (data, uncert) in iterate_gdal_image(ds, (data_band, uncert_band)):
                        # if _debug:
                        #     if ic > 1:
                        #         break

                        # read the uncertainty as an array (if it exists)
                        r, c = numpy.indices(data.shape)  # make indices into array elements that can be converted to x,y coordinates
                        r += ir  # adjust the block r,c to the global raster r,c
                        c += ic
                        # pts = numpy.dstack([r, c, data, uncert]).reshape((-1, 4))
                        pts = numpy.array([r, c, data, uncert]).reshape(4, -1)
                        pts = pts[:, pts[2] != nodata]  # remove nodata points
                        # pts = pts[:, pts[2] > -18.2]  # reduce points to debug
                        if pts.size > 0:
                            # if driver_name == 'BAG':
                            x, y = affine_center(pts[0], pts[1], *geotransform)
                            # else:
                            #     x, y = affine(pts[0], pts[1], *geotransform)
                            if _debug:
                                pass
                                ## clip data to an area of given tolerance around a point
                                # px, py, tol = 400200, 3347921, 8
                                # pts = pts[:, x < px + tol]  # thing the r,c,depth, uncert array
                                # y = y[x < px + tol]  # then thin the y to match the new r,c,depth
                                # x = x[x < px + tol]  # finally thin the x to match y and r,c,depth arrays
                                # pts = pts[:, x > px - tol]
                                # y = y[x > px - tol]
                                # x = x[x > px - tol]
                                # pts = pts[:, y < py + tol]
                                # x = x[y < py + tol]
                                # y = y[y < py + tol]
                                # pts = pts[:, y > py - tol]
                                # x = x[y > py - tol]
                                # y = y[y > py - tol]
                                # if pts.size == 0:
                                #     continue
                            if geo_debug and False:
                                s_pts = numpy.array((x, y, pts[0], pts[1], pts[2]))
                                txs, tys = storage_db.tile_scheme.xy_to_tile_index(x, y)
                                isle_tx, isle_ty = 3532, 4141
                                isle_pts = s_pts[:, numpy.logical_and(txs == isle_tx, tys == isle_ty)]
                                if isle_pts.size > 0:
                                    pass
                            if crs_transformer:
                                x, y = crs_transformer.transform(x, y)
                            depth = pts[2]
                            if reverse_z:
                                depth *= -1
                            uncertainty = pts[3]
                            scores = numpy.full(x.shape, survey_score)
                            flags = numpy.full(x.shape, flag)
                            tiles = self.insert_survey_array(numpy.array((x, y, depth, uncertainty, scores, flags)), path_to_survey_data,
                                                             accumulation_db=storage_db, contrib_id=contrib_id, compare_callback=compare_callback)
                            all_tiles.update(tiles)
                self.db.append_accumulation_db(storage_db)
                shutil.rmtree(storage_db.data_path)
                # @fixme -- turn all_tiles into one consistent, unique list.  Is list of lists with duplicates right now
                # fixme need to use a blocking locker on writing the metadata to the world_database
                self.finished_survey_insertion(path_to_survey_data, all_tiles, contrib_id)
            else:
                raise Exception(f"Survey Exists already in database {contrib_id}")

    def export_area(self, fname, x1, y1, x2, y2, res, target_epsg=None, driver="GTiff",
                    layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
                    gdal_options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"), compare_callback=None):
        """ Retrieves an area from the database at the requested resolution.

        # 1) Create a single tif tile that covers the area desired
        # 2) Get the master db tile indices that the area overlaps and iterate them
        # 3) Read the single tif sub-area as an array that covers this tile being processed
        # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
        # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
        # 6) Sort on score in case multiple points go into a position that the right value is retained
        #      sort based on score then on depth so the shoalest top score is kept
        # 7) Use affine crs_transform convert x,y into the i,j for the exported area
        # 8) Write the data into the export (single) tif.
        #      replace x,y with row, col for the points

        Parameters
        ----------
        fname
            path to export to
        x1
            a corner x coordinate
        y1
            a corner y coordinate
        x2
            a corner x coordinate
        y2
            a corner y coordinate
        res
            Resolution to export with.  If a tuple is supplied it is read as (res_x, res_y) while a single number will be used for x and y resolutions
        target_epsg
            epsg of the coordinate system to export into
        driver
            gdal driver name to use
        layers
            Layers to extract from the database into the output file.  Defaults to Elevation, Uncertainty and Contributor

        Returns
        -------
        int, dataset
            number of database tiles that supplied data into the export area, open gdal dataset for the file location specified

        """
        # 1) Create a single tif tile that covers the area desired
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg
        crs_transform = get_crs_transformer(self.db.tile_scheme.epsg, target_epsg)
        inv_crs_transform = get_crs_transformer(target_epsg, self.db.tile_scheme.epsg)

        try:
            dx, dy = res
        except TypeError:
            dx = dy = res

        fname = pathlib.Path(fname)
        score_name = fname.with_suffix(".score" + fname.suffix)

        dataset = make_gdal_dataset_area(fname, len(layers), x1, y1, x2, y2, dx, dy, target_epsg, driver, gdal_options)
        # probably won't export the score layer but we need it when combining data into the export area

        dataset_score = make_gdal_dataset_area(score_name, 3, x1, y1, x2, y2, dx, dy, target_epsg, driver)

        affine_transform = dataset.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
        score_band = dataset_score.GetRasterBand(1)
        score_key2_band = dataset_score.GetRasterBand(2)
        max_cols, max_rows = score_band.XSize, score_band.YSize

        # 2) Get the master db tile indices that the area overlaps and iterate them
        if crs_transform:
            overview_x1, overview_y1, overview_x2, overview_y2 = transform_rect(x1, y1, x2, y2, inv_crs_transform.transform)
        else:
            overview_x1, overview_y1, overview_x2, overview_y2 = x1, y1, x2, y2
        tile_count = 0
        for txi, tyi, tile_history in self.db.iter_tiles(overview_x1, overview_y1, overview_x2, overview_y2):
            # if txi != 3325 or tyi != 3207:  # utm 16, US5MSYAF_utm, H13193 -- gaps in DB and exported enc cells  217849.73 (m), 3307249.86 (m)
            # if txi != 4614 or tyi != 3227:  # utm 15 h13190 -- area with res = 4.15m (larger than the 4m output)
            #     continue
            try:
                raster_data = tile_history[-1]
            except IndexError:  # empty tile, skip to the next
                continue
            # 3) Read the single tif sub-area as an array that covers this tile being processed
            tx1, ty1, tx2, ty2 = tile_history.get_corners()
            if crs_transform:
                target_x1, target_y1, target_x2, target_y2 = transform_rect(tx1, ty1, tx2, ty2, crs_transform.transform)
                target_xs, target_ys = numpy.array([target_x1, target_x2]), numpy.array([target_y1, target_y2])
            else:
                target_xs, target_ys = numpy.array([tx1, tx2]), numpy.array([ty1, ty2])
            r, c = inv_affine(target_xs, target_ys, *affine_transform)
            start_col, start_row = max(0, int(min(c))), max(0, int(min(r)))
            # the local r, c inside of the sub area
            # figure out how big the block is that we are operating on and if it would extend outside the array bounds
            block_cols, block_rows = int(numpy.abs(numpy.diff(c))) + 1, int(numpy.abs(numpy.diff(r))) + 1
            if block_cols + start_col > max_cols:
                block_cols = max_cols - start_col
            if block_rows + start_row > max_rows:
                block_rows = max_rows - start_row
            # data is at edge of target area and falls just outside edge
            if block_rows < 1 or block_cols < 1:
                continue
            # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
            tile_layers = raster_data.get_arrays(layers)
            if compare_callback is not None:
                all_data = raster_data.get_arrays()
                tile_scoring, _dup, reverse = compare_callback(all_data, all_data)
            else:  # default to score, depth with no reversals
                # @fixme - score and depth have same value, read bug?
                tile_score = raster_data.get_arrays(LayersEnum.SCORE)[0]
                tile_depth = raster_data.get_arrays(LayersEnum.ELEVATION)[0]
                tile_scoring = [tile_score, tile_depth]
                reverse = (False, False)
            # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
            self.merge_rasters(tile_layers, tile_scoring, raster_data,
                               crs_transform, affine_transform, start_col, start_row, block_cols, block_rows,
                               dataset, layers, dataset_score, reverse_sort=reverse)
            tile_count += 1
            # send the data to disk, I forget if this has any affect other than being able to look at the data in between steps to debug progress
            dataset.FlushCache()
            dataset_score.FlushCache()
        del score_key2_band, score_band, dataset_score
        try:
            os.remove(score_name)
        except PermissionError:
            gc.collect()
            try:
                os.remove(score_name)
            except PermissionError:
                print(f"Failed to remove {score_name}, permission denied (in use?)")
        return tile_count, dataset

    @staticmethod
    def merge_rasters(tile_layers, tile_scoring, raster_data,
                      crs_transform, affine_transform, start_col, start_row, block_cols, block_rows,
                      dataset, layers, dataset_score, reverse_sort=(False, False, False)):
        output_scores = []
        for i in range(len(tile_scoring)):
            score_band = dataset_score.GetRasterBand(i + 1)
            output_scores.append(score_band.ReadAsArray(start_col, start_row, block_cols, block_rows))

        sort_key_scores = numpy.array(output_scores)
        export_sub_area = dataset.ReadAsArray(start_col, start_row, block_cols, block_rows)
        # when only one output layer is made the ReadAsArray returns a shape like (x,y)
        # while a three layer output would return something like (3, nx, ny)
        # so we will reshape the (nx, ny) to (1, nx, ny) so the indexing works the same
        if len(export_sub_area.shape) < 3:
            export_sub_area = export_sub_area.reshape((1, *export_sub_area.shape))
        # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading

        tile_r, tile_c = numpy.indices(tile_layers.shape[1:])
        # treating the cells as areas means we want to export based on the center not the corner
        tile_x, tile_y = raster_data.rc_to_xy_using_dims(tile_layers[0].shape[0], tile_layers[0].shape[1], tile_r, tile_c, center=True)
        # if crs_transform:  # convert to target epsg
        #     tile_x, tile_y = crs_transform.transform(tile_x, tile_y)

        merge_arrays(tile_x, tile_y, tile_scoring, tile_layers,
                     export_sub_area, sort_key_scores, crs_transform, affine_transform,
                     start_col, start_row, block_cols, block_rows,
                     reverse_sort=reverse_sort)

        # @todo should export_sub_area be returned and let the caller write into their datasets?
        for band_num in range(len(layers)):
            band = dataset.GetRasterBand(band_num + 1)
            band.WriteArray(export_sub_area[band_num], start_col, start_row)
        for i in range(len(tile_scoring)):
            score_band = dataset_score.GetRasterBand(i+1)
            score_band.WriteArray(sort_key_scores[i], start_col, start_row)


    def extract_soundings(self):
        # this is the same as extract area except score = depth
        pass

    def soundings_from_caris_combined_csar(self):
        # Either fill a world database and then extract soundings or
        # generalize the inner loop of extract area then use it against csar data directly
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



class CustomBackend(WorldTilesBackend):
    def __init__(self, utm_epsg, res_x, res_y,  x1, y1, x2, y2, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = ExactTilingScheme(res_x, res_y, min_x=x1, min_y=y1, max_x=x2, max_y=y2, zoom=zoom_level, epsg=utm_epsg)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class CustomArea(WorldDatabase):
    def __init__(self, epsg, x1, y1, x2, y2, res_x, res_y, storage_directory):
        min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
        shape = max(shape_x, shape_y)
        tiles = shape/512  # this should result in tiles max sizes between 512 and 1024 pixels
        zoom = int(numpy.log2(tiles))
        if zoom < 0:
            zoom = 0
        super().__init__(CustomBackend(epsg, res_x, res_y, min_x, min_y, max_x, max_y, AccumulationHistory, DiskHistory, TiffStorage,
                                       storage_directory, zoom_level=zoom))

    def export(self, fname, driver="GTiff", layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
                    gdal_options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES")):
        """Export the full area of the 'single file database' in the epsg the data is stored in"""
        y1 = self.db.tile_scheme.min_y
        y2 = self.db.tile_scheme.max_y - self.res_y
        x1 = self.db.tile_scheme.min_x
        x2 = self.db.tile_scheme.max_x - self.res_x
        return super().export_area(fname, x1, y1, x2, y2, (self.res_x, self.res_y), driver=driver,
                    layers=layers, gdal_options=gdal_options)

    @property
    def res_x(self):
        return self.db.tile_scheme.res_x

    @property
    def res_y(self):
        return self.db.tile_scheme.res_y


if __name__ == "__main__":

    data_dir = pathlib.Path(r"G:\Data\NBS\H11305_for_Bruty")
    # orig_db = CustomArea(26916, 395813.2, 3350563.98, 406818.2, 3343878.98, 4, 4, data_dir.joinpath('bruty'))
    new_db = CustomArea(None, 395813.20000000007, 3350563.9800000004, 406818.20000000007, 3343878.9800000004, 4, 4, data_dir.joinpath('bruty_debug_center'))
    # use depth band for uncertainty since it's not in upsample data
    new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\1of3.tif", 0, uncert_band=1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\2of3.tif", 0, uncert_band=1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\3of3.tif", 0, uncert_band=1, override_epsg=None)
    new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_1of3.bag", 1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_2of3.bag", 1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_3of3.bag", 1, override_epsg=None)
    new_db.export(r"G:\Data\NBS\H11305_for_Bruty\combine_new_centers.tif")
    import sys
    sys.exit()

    fname = r"G:\Data\NBS\Speed_test\H11045_VB_4m_MLLW_2of2.bag"
    ds = gdal.Open(fname)
    x1, resx, dxy, y1, dyx, resy = ds.GetGeoTransform()
    numx = ds.RasterXSize
    numy = ds.RasterYSize
    epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
    epsg = 26918
    ds = None
    # db = WorldDatabase(UTMTileBackendExactRes(4, 4, epsg, RasterHistory, DiskHistory, TiffStorage,
    #                                     r"G:\Data\NBS\Speed_test\test_db_world"))
    db = CustomArea(epsg, x1, y1, x1+(numx+1)*resx, y1+(numy+1)*resy, 4, 4, r"G:\Data\NBS\Speed_test\test_cust4")
    db.insert_survey_gdal(fname, override_epsg=epsg)
    db.export(r"G:\Data\NBS\Speed_test\test_cust4\export.tif")
    raise Exception("Done")

    from nbs.bruty.history import MemoryHistory
    from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, LayersEnum, arrays_match
    from nbs.bruty.utils import save_soundings_from_image

    # from tests.test_data import master_data, data_dir

    # use_dir = data_dir.joinpath('tile4_vr_utm_db')
    # db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # db.export_area_old(use_dir.joinpath("export_tile_old.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)
    # db.export_area(use_dir.joinpath("export_tile_new.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)

    build_mississippi = True
    export_mississippi = False
    process_utm_15 = True
    output_res = (4, 4)  # desired output size in meters
    data_dir = pathlib.Path(r'G:\Data\NBS\Mississipi')
    if process_utm_15:
        export_dir = data_dir.joinpath("UTM15")
        epsg = 26915
        max_lon = -90
        min_lon = -96
        max_lat = 35
        min_lat = 0
        use_dir = data_dir.joinpath('vrbag_utm15_debug_db')

        data_files = [(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13194_MB_VR_LWRP.bag", 92),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13193_MB_VR_LWRP.bag", 100),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", 94),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13188_MB_VR_LWRP.bag", 95),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13189_MB_VR_LWRP.bag", 96),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag", 97),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13191_MB_VR_LWRP.bag", 98),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag", 99),
                      # (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag.resampled_4m.uncert.tif", 77),
                      # (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag.resampled_4m.uncert.tif", 79),
                   ]
        resamples = []
        # [r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag",
        #            r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag",]
        for vr_path in resamples:
            resampled_path = vr_path + ".resampled_4m.tif"
            bag.VRBag_to_TIF(vr_path, resampled_path, 4, use_blocks=False)
            resampled_with_uncertainty = resampled_path = resampled_path[:-4] + ".uncert.tif"
            add_uncertainty_layer(resampled_path, resampled_with_uncertainty)
            data_files.append(resampled_with_uncertainty)
    else:
        export_dir = data_dir.joinpath("UTM16")
        epsg = 26916
        max_lon = -84
        min_lon = -90
        max_lat = 35
        min_lat = 0
        use_dir = data_dir.joinpath('vrbag_utm16_debug_db')
        data_files = [(r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13195_MB_VR_LWRP.bag", 93),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13196_MB_VR_LWRP.bag", 91),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13193_MB_VR_LWRP.bag", 100),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13194_MB_VR_LWRP.bag", 92),
                   ]

    if build_mississippi:
        if os.path.exists(use_dir):
            shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackendExactRes(*output_res, epsg, RasterHistory, DiskHistory, TiffStorage,
                                      use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    if 0:  # find a specific point in the tiling database
        y, x = 30.120484, -91.030685
        px, py = crs_transform.transform(x, y)
        tile_index_x, tile_index_y = db.db.tile_scheme.xy_to_tile_index(px, py)

    if build_mississippi:

        for data_file, score in data_files:
            # bag_file = directory.joinpath(directory.name + "_MB_VR_LWRP.bag")
            if _debug:
                if 'H13190' not in data_file:
                    print("Skipped for debugging", data_file)
                    continue
            if 'H13194' in data_file:  # this file is encoded in UTM16 even in the UTM15 area
                override_epsg = 26916
            elif 'H13193' in data_file:  # this file is encoded in UTM15 even in the UTM16 area
                override_epsg = 26915
            else:
                override_epsg = epsg
            # db.insert_survey_gdal(bag_file, override_epsg=epsg)  # single res
            if str(data_file)[-4:] in (".bag", ):
                db.insert_survey_vr(data_file, survey_score=score, override_epsg=override_epsg)
            elif str(data_file)[-4:] in ("tiff", ".tif"):
                db.insert_survey_gdal(data_file, survey_score=score)

    if export_mississippi:
        area_shape_fname = r"G:\Data\NBS\Support_Files\MCD_Bands\Band5\Band5_V6.shp"
        ds = gdal.OpenEx(area_shape_fname)
        # ds.GetLayerCount()
        lyr = ds.GetLayer(0)
        srs = lyr.GetSpatialRef()
        export_epsg = rasterio.crs.CRS.from_string(srs.ExportToWkt()).to_epsg()
        lyr.GetFeatureCount()
        lyrdef = lyr.GetLayerDefn()
        for i in range(lyrdef.GetFieldCount()):
            flddef = lyrdef.GetFieldDefn(i)
            if flddef.name == "CellName":
                cell_field = i
                break
        crs_transform = get_crs_transformer(export_epsg, db.db.tile_scheme.epsg)
        inv_crs_transform = get_crs_transformer(db.db.tile_scheme.epsg, export_epsg)
        for feat in lyr:
            geom = feat.GetGeometryRef()
            # geom.GetGeometryCount()
            minx, maxx, miny, maxy = geom.GetEnvelope()  # (-164.7, -164.39999999999998, 67.725, 67.8)
            # output in WGS84
            cx = (minx + maxx) / 2.0
            cy = (miny + maxy) / 2.0
            # crop to the area around Mississippi
            if cx > min_lon and cx < max_lon and cy > min_lat and cy < max_lat:
                cell_name = feat.GetField(cell_field)
                if _debug:

                    ##
                    ## vertical stripes in lat/lon
                    ## "US5MSYAF" for example
                    # if cell_name not in ("US5MSYAF",):  # , 'US5MSYAD'
                    #     continue

                    ## @fixme  There is a resolution issue at ,
                    ## where the raw VR is at 4.2m which leaves stripes at 4m export so need to add
                    ## an upsampled dataset to fill the area (with lower score so it doesn't overwrite the VR itself)
                    if cell_name not in ('US5BPGBD',):  # 'US5BPGCD'):
                        continue

                    # @fixme  missing some data in US5PLQII, US5PLQMB  US5MSYAE -- more upsampling needed?

                print(cell_name)
                # convert user res (4m in testing) size at center of cell for resolution purposes
                dx, dy = compute_delta_coord(cx, cy, *output_res, crs_transform, inv_crs_transform)

                bag_options_dict = {'VAR_INDIVIDUAL_NAME': 'Chief, Hydrographic Surveys Division',
                                     'VAR_ORGANISATION_NAME': 'NOAA, NOS, Office of Coast Survey',
                                     'VAR_POSITION_NAME': 'Chief, Hydrographic Surveys Division',
                                     'VAR_DATE': datetime.now().strftime('%Y-%m-%d'),
                                     'VAR_VERT_WKT': 'VERT_CS["unknown", VERT_DATUM["unknown", 2000]]',
                                     'VAR_ABSTRACT': "This multi-layered file is part of NOAA Office of Coast Survey’s National Bathymetry. The National Bathymetric Source is created to serve chart production and support navigation. The bathymetry is compiled from multiple sources with varying quality and includes forms of interpolation. Soundings should not be extracted from this file as source data is not explicitly identified. The bathymetric vertical uncertainty is communicated through the associated layer. More generic quality and source metrics will be added with 2.0 version of the BAG format.",
                                     'VAR_PROCESS_STEP_DESCRIPTION': f'Generated By GDAL {gdal.__version__} and NBS',
                                     'VAR_DATETIME': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                                     'VAR_VERTICAL_UNCERT_CODE': 'productUncert',
                                     # 'VAR_RESTRICTION_CODE=' + restriction_code,
                                     # 'VAR_OTHER_CONSTRAINTS=' + other_constraints,
                                     # 'VAR_CLASSIFICATION=' + classification,
                                     #'VAR_SECURITY_USER_NOTE=' + security_user_note
                                     }
                tif_tags = {'EMAIL_ADDRESS': 'OCS.NBS@noaa.gov',
                            'ONLINE_RESOURCE': 'https://www.ngdc.noaa.gov',
                            'LICENSE': 'License cc0-1.0',
                            }

                export_path = export_dir.joinpath(cell_name + ".tif")
                cnt, exported_dataset = db.export_area(export_path, minx, miny, maxx, maxy, (dx+dx*.1, dy+dy*.1), target_epsg=export_epsg)

                # export_path = export_dir.joinpath(cell_name + ".bag")
                # bag_options = [key + "=" + val for key, val in bag_options_dict.items()]
                # cnt2, ex_ds = db.export_area(export_path, minx, miny, maxx, maxy, (dx+dx*.1, dy+dy*.1), target_epsg=export_epsg,
                #                       driver='BAG', gdal_options=bag_options)

                if cnt > 0:
                    # output in native UTM -- Since the coordinates "twist" we need to check all four corners,
                    # not just lower left and upper right
                    x1, y1, x2, y2 = transform_rect(minx, miny, maxx, maxy, crs_transform.transform)
                    cnt, utm_dataset = db.export_area(export_dir.joinpath(cell_name + "_utm.tif"),x1, y1, x2, y2, output_res)
                else:
                    exported_dataset = None  # close the gdal file
                    os.remove(export_path)
                os.remove(export_path.with_suffix(".score.tif"))

    test_soundings = False
    if test_soundings:
        soundings_files = [pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20210219_source.tiff"),
                           pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20201118_source.tiff"),
                           ]
        for soundings_file in soundings_files:
            ds = gdal.Open(str(soundings_file))
            # epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
            xform = ds.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
            d_val = ds.GetRasterBand(1)
            col_size = d_val.XSize
            row_size = d_val.YSize
            del d_val, ds
            x1, y1 = affine(0, 0, *xform)
            x2, y2 = affine(row_size, col_size, *xform)
            res = 50
            res_x = res
            res_y = res
            # move the minimum to an origin based on the resolution so future exports would match
            if x1 < x2:
                x1 -= x1 % res_x
            else:
                x2 -= x2 % res_x

            if y1 < y2:
                y1 -= y1 % res_y
            else:
                y2 -= y2 % res_y

            #  note: there is an issue where the database image and export image are written in reverse Y direction
            #  because of this the first position for one is top left and bottom left for the other.
            #  when converting the coordinate of the cell it basically ends up shifting by one
            #  image = (273250.0, 50.0, 0.0, 4586700.0, 0.0, -50.0)  db = (273250.0, 50, 0, 4552600.0, 0, 50)
            #  fixed by using cell centers rather than corners.
            #  Same problem could happen of course if the centers are the edges of the export tiff
            # db = CustomArea(26919, x1, y1, x2, y2, res_x, res_y, soundings_file.parent.joinpath('debug'))  # NAD823 zone 19.  WGS84 would be 32619
            # db.insert_survey_gdal(str(soundings_file))
            # db.export_area_new(str(soundings_file.parent.joinpath("output_soundings_debug5.tiff")), x1, y1, x2, y2, (res_x, res_y), )
            save_soundings_from_image(soundings_file, str(soundings_file) + "_3.gpkg", 50)


# test positions -- H13190, US5GPGBD, Mississipi\vrbag_utm15_full_db\4615\3227\_000001_.tif, Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP_resampled.tif
# same approx position
# 690134.03 (m), 3333177.81 (m)  is 41.7 in the H13190
# 690134.03 (m), 3333177.81 (m)  is 42.4 in the resampled
# 690133.98 (m), 3333178.01 (m)  is 42.3 in the \4615\3227\000001.tif
# 690133.60 (m), 3333177.74 (m)  is 42.3 in the US5GPGBD

# seems to be the same Z value of 41.7
# 690134.03 (m), 3333177.81 (m)  H13190
# 690138.14 (m), 3333177.79 (m)  resample  (right (east) one column)
# 690129.99 (m), 3333173.99 (m)  \4615\3227\000001.tif  (down+left (south west) one row+col)
# 690129.62 (m), 3333173.76 (m)  US5GPGBD  (down+left (south west) one row+col)

# from importlib import reload
# import HSTB.shared.gridded_coords
# bag.VRBag_to_TIF(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag", r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_resample.tif", 4.105774879455566, bag.MEAN, nodata=1000000.)

# index2d = numpy.array([(655, 265)], dtype=numpy.int32)

# >>> print('x', refinement_llx + 9 * resolution_x + resolution_x / 2.0, 'y',refinement_lly + 8 * resolution_y + resolution_y / 2.0)
# x 690134.0489868548 y 3333177.797961975
# >>> print("x (cols)",xstarts[9],":", xends[9], "y (rows)",ystarts[8],":", yends[8])
# x (cols) 690131.9960994151 : 690136.1018732946 y (rows) 3333175.7450745353 : 3333179.850848415
# >>> print("rows",row_start_indices[8],":",row_end_indices[8], "cols",col_start_indices[9],":", col_end_indices[9])
# rows 4052 : 4053 cols 2926 : 2927
# >>> print('starts',HSTB.shared.gridded_coords.affine(row_start_indices[8], col_start_indices[9], *ds_val.GetGeoTransform()), ',  ends',HSTB.shared.gridded_coords.affine(row_end_indices[8], col_end_indices[9], *ds_val.GetGeoTransform()))
# starts (690131.995748028, 3333183.9557557716) ,  ends (690136.1015229075, 3333179.849980892)
# >>> ds_val.GetGeoTransform(), sr_grid.geotransform
# ((678118.498450741,  4.105774879455566,  0.0,  3349820.5555673256,  0.0,  -4.105774879455566),
#  (678118.498450741,  4.105774879455566,  0,  3303552.578450741,  0,  4.105774879455566))

# ds = gdal.Open(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_resample4.tif")
# b = ds.GetRasterBand(1)
# dep = b.ReadAsArray()
# b.GetNoDataValue()
# (dep!=0.0).any()
