import os
import pathlib
import shutil
import tempfile

import numpy

from pyproj import Transformer, CRS
import rasterio.crs
from osgeo import gdal, osr

from HSTB.drivers import bag
from xipe_dev.xipe2.raster_data import LayersEnum, RasterData
from xipe_dev.xipe2.abstract import VABC, abstractmethod
from xipe_dev.xipe2.tile_calculations import TMSTilesMercator, GoogleTilesMercator, GoogleTilesLatLon, UTMTiles, LatLonTiles

geo_debug = False

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
        self.tile_scheme = tile_scheme
        self.data_path = pathlib.Path(data_path)
        self.data_class = data_class
        self.history_class = history_class
        self.storage_class = storage_class
        if data_path:
            os.makedirs(self.data_path, exist_ok=True)
    def iterate_filled_tiles(self):
        for tx_dir in os.scandir(self.data_path):
            if tx_dir.is_dir():
                for ty_dir in os.scandir(tx_dir):
                    if ty_dir.is_dir():
                        tx, ty = self.str_to_tile_index(tx_dir.name, ty_dir.name)
                        tile_history = self.get_tile_history_by_index(tx, ty)
                        try:
                            raster = tile_history[-1]
                            yield tx, ty, raster
                        except IndexError:
                            print("accumulation db made directory but not filled?", ty_dir.path)

    def append_accumulation_db(self, accumulation_db):
        # iterate the acculation_db and append the last rasters from that into this db
        for tx, ty, raster in accumulation_db.iterate_filled_tiles():
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
        new_db = WorldTilesBackend(self.tile_scheme, self.history_class, self.storage_class, self.data_class, data_path)
        return new_db

    @property
    def __version__(self) -> int:
        return 1

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

    def get_tile_history_by_index(self, tx, ty):
        tx_str, ty_str = self.tile_index_to_str(tx, ty)
        history = self.history_class(self.storage_class(self.data_class, self.data_path.joinpath(tx_str).joinpath(ty_str)))
        lx, ly, ux, uy = self.tile_scheme.tile_index_to_xy(tx, ty)
        history.set_corners(lx, ly, ux, uy)
        history.set_epsg(self.epsg)
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
        tile_scheme.epsg = utm_epsg
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

class WorldDatabase(VABC):
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
        self.db = backend
        self.next_contributor = 0  # fixme: this is a temporary hack until we have a database of surveys with unique ids available
    def insert_txt_survey(self, path_to_survey_data, survey_score=100, flags=0, format=None, transformer=None):
        if not format:
            format = [('y', 'f8'), ('x', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')]
        data = numpy.loadtxt(path_to_survey_data, dtype=format)
        x = data['x']
        y = data['y']
        if transformer:
            x, y = transformer.transform(x, y)
        depth = data['depth']
        uncertainty = data['uncertainty']
        score = numpy.full(x.shape, survey_score)
        flags = numpy.full(x.shape, flags)
        self.insert_survey_array(numpy.array((x, y, depth, uncertainty, score, flags)), path_to_survey_data)

    def insert_survey_array(self, input_survey_data, contrib_name, accumulation_db=None):
        # fixme for survey_data - use pandas?  structured arrays?

        # @todo Allow for pixel sizes, right now treating as point rather than say 8m or 4m coverages
        if not accumulation_db:
            accumulation_db = self.db
        # Compute the tile indices for each point
        contributor = numpy.full(input_survey_data[0].shape, self.next_contributor)
        survey_data = numpy.array((input_survey_data[0], input_survey_data[1], input_survey_data[2],
                                   input_survey_data[3], contributor, input_survey_data[4], input_survey_data[5]))
        txs, tys = accumulation_db.tile_scheme.xy_to_tile_index(survey_data[0], survey_data[1])
        tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
        # itererate each tile that was found to have data
        # @todo figure out the contributor - should be the unique id from the database of surveys
        for i_tile, (tx, ty) in enumerate(tile_list):
            print(f'processing tile {i_tile} of {len(tile_list)}')
            pts = survey_data[:, numpy.logical_and(txs == tx, tys == ty)]
            sorted_pts = pts[:, numpy.argsort(pts[2])[::-1]]  # sort from deepest to shoalest
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
                    res_x, res_y = self.init_tile(tx, ty)
                    raster_data = tile_history.make_empty_data(res_x, res_y)
                    make_outlines = raster_data.get_arrays()
                    if True or geo_debug:  # draw a box around the tile and put a notch in the 0,0 corner
                        make_outlines[:, :, 0] = 99  # outline around box
                        make_outlines[:, :, -1] = 99
                        make_outlines[:, 0, :] = 99
                        make_outlines[:, -1, :] = 99
                        make_outlines[:, 0:15, 0:15] = 44  # registration notch at 0,0
                    raster_data.set_arrays(make_outlines)
            new_arrays = raster_data.get_arrays()
            # replace x,y with row, col for the points
            i, j = raster_data.xy_to_rc_using_dims(new_arrays.shape[1], new_arrays.shape[2], sorted_pts[0], sorted_pts[1])
            replace_cells = numpy.logical_or(sorted_pts[4] >= new_arrays[LayersEnum.SCORE, i, j], numpy.isnan(new_arrays[LayersEnum.SCORE, i, j]))
            replacements = sorted_pts[:, replace_cells]
            ri = i[replace_cells]
            rj = j[replace_cells]
            new_arrays[:, ri, rj] = replacements[2:]
            # new_arrays[LayersEnum., ri, rj] = replacements[2]
            # new_arrays[2, ri, rj] = replacements[2]
            # new_arrays[3, ri, rj] = replacements[2]
            # new_arrays[4, ri, rj] = replacements[2]

            rd = RasterData.from_arrays(new_arrays)
            rd.set_metadata(raster_data.get_metadata())  # copy the metadata to the new raster
            rd.contributor = contrib_name
            tile_history.append(rd)

            # for x, y, depth, uncertainty, score, flag in sorted_pts:
            #     # fixme: score should be a lookup into the database so that data/decay is considered correctly
            #
            #     # @todo If scores are all computed then we could overwrite rapidly based on index,
            #     #   but has to be sorted by depth so the shallowest sounding in a cell is retained in case there were multiple
            #     #   We are not trying to implement CUBE or CHRT here, just pick a value and shoalest is safest
            #     # raster_data = raster_data[]


        self.next_contributor += 1
    def init_tile(self, tx, ty):
        # @todo lookup the resolution to use by default.
        #   Probably will be a lookup based on the ENC cell the tile covers and then twice the resolution needed for that cell
        # @todo once resolution is determined then convert to the right size in the coordinate system to represent that sizing in meters

        # this should be ~2m when zoom 13 is the zoom level used (zoom 13 = 20m at 256 pix, so 8 times finer)
        return 512, 512

    def insert_survey_gdal(self, path_to_survey_data, survey_score=100, flag=0):
        ds = gdal.Open(path_to_survey_data)
        x0, dxx, dyx, y0, dxy, dyy = ds.GetGeoTransform()
        epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
        # fixme - do coordinate transforms correctly
        print("@todo - do transforms correctly wirht proj/vdatum etc")
        if epsg != self.db.epsg:
            input_crs = CRS.from_epsg(epsg)
            output_crs = CRS.from_epsg(self.db.utm_epsg)
            georef_transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        else:
            georef_transformer = None

        # @todo test block read and coordinate transforms
        d_val = ds.GetRasterBand(1)  # elevation or depth band
        u_val = ds.GetRasterBand(2)  # uncertainty band
        block_sizes = d_val.GetBlockSize()
        row_block_size = min(max(block_sizes[1], 512), 1024)
        col_block_size = min(max(block_sizes[0], 512), 1024)
        col_size = d_val.XSize
        row_size = d_val.YSize
        nodata = d_val.GetNoDataValue()
        temp_path = tempfile.mkdtemp(dir=self.db.data_path)
        storage_db = self.db.make_accumulation_db(temp_path)
        # read the data array in blocks
        # for ic in tqdm(range(0, col_size, col_block_size), mininterval=.7):
        if False:
            col_block_size = col_size
            row_block_size = row_size
        for ic in range(0, col_size, col_block_size):
            if ic + col_block_size < col_size:
                cols = col_block_size
            else:
                cols = col_size - ic
            for ir in range(0, row_size, row_block_size):
                if ir + row_block_size < row_size:
                    rows = row_block_size
                else:
                    rows = row_size - ir
                # Get the depth data as an array
                uncert = u_val.ReadAsArray(ic, ir, cols, rows)
                data = d_val.ReadAsArray(ic, ir, cols, rows)  # depth or elevation data
                if geo_debug:  # draw an X in each block
                    diag = numpy.arange(min(data.shape), dtype=numpy.int32)
                    rdiag = diag[::-1]
                    data[diag, diag] = 55
                    data[rdiag, diag] = 66
                # read the uncertainty as an array (if it exists)
                r, c = numpy.indices(data.shape)  # make indices into array elements that can be converted to x,y coordinates
                r += ir  # adjust the block r,c to the global raster r,c
                c += ic
                # pts = numpy.dstack([r, c, data, uncert]).reshape((-1, 4))
                pts = numpy.array([r, c, data, uncert]).reshape(4,-1)
                pts = pts[:, pts[2] != nodata]  # remove nodata points
                # pts = pts[:, pts[2] > -18.2]  # reduce points to debug
                if pts.size > 0:
                    x = x0 + pts[1] * dxx + pts[0] * dyx
                    y = y0 + pts[1] * dxy + pts[0] * dyy
                    if False:
                        s_pts = numpy.array((x, y, pts[0], pts[1], pts[2]))
                        txs, tys = storage_db.tile_scheme.xy_to_tile_index(x, y)
                        isle_tx, isle_ty = 3532, 4141
                        isle_pts = s_pts[:, numpy.logical_and(txs==isle_tx, tys==isle_ty)]
                        if isle_pts.size >0:
                            pass
                    if georef_transformer:
                        x, y = georef_transformer.transform(x, y)
                    depth = pts[2]
                    uncertainty = pts[3]
                    scores = numpy.full(x.shape, survey_score)
                    flags = numpy.full(x.shape, flag)
                    self.insert_survey_array(numpy.array((x, y, depth, uncertainty, scores, flags)), path_to_survey_data, accumulation_db=storage_db)
                    self.next_contributor -= 1
        self.next_contributor += 1
        self.db.append_accumulation_db(storage_db)
        shutil.rmtree(storage_db.data_path)


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


