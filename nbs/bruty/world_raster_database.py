import os
import pathlib
import shutil
import tempfile
import time

import numpy

import rasterio.crs
from osgeo import gdal, osr, ogr

from HSTB.drivers import bag
from bruty.utils import merge_arrays, merge_array, get_geotransformer, onerr, tqdm, make_gdal_dataset_area, calc_area_array_params
from bruty.raster_data import LayersEnum, RasterData, affine, inv_affine, affine_center, arrays_dont_match
from bruty.history import RasterHistory, AccumulationHistory
from bruty.abstract import VABC, abstractmethod
from bruty.tile_calculations import TMSTilesMercator, GoogleTilesMercator, GoogleTilesLatLon, UTMTiles, LatLonTiles, TilingScheme

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
                            yield tx, ty, raster, tile_history.get_metadata()
                        except IndexError:
                            print("accumulation db made directory but not filled?", ty_dir.path)

    def append_accumulation_db(self, accumulation_db):
        # iterate the acculation_db and append the last rasters from that into this db
        for tx, ty, raster, meta in accumulation_db.iterate_filled_tiles():
            tile_history = self.get_tile_history_by_index(tx, ty)
            tile_history.append(raster)
            master_meta = tile_history.get_metadata()
            contr = master_meta.setdefault("contributors", {})
            contr.update(meta["contributors"])

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
        # if self.history_class is RasterHistory:
        use_history = AccumulationHistory
        new_db = WorldTilesBackend(self.tile_scheme, use_history, self.storage_class, self.data_class, data_path)
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


class SingleFileBackend(WorldTilesBackend):
    def __init__(self, epsg, x1, y1, x2, y2, history_class, storage_class, data_class, data_path):
        tile_scheme = TilingScheme(x1, y1, x2, y2, zoom=0)
        tile_scheme.epsg = epsg
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


# @todo - Implement locks
# class Lock:
#     def __init__(self, storage, idn):
#         pass
#     def aquire(self, timeout=0):
#         pass
#     def release(self):
#         pass
#     def is_active(self):
#         pass
#     def __del__(self):
#         self.release()
#     def notify(self):
#         pass
# class ReadLock(Lock):  # actively being read
#     pass
# class WriteLock(Lock):  # actively being modified
#     pass
# class PendingReadLock(WriteLock):  # something wants to read but there are active/pending writes
#     pass
# class PendingWriteLock(WriteLock):  # something wants to modify but there are active reads
#     pass
# class Storage(VABC):
#     """ Class to control the metadata associated with a WorldDatabaseBackend.
#     This would store things like read/write/pending_write locks and hashes etc per Tile id.
#     """
#     @abstractmethod
#     def readable_tile(self, idnum):
#         # lock the file for reading
#         pass
#
#     @abstractmethod
#     def writeable_tile(self, tile):
#         # request lock for writing
#         pass
#
#     def get_read_lock(self, idnum):
#         pass
#
#     def get_write_lock(self, idnum):
#         pass

# @todo Implement database of files, locks, processing states etc
# class PostgresStorage(Storage):
#     pass
# class SQLite(Storage):
#     pass
# class PickleStorage(Storage):  # this would only work for single instances
#     pass


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
        self.next_contributor = 0  # fixme: this is a temporary hack until we have a database of surveys with unique ids available

    def insert_txt_survey(self, path_to_survey_data, survey_score=100, flags=0, format=None, transformer=None):
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

        Returns
        -------
        None
        """
        # fixme for survey_data - use pandas?  structured arrays?

        # @todo Allow for pixel sizes, right now treating as point rather than say 8m or 4m coverages
        if not accumulation_db:
            accumulation_db = self.db
        # Compute the tile indices for each point
        contributor = numpy.full(input_survey_data[0].shape, self.next_contributor)
        survey_data = numpy.array((input_survey_data[0], input_survey_data[1], input_survey_data[2],
                                   input_survey_data[3], contributor, input_survey_data[4], input_survey_data[5]))
        # if a multidemensional array was passed in then reshape it to be one dimenion.
        # ex: the input_survey_data is 4x4 data with the 6 required data fields (x,y,z, uncertainty, score, flag),
        #   so 6 x 4 x 5 -- becomes a 6 x 20 array instead
        if len(survey_data.shape) > 2:
            survey_data = survey_data.reshape(survey_data.shape[0], -1)
        txs, tys = accumulation_db.tile_scheme.xy_to_tile_index(survey_data[0], survey_data[1])
        tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
        # itererate each tile that was found to have data
        # @todo figure out the contributor - should be the unique id from the database of surveys
        for i_tile, (tx, ty) in enumerate(tile_list):
            print(f'processing tile {i_tile} of {len(tile_list)}')
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
            pts = survey_data[:, numpy.logical_and(txs == tx, tys == ty)]

            # replace x,y with row, col for the points
            i, j = raster_data.xy_to_rc_using_dims(new_arrays.shape[1], new_arrays.shape[2], pts[0], pts[1])
            new_sort_values = numpy.array((new_arrays[LayersEnum.SCORE], new_arrays[LayersEnum.ELEVATION]))
            # negative depths, so don't reverse the sort of the second key (depth)
            merge_arrays(i, j, (pts[LayersEnum.SCORE + 2], pts[LayersEnum.ELEVATION + 2]), pts[2:], new_arrays, new_sort_values,
                         reverse_sort=(False, False))

            rd = RasterData.from_arrays(new_arrays)
            rd.set_metadata(raster_data.get_metadata())  # copy the metadata to the new raster
            rd.contributor = contrib_name
            tile_history.append(rd)
            meta = tile_history.get_metadata()
            meta.setdefault("contributors", {})[
                str(self.next_contributor)] = contrib_name  # JSON only allows strings as keys - first tried an integer key which gets weird
            tile_history.set_metadata(meta)

            # for x, y, depth, uncertainty, score, flag in sorted_pts:
            #     # fixme: score should be a lookup into the database so that data/decay is considered correctly
            #
            #     # @todo If scores are all computed then we could overwrite rapidly based on index,
            #     #   but has to be sorted by depth so the shallowest sounding in a cell is retained in case there were multiple
            #     #   We are not trying to implement CUBE or CHRT here, just pick a value and shoalest is safest
            #     # raster_data = raster_data[]

        self.next_contributor += 1

    def init_tile(self, tx, ty, tile_history):
        # @todo lookup the resolution to use by default.
        #   Probably will be a lookup based on the ENC cell the tile covers and then twice the resolution needed for that cell
        # @todo once resolution is determined then convert to the right size in the coordinate system to represent that sizing in meters

        # this should be ~2m when zoom 13 is the zoom level used (zoom 13 = 20m at 256 pix, so 8 times finer)
        # return rows, cols
        return 512, 512

    def insert_survey_vr(self, path_to_survey_data, survey_score=100, flag=0):
        """
        Parameters
        ----------
        path_to_survey_data
        survey_score
        flag

        Returns
        -------

        """
        vr = bag.VRBag(path_to_survey_data)
        refinement_list = numpy.argwhere(vr.get_valid_refinements())
        print("@todo - do transforms correctly with proj/vdatum etc")
        epsg = rasterio.crs.CRS.from_string(vr.srs.ExportToWkt()).to_epsg()
        georef_transformer = get_geotransformer(epsg, self.db.epsg)
        temp_path = tempfile.mkdtemp(dir=self.db.data_path)
        storage_db = self.db.make_accumulation_db(temp_path)

        for iref, (ti, tj) in enumerate(refinement_list):
            refinement = vr.read_refinement(ti, tj)
            r, c = numpy.indices(refinement.depth.shape)  # make indices into array elements that can be converted to x,y coordinates
            pts = numpy.array([r, c, refinement.depth, refinement.uncertainty]).reshape(4, -1)
            pts = pts[:, pts[2] != vr.fill_value]  # remove nodata points

            bag_supergrid_dx = vr.cell_size_x
            bag_supergrid_nx = vr.numx
            bag_supergrid_dy = vr.cell_size_y
            bag_supergrid_ny = vr.numy
            bag_llx = vr.minx - bag_supergrid_dx / 2.0  # @todo seems the llx is center of the supergridd cel?????
            bag_lly = vr.miny - bag_supergrid_dy / 2.0

            # index_start = vr.varres_metadata[ti, tj, "index"]
            # dimensions_x = vr.varres_metadata[ti, tj, "dimensions_x"]
            # dimensions_y = vr.varres_metadata[ti, tj, "dimensions_y"]
            resolution_x = vr.varres_metadata[ti, tj, "resolution_x"]
            resolution_y = vr.varres_metadata[ti, tj, "resolution_y"]
            sw_corner_x = vr.varres_metadata[ti, tj, "sw_corner_x"]
            sw_corner_y = vr.varres_metadata[ti, tj, "sw_corner_y"]

            supergrid_x = tj * bag_supergrid_dx
            supergrid_y = ti * bag_supergrid_dy
            refinement_llx = bag_llx + supergrid_x + sw_corner_x - resolution_x / 2.0  # @TODO implies swcorner is to the center and not the exterior
            refinement_lly = bag_lly + supergrid_y + sw_corner_y - resolution_y / 2.0

            x, y = affine(pts[0], pts[1], refinement_llx, resolution_x, 0, refinement_lly, 0, resolution_y)
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
        shutil.rmtree(storage_db.data_path, onerror=onerr)

    def insert_survey_gdal(self, path_to_survey_data, survey_score=100, flag=0):
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
        ds = gdal.Open(path_to_survey_data)
        x0, dxx, dyx, y0, dxy, dyy = ds.GetGeoTransform()
        epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
        # fixme - do coordinate transforms correctly
        print("@todo - do transforms correctly with proj/vdatum etc")
        georef_transformer = get_geotransformer(epsg, self.db.epsg)

        d_val = ds.GetRasterBand(1)  # elevation or depth band
        u_val = ds.GetRasterBand(2)  # uncertainty band
        block_sizes = d_val.GetBlockSize()
        row_block_size = min(max(block_sizes[1], 1024), 2048)
        col_block_size = min(max(block_sizes[0], 1024), 2048)
        col_size = d_val.XSize
        row_size = d_val.YSize
        nodata = d_val.GetNoDataValue()
        temp_path = tempfile.mkdtemp(dir=self.db.data_path)
        storage_db = self.db.make_accumulation_db(temp_path)
        # read the data array in blocks
        if geo_debug and False:
            col_block_size = col_size
            row_block_size = row_size
        for ic in tqdm(range(0, col_size, col_block_size), mininterval=.7):
            if ic + col_block_size < col_size:
                cols = col_block_size
            else:
                cols = col_size - ic
            for ir in tqdm(range(0, row_size, row_block_size), mininterval=.7):
                if ir + row_block_size < row_size:
                    rows = row_block_size
                else:
                    rows = row_size - ir
                # Get the depth data as an array
                uncert = u_val.ReadAsArray(ic, ir, cols, rows)
                data = d_val.ReadAsArray(ic, ir, cols, rows)  # depth or elevation data
                if geo_debug and False:  # draw an X in each block
                    diag = numpy.arange(min(data.shape), dtype=numpy.int32)
                    rdiag = diag[::-1]
                    data[diag, diag] = 55
                    data[rdiag, diag] = 66
                # read the uncertainty as an array (if it exists)
                r, c = numpy.indices(data.shape)  # make indices into array elements that can be converted to x,y coordinates
                r += ir  # adjust the block r,c to the global raster r,c
                c += ic
                # pts = numpy.dstack([r, c, data, uncert]).reshape((-1, 4))
                pts = numpy.array([r, c, data, uncert]).reshape(4, -1)
                pts = pts[:, pts[2] != nodata]  # remove nodata points
                # pts = pts[:, pts[2] > -18.2]  # reduce points to debug
                if pts.size > 0:
                    x, y = affine(pts[0], pts[1], x0, dxx, dyx, y0, dxy, dyy)
                    # x = x0 + pts[1] * dxx + pts[0] * dyx
                    # y = y0 + pts[1] * dxy + pts[0] * dyy
                    if geo_debug and False:
                        s_pts = numpy.array((x, y, pts[0], pts[1], pts[2]))
                        txs, tys = storage_db.tile_scheme.xy_to_tile_index(x, y)
                        isle_tx, isle_ty = 3532, 4141
                        isle_pts = s_pts[:, numpy.logical_and(txs == isle_tx, tys == isle_ty)]
                        if isle_pts.size > 0:
                            pass
                    if georef_transformer:
                        x, y = georef_transformer.transform(x, y)
                    depth = pts[2]
                    uncertainty = pts[3]
                    scores = numpy.full(x.shape, survey_score)
                    flags = numpy.full(x.shape, flag)
                    self.insert_survey_array(numpy.array((x, y, depth, uncertainty, scores, flags)), path_to_survey_data, accumulation_db=storage_db)
                    self.next_contributor -= 1  # undoing the next_contributor increment that happens in insert_survey_array since we call it multiple times and we'll increment it below
        self.next_contributor += 1
        self.db.append_accumulation_db(storage_db)
        shutil.rmtree(storage_db.data_path)

    def export_area_old(self, fname, x1, y1, x2, y2, res, target_epsg=None, driver="GTiff",
                        layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR)):
        """ Retrieves an area from the database at the requested resolution.

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
        None

        """
        # 1) Create a single tif tile that covers the area desired
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg
        geotransform = get_geotransformer(self.db.tile_scheme.epsg, target_epsg)

        try:
            dx, dy = res
        except TypeError:
            dx = dy = res

        dataset = make_gdal_dataset_area(fname, len(layers), x1, y1, x2, y2, dx, dy, target_epsg, driver)
        # probably won't export the score layer but we need it when combining data into the export area
        dataset_score = make_gdal_dataset_area(fname, 1, x1, y1, x2, y2, dx, dy, target_epsg, driver)

        x0, dxx, dyx, y0, dxy, dyy = dataset.GetGeoTransform()
        score_band = dataset_score.GetRasterBand(1)
        max_cols, max_rows = score_band.XSize, score_band.YSize

        # 2) Get the master db tile indices that the area overlaps and iterate them
        for txi, tyi, tile_history in self.db.iter_tiles(x1, y1, x2, y2):
            # if txi != 3504 or tyi != 4155:
            #     continue
            try:
                raster_data = tile_history[-1]
            except IndexError:  # empty tile, skip to the next
                continue
            # 3) Read the single tif sub-area as an array that covers this tile being processed
            tx1, ty1, tx2, ty2 = tile_history.get_corners()
            r, c = inv_affine(numpy.array([tx1, tx2]), numpy.array([ty1, ty2]), x0, dxx, dyx, y0, dxy, dyy)
            start_col, start_row = max(0, int(min(c))), max(0, int(min(r)))
            block_cols, block_rows = int(numpy.abs(numpy.diff(c))) + 1, int(numpy.abs(numpy.diff(r))) + 1
            if block_cols + start_col > max_cols:
                block_cols = max_cols - start_col
            if block_rows + start_row > max_rows:
                block_rows = max_rows - start_row

            export_sub_area_scores = dataset_score.ReadAsArray(start_col, start_row, block_cols, block_rows)
            export_sub_area = dataset.ReadAsArray(start_col, start_row, block_cols, block_rows)
            # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
            # fixme - score and depth have same value, read bug?
            tile_score = raster_data.get_arrays(LayersEnum.SCORE)[0]
            tile_depth = raster_data.get_arrays(LayersEnum.ELEVATION)[0]
            tile_layers = raster_data.get_arrays(layers)
            tile_r, tile_c = numpy.indices(tile_layers.shape[1:])
            tile_x, tile_y = raster_data.rc_to_xy_using_dims(tile_score.shape[0], tile_score.shape[1], tile_r, tile_c)
            if geotransform:  # convert to target epsg
                tile_x, tile_y = geotransform(tile_x, tile_y)

            # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
            # 6) Sort on score in case multiple points go into a position that the right value is retained
            # sort based on score then on depth so the shoalest top score is kept
            pts = numpy.array((tile_x, tile_y, tile_score, tile_depth, *tile_layers)).reshape(4 + len(layers), -1)
            pts = pts[:, ~numpy.isnan(pts[2])]  # remove empty cells (no score = empty)
            sorted_ind = numpy.lexsort((-pts[3], pts[2]))
            sorted_pts = pts[:, sorted_ind]
            # 7) Use affine geotransform convert x,y into the i,j for the exported area
            export_row, export_col = inv_affine(sorted_pts[0], sorted_pts[1], x0, dxx, dyx, y0, dxy, dyy)
            export_row -= start_row  # adjust to the sub area in memory
            export_col -= start_col
            # clip to the edges of the export area since our db tiles can cover the earth [0:block_rows-1, 0:block_cols]
            row_out_of_bounds = numpy.logical_or(export_row < 0, export_row >= block_rows)
            col_out_of_bounds = numpy.logical_or(export_col < 0, export_col >= block_cols)
            out_of_bounds = numpy.logical_or(row_out_of_bounds, col_out_of_bounds)
            if out_of_bounds.any():
                sorted_pts = sorted_pts[:, ~out_of_bounds]
                export_row = export_row[~out_of_bounds]
                export_col = export_col[~out_of_bounds]
            # 8) Write the data into the export (single) tif.
            # replace x,y with row, col for the points
            # @todo write unit test to confirm that the sort is working in case numpy changes behavior.
            #   currently assumes the last value is stored in the array if more than one have the same ri, rj indices.
            replace_cells = numpy.logical_or(sorted_pts[2] >= export_sub_area_scores[export_row, export_col],
                                             numpy.isnan(export_sub_area_scores[export_row, export_col]))
            replacements = sorted_pts[4:, replace_cells]
            ri = export_row[replace_cells]
            rj = export_col[replace_cells]
            export_sub_area[:, ri, rj] = replacements
            export_sub_area_scores[ri, rj] = sorted_pts[2, replace_cells]
            for band_num in range(len(layers)):
                band = dataset.GetRasterBand(band_num + 1)
                band.WriteArray(export_sub_area[band_num], start_col, start_row)
            score_band.WriteArray(export_sub_area_scores, start_col, start_row)
            dataset.FlushCache()
            dataset_score.FlushCache()

    def export_area(self, fname, x1, y1, x2, y2, res, target_epsg=None, driver="GTiff",
                    layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR)):
        """ Retrieves an area from the database at the requested resolution.

        # 1) Create a single tif tile that covers the area desired
        # 2) Get the master db tile indices that the area overlaps and iterate them
        # 3) Read the single tif sub-area as an array that covers this tile being processed
        # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
        # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
        # 6) Sort on score in case multiple points go into a position that the right value is retained
        #      sort based on score then on depth so the shoalest top score is kept
        # 7) Use affine geotransform convert x,y into the i,j for the exported area
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
        None

        """
        # 1) Create a single tif tile that covers the area desired
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg
        geotransform = get_geotransformer(self.db.tile_scheme.epsg, target_epsg)

        try:
            dx, dy = res
        except TypeError:
            dx = dy = res

        dataset = make_gdal_dataset_area(fname, len(layers), x1, y1, x2, y2, dx, dy, target_epsg, driver)
        # probably won't export the score layer but we need it when combining data into the export area
        dataset_score = make_gdal_dataset_area(fname, 1, x1, y1, x2, y2, dx, dy, target_epsg, driver)
        dataset_score_key2 = make_gdal_dataset_area(fname, 1, x1, y1, x2, y2, dx, dy, target_epsg, driver)

        affine_transform = dataset.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
        score_band = dataset_score.GetRasterBand(1)
        score_key2_band = dataset_score_key2.GetRasterBand(1)
        max_cols, max_rows = score_band.XSize, score_band.YSize

        # 2) Get the master db tile indices that the area overlaps and iterate them
        for txi, tyi, tile_history in self.db.iter_tiles(x1, y1, x2, y2):
            # if txi != 3504 or tyi != 4155:
            #     continue
            try:
                raster_data = tile_history[-1]
            except IndexError:  # empty tile, skip to the next
                continue
            # 3) Read the single tif sub-area as an array that covers this tile being processed
            tx1, ty1, tx2, ty2 = tile_history.get_corners()
            r, c = inv_affine(numpy.array([tx1, tx2]), numpy.array([ty1, ty2]), *affine_transform)
            start_col, start_row = max(0, int(min(c))), max(0, int(min(r)))
            # the local r, c inside of the sub area
            # figure out how big the block is that we are operating on and if it would extend outside the array bounds
            block_cols, block_rows = int(numpy.abs(numpy.diff(c))) + 1, int(numpy.abs(numpy.diff(r))) + 1
            if block_cols + start_col > max_cols:
                block_cols = max_cols - start_col
            if block_rows + start_row > max_rows:
                block_rows = max_rows - start_row

            # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
            # fixme - score and depth have same value, read bug?
            tile_score = raster_data.get_arrays(LayersEnum.SCORE)[0]
            tile_depth = raster_data.get_arrays(LayersEnum.ELEVATION)[0]
            tile_layers = raster_data.get_arrays(layers)
            # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
            self.merge_rasters(tile_layers, tile_score, raster_data, tile_depth,
                               geotransform, affine_transform, start_col, start_row, block_cols, block_rows,
                               dataset, dataset_score, dataset_score_key2, layers,
                               score_band, score_key2_band)
            # send the data to disk, I forget if this has any affect other than being able to look at the data in between steps to debug progress
            dataset.FlushCache()
            dataset_score.FlushCache()

    @staticmethod
    def merge_rasters(tile_layers, tile_score, raster_data, tile_depth,
                      geotransform, affine_transform, start_col, start_row, block_cols, block_rows,
                      dataset, dataset_score, dataset_scorekey2, layers,
                      score_band, key2_band, reverse_sort=(False, False)):
        export_sub_area_scores = dataset_score.ReadAsArray(start_col, start_row, block_cols, block_rows)
        export_sub_area_key2 = dataset_scorekey2.ReadAsArray(start_col, start_row, block_cols, block_rows)
        sort_key_scores = numpy.array((export_sub_area_scores, export_sub_area_key2))
        export_sub_area = dataset.ReadAsArray(start_col, start_row, block_cols, block_rows)

        # 5) @todo make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading

        tile_r, tile_c = numpy.indices(tile_layers.shape[1:])
        # treating the cells as areas means we want to export based on the center not the corner
        tile_x, tile_y = raster_data.rc_to_xy_using_dims(tile_score.shape[0], tile_score.shape[1], tile_r, tile_c, center=True)
        if geotransform:  # convert to target epsg
            tile_x, tile_y = geotransform(tile_x, tile_y)

        merge_arrays(tile_x, tile_y, (tile_score, tile_depth), tile_layers,
                     export_sub_area, sort_key_scores, affine_transform,
                     start_col, start_row, block_cols, block_rows,
                     reverse_sort=reverse_sort)

        # @todo should export_sub_area be returned and let the caller write into their datasets?
        for band_num in range(len(layers)):
            band = dataset.GetRasterBand(band_num + 1)
            band.WriteArray(export_sub_area[band_num], start_col, start_row)
        score_band.WriteArray(sort_key_scores[0], start_col, start_row)
        key2_band.WriteArray(sort_key_scores[1], start_col, start_row)

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


def make_gdal_dataset_size(fname, bands, min_x, max_y, res_x, res_y, shape_x, shape_y, epsg, driver="GTiff"):
    """ Makes a north up gdal dataset with nodata = numpy.nan and LZW compression.
    Specifying a positive res_y will be input as a negative value into the gdal file,
    since tif/gdal likes max_y and a negative Y pixel size.
    i.e. the geotransform in gdal will be stored as [min_x, res_x, 0, max_y, 0, -res_y]
    Parameters
    ----------
    fname
        filename to create
    bands
        list of names of bands in the file
    min_x
        minimum X coordinate
    max_y
        maximum Y coordinate (because tiff images like to specify max Y and a negative res_y)
    res_x
        pixel size in x direction
    res_y
        pixel size in y direction
    shape_x
        number of pixels in X direction (columns)
    shape_y
        number of pixels in Y directions (rows)
    epsg
        epsg of the target coordinate system
    driver
        gdal driver name of the output file (defaults to geotiff)

    Returns
    -------
    gdal.dataset

    """
    driver = gdal.GetDriverByName(driver)
    dataset = driver.Create(str(fname), xsize=shape_x, ysize=shape_y, bands=bands, eType=gdal.GDT_Float32,
                            options=['COMPRESS=LZW'])

    # Set location
    gt = [min_x, res_x, 0, max_y, 0, -res_y]  # north up
    dataset.SetGeoTransform(gt)

    # Get raster projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dest_wkt = srs.ExportToWkt()

    # Set projection
    dataset.SetProjection(dest_wkt)
    band = dataset.GetRasterBand(1)
    band.SetNoDataValue(numpy.nan)
    del band
    return dataset



class SingleFile(WorldDatabase):
    def __init__(self, epsg, x1, y1, x2, y2, res_x, res_y, storage_directory):
        min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
        super().__init__(SingleFileBackend(epsg, min_x, min_y, max_x, max_y, AccumulationHistory, DiskHistory, TiffStorage, storage_directory))

    def init_tile(self, tx, ty, tile_history):
        return self.shape_y, self.shape_x  # rows and columns



if __name__ == "__main__":
    from bruty.history import DiskHistory, MemoryHistory, RasterHistory
    from bruty.raster_data import MemoryStorage, RasterDelta, RasterData, TiffStorage, LayersEnum, arrays_match
    from bruty.utils import save_soundings_from_image

    # from tests.test_data import master_data, data_dir

    # use_dir = data_dir.joinpath('tile4_vr_utm_db')
    # db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # db.export_area_old(use_dir.joinpath("export_tile_old.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)
    # db.export_area(use_dir.joinpath("export_tile_new.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)

    soundings_files = [pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20210219_source.tiff"),
                       pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20201118_source.tiff"), ]

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
        # db = SingleFile(26919, x1, y1, x2, y2, res_x, res_y, soundings_file.parent.joinpath('debug'))  # NAD823 zone 19.  WGS84 would be 32619
        # db.insert_survey_gdal(str(soundings_file))
        # db.export_area_new(str(soundings_file.parent.joinpath("output_soundings_debug5.tiff")), x1, y1, x2, y2, (res_x, res_y), )
        save_soundings_from_image(soundings_file, str(soundings_file) + "_3.gpkg", 50)
