import os
import pathlib
from dataclasses import dataclass
import logging
import re
import time

import numpy
import psycopg2
from osgeo import ogr
from shapely import wkt, wkb

from nbs.bruty.nbs_postgres import get_tablenames, show_last_ids, connection_with_retries, ConnectionInfo, pg_update
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
from nbs.bruty.raster_data import TiffStorage, LayersEnum
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.configs import get_logger, run_configs, parse_ints_with_ranges, parse_multiple_values, iter_configs, set_stream_logging, log_config, make_family_of_logs
from nbs.bruty.nbs_postgres import NOT_NAV, get_nbs_records, get_sorting_info, get_transform_metadata, \
    connect_params_from_config, hash_id,REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE, BaseLockException
from nbs.bruty.utils import ConsoleProcessTracker

NO_DATA = -1
SUCCEEDED = 0
DATA_ERRORS = 3
TILE_LOCKED = 4
FAILED_VALIDATION = 5
SQLITE_READ_FAILURE = 6
UNHANDLED_EXCEPTION = 99

_debug = False
ogr.UseExceptions()
LOGGER = get_logger('nbs.bruty.create_tiles')

class BrutyOperation:
    OPERATION = ''
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    EXIT_CODE = 'code'
    WARNINGS_LOG = 'warnings_log'
    INFO_LOG = 'info_log'
    REQUEST_TIME = 'request_time'
    TRIES = 'tries'
    DATA_LOCATION = 'data_location'

    def __init__(self, record=None):
        if record:
            self.read(record)

    def read(self, record):
        self.start_time = record.get(self.START_TIME, None)
        self.end_time = record.get(self.END_TIME, None)
        self.exit_code = record.get(self.EXIT_CODE, None)
        self.tries = record.get(self.TRIES, None)
        self.warnings_log = record.get(self.WARNINGS_LOG, None)
        self.info_log = record.get(self.INFO_LOG, None)
        self.data_location = record.get(self.DATA_LOCATION, None)


class CombineOperation(BrutyOperation):
    OPERATION = 'combine'
    # I wanted these to be properties but needed access from just a class name
    # and I wanted it to look like a string not a method so am defining in both classes (without the complexity of a class factory)
    START_TIME = OPERATION + "_" + BrutyOperation.START_TIME
    END_TIME = OPERATION + "_" + BrutyOperation.END_TIME
    EXIT_CODE = OPERATION + "_" + BrutyOperation.EXIT_CODE
    WARNINGS_LOG = OPERATION + "_" + BrutyOperation.WARNINGS_LOG
    INFO_LOG = OPERATION + "_" + BrutyOperation.INFO_LOG
    REQUEST_TIME = OPERATION + "_" + BrutyOperation.REQUEST_TIME
    TRIES = OPERATION + "_" + BrutyOperation.TRIES
    DATA_LOCATION = OPERATION + "_" + BrutyOperation.DATA_LOCATION

    def __init__(self, record=None):
        super().__init__(record)

class ExportOperation(BrutyOperation):
    OPERATION = 'export'
    START_TIME = OPERATION + "_" + BrutyOperation.START_TIME
    END_TIME = OPERATION + "_" + BrutyOperation.END_TIME
    EXIT_CODE = OPERATION + "_" + BrutyOperation.EXIT_CODE
    WARNINGS_LOG = OPERATION + "_" + BrutyOperation.WARNINGS_LOG
    INFO_LOG = OPERATION + "_" + BrutyOperation.INFO_LOG
    REQUEST_TIME = OPERATION + "_" + BrutyOperation.REQUEST_TIME
    TRIES = OPERATION + "_" + BrutyOperation.TRIES
    DATA_LOCATION = OPERATION + "_" + BrutyOperation.DATA_LOCATION

    def __init__(self, record=None):
        super().__init__(record)


class TileInfo:
    SOURCE_DATABASE = "tile_specifications"
    SOURCE_TABLE = "spec_tiles"
    JOINED_TABLE = "spec_tiles"

    PB = 'production_branch'
    UTM = 'utm'
    TILE = 'tile'
    DATUM = 'datum'
    LOCALITY = 'locality'
    HEMISPHERE = 'hemisphere'
    PRIORITY = 'priority'
    EXPORT_PUBLIC = 'combine_public'
    EXPORT_INTERNAL = 'combine_internal'
    EXPORT_NAVIGATION = 'combine_navigation'
    BUILD = 'build'
    TILE_ID = 't_id'
    IS_LOCKED = 'is_locked'
    PRIMARY_KEY = TILE_ID

    def __init__(self, **review_tile):
        # @TODO if this becomes used more, make this into a static method called "def from_combine_spec"
        #   and make a different init not based on the combine_spec table layout
        self.conn = None
        self.cursor = None
        self.read_record(**review_tile)

    def read_record(self, **review_tile):
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None
        self.hemi = review_tile[self.HEMISPHERE].lower()
        self.pb = review_tile[self.PB]
        self.utm = review_tile[self.UTM]
        self.tile = review_tile[self.TILE]
        self.datum = review_tile[self.DATUM]
        self.locality = review_tile[self.LOCALITY]
        # if NULL, make it 0 -- using the ternary operator lets us make Null be -1 without changing the value of 0
        self.priority = review_tile[self.PRIORITY] if review_tile.get(self.PRIORITY, None) is not None else 0
        self.build = review_tile.get(self.BUILD, None)
        self.public = review_tile.get(self.EXPORT_PUBLIC, None)  # public - included "unqualified"
        self.internal = review_tile.get(self.EXPORT_INTERNAL, None)  # includes sensitive
        self.navigation = review_tile.get(self.EXPORT_NAVIGATION, None)  # only QC'd (qualified) data
        self.is_locked = review_tile.get(self.IS_LOCKED, None)
        self.pk = review_tile[self.PRIMARY_KEY]

    def is_running(self):
        return self.is_locked

    def __repr__(self):
        return f"TileInfo:{self.pb}_{self.utm}{self.hemi}_{self.tile}_{self.locality}"

    def hash_id(self):
        """ Returns a hash of the tile info so it can be used as a dictionary key"""
        return hash_id(self.pb, self.utm, self.hemi, self.tile, self.datum)

    @property
    def base_name(self):
        return "_".join([self.pb, self.locality, f"utm{self.utm}{self.hemi}", self.datum])

    def acquire_lock(self, conn_info):
        """ Acquire a lock on the record in the database.
        If successful, it will return the connection and cursor and update the record with the full record.
        If not, it will raise a BaseLockException.
        """
        conn_copy = ConnectionInfo(CombineTileInfo.SOURCE_DATABASE, conn_info.username, conn_info.password, conn_info.hostname, conn_info.port,
                                    [CombineTileInfo.SOURCE_TABLE])
        self.conn, self.cursor = connection_with_retries(conn_copy, autocommit=False)
        self.cursor.execute(f"""SELECT * FROM {self.SOURCE_TABLE} WHERE {self.PRIMARY_KEY}=%s FOR UPDATE SKIP LOCKED""", (self.pk,))
        record = self.cursor.fetchone()
        if record is None:
            self.release_lock()
            full_info = None
            raise BaseLockException(f"Failed to acquire lock for {self} where PK={self.pk}")
        else:
            self.cursor.execute(f"""SELECT * FROM {self.JOINED_TABLE} WHERE {self.PRIMARY_KEY}=%s""", (self.pk,))
            record = self.cursor.fetchone()
            # udpate to the full record in case it was a partial record
            self.read_record(**record)
        return self.conn, self.cursor

    def release_lock(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None
        self.cursor = None


class ResolutionTileInfo(TileInfo):
    SOURCE_TABLE = "spec_resolutions"
    RESOLUTION = "resolution"
    CLOSING_DISTANCE = 'closing_distance'
    RESOLUTION_ID = 'r_id'
    TILE_ID = 'tile_id'
    PRIMARY_KEY = RESOLUTION_ID
    JOINED_TABLE = f"{SOURCE_TABLE} R JOIN {TileInfo.SOURCE_TABLE} TI ON (R.{TILE_ID}=TI.{TileInfo.PRIMARY_KEY})"

    def __init__(self, **review_tile):
        # @TODO if this becomes used more, make this into a static method called "def from_combine_spec"
        #   and make a different init not based on the combine_spec table layout
        super().__init__(**review_tile)

    def read_record(self, **review_tile):
        super().read_record(**review_tile)
        self.resolution = review_tile[self.RESOLUTION]
        self.closing_dist = review_tile.get(self.CLOSING_DISTANCE, None)
        self.res_tile_id = review_tile.get(self.RESOLUTION_ID, None)
        self.export = ExportOperation(review_tile)
        self.epsg = review_tile.get('st_srid', None)
        try:
            self.geometry = review_tile['geometry'] if 'geometry_buffered' not in review_tile else review_tile['geometry_buffered']
        except KeyError:
            self.geometry = None

    def __repr__(self):
        return super().__repr__()+f"_{self.resolution}m"

    def hash_id(self):
        return super().hash_id(), self.resolution  # hash_id(self.resolution)

    def acquire_lock_and_combine_locks(self, conn_info):
        # lock the current record in the spec_resolutions table and set up a connection+cursor
        self.acquire_lock(conn_info)
        # lock all the spec_combines that this resolution would affect
        self.cursor.execute(f"""SELECT * FROM {CombineTileInfo.SOURCE_TABLE} WHERE {CombineTileInfo.RESOLUTION_ID}=%s FOR UPDATE SKIP LOCKED""", (self.pk,))
        record = self.cursor.fetchone()
        if record is None:
            self.release_lock()
            full_info = None
            raise BaseLockException(f"Failed to acquire lock for {self} where PK={self.pk}")
        return self.conn, self.cursor



    @property
    def geometry(self):
        return self._geom

    @geometry.setter
    def geometry(self, geom):
        if geom is None:
            self._geom = None
            self.minx, self.miny = None, None
            self.maxx, self.maxy = None, None
        else:
            try:
                minx, maxx, miny, maxy = geom.GetEnvelope()
                g = geom
            except AttributeError:
                try:
                    minx, miny, maxx, maxy = geom.bounds
                    g = geom
                except:
                    try:
                        g = ogr.CreateGeometryFromWkb(bytes.fromhex(geom))
                        minx, maxx, miny, maxy = g.GetEnvelope()
                        # Out[44]: (-74.4, -73.725, 40.2, 40.5)

                        # g.GetGeometryRef(0).GetPoints()
                        # Out[48]:
                        # [(-73.95, 40.2), (-74.1, 40.2), (-74.1, 40.425),  (-74.4, 40.425), (-74.4, 40.5),
                        #  (-73.725, 40.5), (-73.725, 40.35), (-73.95, 40.35), (-73.95, 40.2)]
                    except (RuntimeError, AttributeError):
                        # *************************************************************************************************
                        # USING SHAPELY
                        g = wkb.loads(geom, hex=True)
                        minx, miny, maxx, maxy = g.bounds
            self._geom = g
            self.minx = minx
            self.miny = miny
            self.maxx = maxx
            self.maxy = maxy

    @property
    def tile_name(self):
        names = [self.base_name]
        if self.tile not in (None, ""):  # putting tile before the type makes it sort better by name
            names.append(f"Tile{self.tile}")
        if self.resolution not in (None, ""):  # putting tile before the type makes it sort better by name
            use_res = self.resolution
            if int(self.resolution) == self.resolution:  # remove the decimal point if it's exact integer ("4" instead of "4.0")
                use_res = int(self.resolution)
            names.append("res" + str(use_res))
        return "_".join(names)

    @property
    def full_name(self):
        names = [self.tile_name]
        # Resolution is now in the Tile_name since combines are node based and have to be done custom for each resolution
        # use_res = self.resolution
        # if int(self.resolution) == self.resolution:  # remove the decimal point if it's exact integer ("4" instead of "4.0")
        #     use_res = int(self.resolution)
        # names.append(str(use_res))
        return "_".join(names)

    def update_export_status(self,  connection_info: (ConnectionInfo, psycopg2.extras.DictCursor), database=None, table=None):
        if database is None:
            database = self.SOURCE_DATABASE
        if table is None:
            table = self.RESOLUTIONS_TABLE
        if isinstance(connection_info, ConnectionInfo):
            conn_info = ConnectionInfo(database, connection_info.username, connection_info.password, connection_info.hostname, connection_info.port, [table])
            conn, cursor = connection_with_retries(conn_info)
        else:
            cursor = connection_info
        # trying to just save the c_id from the combine_spec_view and use that to update the combine_spec_bruty table
        # cursor.execute(
        #     f"""update {tablename} set ({self.OUT_OF_DATE},{self.SUMMARY})=(%s, %s)
        #     where ({self.PB},{self.UTM},{self.HEMISPHERE},{self.TILE},{self.DATUM},{self.LOCALITY},{self.RESOLUTION}, {self.DATATYPE},{self.NOT_FOR_NAV})=(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        #     (self.out_of_date, self.summary, self.pb, self.utm, self.hemi.upper(), self.tile, self.datum, self.locality, self.resolution, self.datatype, self.not_for_nav))
        cursor.execute(
            f"""update {table} set ({self.export.START_TIME},{self.export.END_TIME},{self.export.EXIT_CODE},{self.export.WARNINGS_LOG},
            {self.export.INFO_LOG},{self.export.TRIES})=(%s, %s, %s, %s, %s, %s) 
            where ({self.RESOLUTION_ID})=(%s)""",
            (self.export.start_time, self.export.end_time, self.export.exit_code, self.export.warnings_log, self.export.info_log, self.export.tries,
             self.res_tile_id))


class CombineTileInfo(ResolutionTileInfo):
    SOURCE_TABLE = "spec_combines"
    VIEW_TABLE = "view_individual_combines"

    DATA_LOCATION = 'data_location'
    COMBINE_ID = 'c_id'
    FOR_NAV = 'for_navigation'
    DATATYPE = 'datatype'
    RESOLUTION_ID = 'res_id'
    OUT_OF_DATE = "out_of_date"
    SUMMARY = "change_summary"
    PRIMARY_KEY = COMBINE_ID
    JOINED_TABLE = f"""{SOURCE_TABLE} C JOIN {ResolutionTileInfo.SOURCE_TABLE} R ON (C.{RESOLUTION_ID}=R.{ResolutionTileInfo.PRIMARY_KEY}) 
                        JOIN {TileInfo.SOURCE_TABLE} TI ON (R.{ResolutionTileInfo.TILE_ID}=TI.{TileInfo.PRIMARY_KEY})"""

    def __init__(self, **review_tile):
        # @TODO if this becomes used more, make this into a static method called "def from_combine_spec"
        #   and make a different init not based on the combine_spec table layout
        super().__init__(**review_tile)

    def read_record(self, **review_tile):
        super().read_record(**review_tile)

        self.datatype = review_tile[self.DATATYPE]
        self.for_nav = review_tile[self.FOR_NAV]
        self.combine_id = review_tile[self.COMBINE_ID]
        self.combine = CombineOperation(review_tile)
        self.out_of_date = review_tile.get(self.OUT_OF_DATE, None)
        self.summary = review_tile.get(self.SUMMARY, None)

    def __repr__(self):
        return super().__repr__()+f"_{self.datatype}_{'nav' if self.for_nav else 'NOT_NAV'}"

    def hash_id(self):
        return super().hash_id(), hash_id(self.datatype, self.for_nav)

    @classmethod
    def from_combine_spec_view(cls, connection_info: (ConnectionInfo, psycopg2.extras.DictCursor), pk_id, database=None, table=None):
        if database is None:
            database = cls.SOURCE_DATABASE
        if table is None:
            table = cls.SOURCE_TABLE
        if isinstance(connection_info, ConnectionInfo):
            conn_info = ConnectionInfo(database, connection_info.username, connection_info.password, connection_info.hostname, connection_info.port, [table])
            conn, cursor = connection_with_retries(conn_info)
        else:
            cursor = connection_info
        cursor.execute(f"""SELECT * from {table} WHERE {cls.COMBINE_ID}={pk_id}""",)
        record = cursor.fetchone()
        return cls(**record)

    def update_table(self, where, **kwargs):
        """ Update a table based on a a dictionary (from kwargs) of values
        Note, acquire_lock must have been called in order to open a connection, get a cursor and lock the record.
        The function will raise a BaseLockException if the lock has not been acquired.

        Parameters
        ----------
        where
            dictionary of column name(s) and value(s) to match
        kwargs
            dictionary of column name and value to update.  Make sure to wrap strings in single quotes.
        Returns
        -------
        None
        """

        # if database is None:
        #     database = cls.SOURCE_DATABASE
        # if table is None:
        #     table = cls.SOURCE_TABLE
        # if isinstance(connection_info, ConnectionInfo):
        #     conn_info = ConnectionInfo(database, connection_info.username, connection_info.password, connection_info.hostname, connection_info.port, [table])
        #     conn, cursor = connection_with_retries(conn_info)
        # else:
        #     cursor = connection_info
        if not self.cursor:
            raise BaseLockException("No cursor to update the table")
        pg_update(self.cursor, self.SOURCE_TABLE, where, **kwargs)

    def update_table_record(self, **kwargs):
        where = {self.COMBINE_ID: self.combine_id}
        self.update_table(where, **kwargs)

    def metadata_table_name(self, dtype: str=None):
        if dtype is None:
            dtype = self.datatype
        return self.base_name + "_" + dtype  # doesn't have Tile numbers in the metadata table

    def bruty_db_name(self, dtype: str=None, for_nav: bool=None):
        if dtype is None:
            dtype = self.datatype
        if for_nav is None:
            for_nav = self.for_nav
        names = [self.tile_name, dtype]
        if not for_nav:
            names.append(NOT_NAV)
        return "_".join(names)


@dataclass
class TileProcess:
    """ A database instance can be passed into db.
    Alternatively a string of the path can be passed into db and it will open the database when needed.
    """
    console_process: ConsoleProcessTracker
    tile_info: TileInfo
    db: WorldDatabase
    fingerprint: str

    def clear_finish_code(self):
        if isinstance(self.db, (str, pathlib.Path)):
            self.db = WorldDatabase.open(self.db)
        try:
            del self.db.completion_codes[self.fingerprint]
        except KeyError:
            pass

    def finish_code(self):
        if isinstance(self.db, (str, pathlib.Path)):
            self.db = WorldDatabase.open(self.db)
        try:
            return self.db.completion_codes[self.fingerprint].code
        except KeyError:
            return None

    def succeeded(self):
        return self.finish_code() == SUCCEEDED


@dataclass(frozen=True)
class TileToProcess:
    hash_id: str
    resolution: float
    dtype: str = ""
    nav_flag: bool = True
    closing: float = 0


class TileManager:
    RUNNING_PRIORITY = -999

    def __init__(self, config, max_tries, allow_res=False):
        self.config = config
        self.max_tries = max_tries
        self.allow_res = allow_res
        self.user_dtypes, self.user_res = None, None
        self.read_user_settings(self.allow_res)
        self.remaining_tiles = {}

    def read_user_settings(self, allow_res=False):
        try:
            self.user_dtypes = [dt.strip() for dt in parse_multiple_values(self.config['dtypes'])]
        except KeyError:
            self.user_dtypes = None
        try:
            if allow_res:
                self.user_res = [float(dt.strip()) for dt in parse_multiple_values(self.config['res'])]
            else:
                self.user_res = None
        except KeyError:
            self.user_res = None
        self.process_combines = self.config.getboolean('process_combines', True)
        self.process_exports = self.config.getboolean('process_exports', True)

    def refresh_tiles_list(self, needs_combining=False, needs_exporting=False):
        self.remaining_tiles = {}
        if self.process_combines:
            for tile_info in iterate_combine_table(self.config, needs_to_process=needs_combining, max_retries=self.max_tries):
                res = tile_info.resolution
                if self.user_res and res not in self.user_res:
                    continue
                self.remaining_tiles[tile_info.hash_id()] = tile_info
        if self.process_exports:
            for tile_info in iterate_export_table(self.config, needs_to_process=needs_exporting, max_retries=self.max_tries):
                res = tile_info.resolution
                if self.user_res and res not in self.user_res:
                    continue
                self.remaining_tiles[tile_info.hash_id()] = tile_info

        # sort remaining_tiles by priority and then by balance of production branches
        # The remaining_tiles list is already sorted by tile number, so we can just sort by priority and then by production branch
        self.priorities = {}
        for hash, tile in self.remaining_tiles.items():
            use_priority = self._revised_priority(tile)
            try:
                self.priorities[use_priority][tile.pb][hash] = tile
            except KeyError:
                try:
                    self.priorities[use_priority][tile.pb] = {hash: tile}
                except KeyError:
                    self.priorities[use_priority] = {tile.pb: {hash: tile}}
    @staticmethod
    def _revised_priority(tile):
        use_priority = tile.priority if not tile.is_running() else TileManager.RUNNING_PRIORITY
        if use_priority is None:
            use_priority = 0
        return use_priority

    def pick_next_tile(self, currently_running):
        # currently_running: dict(TileProcess)

        # pick the highest priority tile
        for priority in sorted(self.priorities.keys(), reverse=True):
            # balance the production branches
            # find the number of each production branch that is currently running
            # default the count to 0 for each production branch in case none are running
            pb_counts = {pb: 0 for pb in self.priorities[priority].keys()}
            for tile_process in currently_running.values():
                try:
                    pb_counts[tile_process.tile_info.pb] += 1
                except KeyError:  # something is running and no more of its production branch are in the list
                    pb_counts[tile_process.tile_info.pb] = 1

            # sort the production_branches by the number of tiles running
            for pb in sorted(pb_counts.keys(), key=lambda x: pb_counts[x]):
                # if the production branch has tiles at the current priority level then return the first one that isn't running
                if pb in self.priorities[priority]:
                    for hash, tile in self.priorities[priority][pb].items():
                        if hash not in currently_running:
                            return tile
        return None
    def remove(self, tile):
        # The processing loop removes the tile from the remaining_tiles list as part of its processing loop.
        # but if the loop finishes then it refreshes the list and the tile may reappear in the list.
        try:
            hash = tile.hash_id()
            del self.remaining_tiles[hash]
            # if the start/end times got modified then a running tile may now look finished or vice versa, so check both places in our dictionary
            try:
                del self.priorities[self._revised_priority(tile)][tile.pb][hash]
            except KeyError:
                del self.priorities[self.RUNNING_PRIORITY][tile.pb][hash]
        except KeyError:
            pass


def create_world_db(root_path, tile_info: TileInfo, log_level=logging.INFO):
    full_path = pathlib.Path(root_path).joinpath(tile_info.bruty_db_name())

    try:  # see if there is an exising Bruty database
        db = WorldDatabase.open(full_path, log_level=log_level)
    except FileNotFoundError:  # create an empty bruty database
        epsg = tile_info.epsg
        if tile_info.geometry is not None:
            aoi = ((tile_info.minx, tile_info.miny), (tile_info.maxx, tile_info.maxy))
        else:
            aoi = None
        if tile_info.resolution > 4:
            zoom = 10
        elif tile_info.resolution > 2:
            zoom = 11
        elif tile_info.resolution > 1:
            zoom = 12
        else:
            zoom = 13
        # offset comes from NOAA wanting cell center to align at origin
        os.makedirs(root_path, exist_ok=True)
        db = WorldDatabase(
            UTMTileBackendExactRes(tile_info.resolution, tile_info.resolution, epsg, RasterHistory, DiskHistory, TiffStorage, full_path,
                                   offset_x=tile_info.resolution / 2, offset_y=tile_info.resolution / 2,
                                   zoom_level=zoom, log_level=log_level), aoi, log_level=log_level)
    if db.res_x > tile_info.resolution:
        raise BrutyError(f"Existing Bruty data has resolution of {db.res_x} but requested Tile {tile_info.full_name} wants at least {tile_info.resolution}")
    return db

def basic_conditions(config):
    tiles = parse_ints_with_ranges(config.get('tiles', ""))
    zones = parse_ints_with_ranges(config.get('zones', ""))
    production_branches = parse_multiple_values(config.get('production_branches', ""))
    datums = parse_multiple_values(config.get('datums', ""))
    force = config.getboolean('force_build', False)
    conditions = []
    if production_branches:  # put strings in single quotes
        conditions.append(f"""UPPER(production_branch) IN ({', '.join(["'"+pb.upper()+"'" for pb in production_branches])})""")
    if zones:
        conditions.append(f"utm IN ({', '.join([str(utm) for utm in zones])})")
    if tiles:
        conditions.append(f"tile IN ({', '.join([str(t) for t in tiles])})")
    if datums:  # put strings in single quotes
        conditions.append(f"""datum IN ({', '.join(["'"+str(d)+"'" for d in datums])})""")
    if not force:
        conditions.append(f"""(build IS TRUE)""")
    return conditions

# def exclude_running_condition(cls):
#     # SELECT (c_id NOT IN (SELECT c_id FROM spec_combines FOR UPDATE SKIP LOCKED)) as running FROM spec_combines
#     return f"""({cls.END_TIME} > {cls.START_TIME}  OR -- finished running previously or
#                             {cls.START_TIME} IS NULL) -- never ran"""

def needs_processing_condition(cls, max_retries=3):
    return f"""({cls.REQUEST_TIME} IS NOT NULL AND 
                 -- never started, hasn't finished, or has a positive exit code and hasn't retried too many times
                 (({cls.START_TIME} IS NULL OR {cls.END_TIME} IS NULL OR {cls.REQUEST_TIME} > {cls.START_TIME} OR {cls.START_TIME} > {cls.END_TIME})
                   OR ({cls.EXIT_CODE} > 0 AND ({cls.TRIES} IS NULL OR {cls.TRIES} < {max_retries})))) -- failed with a positive exit code"""
    # Seems that the time check is slow in SQL but fast in python, so grab the columns and do the check in python

def needs_processing(record, operation_type, max_retries=3):
    t = record[operation_type.REQUEST_TIME]
    process = False
    if t is not None:
        s = record[operation_type.START_TIME]
        e = record[operation_type.END_TIME]
        if s is None or e is None:
            process = True
        elif t>s or s>e:
            process = True
        elif record[operation_type.EXIT_CODE] > 0 and record[operation_type.TRIES] < max_retries:
            process = True
    return process

def lock_column_query(table_name, alias, do_lookup=True, col_name=TileInfo.IS_LOCKED):
    if do_lookup:
        # the R comes from the table alias in the execute statement below

        lock_column = f"""(SELECT ({alias}.ctid NOT IN (SELECT ctid FROM {table_name} FOR UPDATE SKIP LOCKED))) as {col_name}"""
    else:
        lock_column = f"False as {col_name}"
    return lock_column

def get_export_records(config, needs_to_process=False, get_lock_status=True, max_retries=3):
    """
    Parameters
    ----------
    config
    needs_to_process
        This is very slow in SQL, so it's better to do this in python
    get_lock_status
    max_retries

    Returns
    -------

    """
    conn_info = connect_params_from_config(config)
    conn_info.database = TileInfo.SOURCE_DATABASE
    connection, cursor = connection_with_retries(conn_info)
    conditions = basic_conditions(config)
    if needs_to_process:
        conditions.append(needs_processing_condition(ExportOperation, max_retries=max_retries))
    where_clause = "\nAND ".join(conditions)  # a comment at the end was making this fail, so put a leading newline
    if where_clause:
        where_clause = " WHERE " + where_clause
    lock_query = lock_column_query(ResolutionTileInfo.SOURCE_TABLE, "R", get_lock_status)
    # This is the tiles with their resolutions and the SRID is based on geometry_buffered
    # SELECT R.*, ST_SRID(geometry_buffered), TI.*, {lock_query}
    # Querying the geometry is very slow - do a query when we need that specific record
    # and doing the time check in python is faster than doing it in SQL
    cursor.execute(f"""SELECT {TileInfo.TILE}, {TileInfo.UTM}, {TileInfo.PB},{TileInfo.DATUM}, {TileInfo.LOCALITY}, {TileInfo.HEMISPHERE},
                            {TileInfo.EXPORT_PUBLIC}, {TileInfo.EXPORT_INTERNAL}, {TileInfo.EXPORT_NAVIGATION}, {TileInfo.PRIORITY}, {TileInfo.BUILD},
                            {ExportOperation.START_TIME}, {ExportOperation.END_TIME}, {ExportOperation.EXIT_CODE}, {ExportOperation.TRIES}, {ExportOperation.REQUEST_TIME},
                            {ResolutionTileInfo.RESOLUTION}, {ResolutionTileInfo.CLOSING_DISTANCE}, {ResolutionTileInfo.PRIMARY_KEY}, 
                            {lock_query} 
                       FROM spec_resolutions R 
                       JOIN spec_tiles TI ON (R.tile_id = TI.t_id)
                       {where_clause}
                       ORDER BY priority DESC, tile ASC
                       """)
    records = cursor.fetchall()
    connection.close()
    return records

def get_combine_records(config, needs_to_process=False, get_lock_status=True, max_retries=3):
    """ Read the NBS postgres tile_specifications database for the bruty_tile table and, if not existing,
    create Bruty databases for the area of interest for the polygon listed in the postgres table.

    Parameters
    ----------
    config
    needs_to_process
        This is very slow in SQL, so it's better to do this in python
    get_lock_status
    max_retries

    Returns
    -------

    """
    dtypes = parse_multiple_values(config.get('dtypes', ""))

    conn_info = connect_params_from_config(config)
    conn_info.database = TileInfo.SOURCE_DATABASE
    connection, cursor = connection_with_retries(conn_info)
    # tile specs are now being held in views, so we have to read the combine_specs table
    # then read the geometry from the view of the equivalent area/records
    # grab them all at run time so that if records are changed during the run we at least know all the geometries should have been from the run time
    conditions = basic_conditions(config)
    if needs_to_process:
        conditions.append(needs_processing_condition(CombineOperation, max_retries=max_retries))
    if dtypes:
        conditions.append(f"""LOWER(datatype) IN ({', '.join(["'" + dt.lower() + "'" for dt in dtypes])})""")
    where_clause = "\nAND ".join(conditions)  # a comment at the end was making this fail, so put a leading newline
    if where_clause:
        where_clause = " WHERE " + where_clause
    lock_query = lock_column_query(CombineTileInfo.SOURCE_TABLE, "B", get_lock_status)
    # This is very similar to the combine_spec_view except we are getting the geometry_buffered and its SRID which are not in the view because of QGIS performance
    # Actually it seems that when querying the whole table it's significantly faster to get less columns, so just get the critical ones then query the rest when needed.
    # The killer is the geometry, a query without geometry is taking 0.5 seconds, with geometry it's taking 25 seconds
    # Using the time comparison as SQL would take a 0.6 second query to 13 seconds
    # SELECT B.*, R.*, ST_SRID(geometry_buffered), TI.*, {lock_query}
    cursor.execute(f"""SELECT {TileInfo.TILE}, {TileInfo.UTM}, {TileInfo.PB},{TileInfo.DATUM}, {TileInfo.LOCALITY}, {TileInfo.HEMISPHERE}, 
                            {TileInfo.PRIORITY}, {TileInfo.BUILD},
                            {CombineOperation.START_TIME}, {CombineOperation.END_TIME}, {CombineOperation.EXIT_CODE}, {CombineOperation.TRIES}, {CombineOperation.REQUEST_TIME},
                            {ResolutionTileInfo.RESOLUTION}, 
                            {CombineTileInfo.DATATYPE}, {CombineTileInfo.FOR_NAV}, {CombineTileInfo.PRIMARY_KEY},  
                            {lock_query} 
                       FROM {CombineTileInfo.SOURCE_TABLE} B 
                       JOIN {ResolutionTileInfo.SOURCE_TABLE} R ON (B.res_id = R.r_id) 
                       JOIN {TileInfo.SOURCE_TABLE} TI ON (R.tile_id = TI.t_id)
                       {where_clause}
                       ORDER BY priority DESC, tile ASC
                       """)
    records = cursor.fetchall()
    connection.close()
    return records

def iterate_export_table(config, needs_to_process=False, get_lock_status=True, max_retries=3):
    records = get_export_records(config, get_lock_status=get_lock_status, max_retries=max_retries)
    for review_tile in records:
        if needs_to_process and not needs_processing(review_tile, ExportOperation, max_retries=max_retries):
            continue
        info = ResolutionTileInfo(**review_tile)
        yield info

def iterate_combine_table(config, needs_to_process=False, get_lock_status=True, max_retries=3):
    records = get_combine_records(config, get_lock_status=get_lock_status, max_retries=max_retries)
    for review_tile in records:
        if needs_to_process and not needs_processing(review_tile, CombineOperation, max_retries=max_retries):
            continue
        info = CombineTileInfo(**review_tile)
        yield info

