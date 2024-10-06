import os
import pathlib
from dataclasses import dataclass
import logging
import re

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
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.utils import ConsoleProcessTracker

NO_DATA = -1
SUCCEEDED = 0
DATA_ERRORS = 3
TILE_LOCKED = 4
FAILED_VALIDATION = 5
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

    def __init__(self, record=None):
        if record:
            self.read(record)

    def read(self, record):
        self.start_time = record[self.START_TIME]
        self.end_time = record[self.END_TIME]
        self.exit_code = record[self.EXIT_CODE]
        self.warnings_log = record[self.WARNINGS_LOG]
        self.info_log = record[self.INFO_LOG]
        self.tries = record[self.TRIES]


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

    def __init__(self, record=None):
        super().__init__(record)


class TileInfo:
    SOURCE_DATABASE = "tile_specifications"
    SOURCE_TABLE = "combine_spec_view"

    OUT_OF_DATE = "out_of_date"
    SUMMARY = "change_summary"
    PB = 'production_branch'
    UTM = 'utm'
    TILE = 'tile'
    DATUM = 'datum'
    LOCALITY = 'locality'
    HEMISPHERE = 'hemisphere'
    RESOLUTION = "resolution"
    CLOSING_DISTANCE = 'closing_distance'
    FOR_NAV = 'for_navigation'
    DATATYPE = 'datatype'
    VIEW_ID = 'b_id'
    TRIES = 'tries'
    PRIORITY = 'priority'
    DATA_LOCATION = 'data_location'

    def __init__(self, **review_tile):
        # @TODO if this becomes used more, make this into a static method called "def from_combine_spec"
        #   and make a different init not based on the combine_spec table layout
        self.minx = None
        self.miny = None
        self.maxx = None
        self.maxy = None
        self.hemi = review_tile[self.HEMISPHERE].lower()
        self.pb = review_tile[self.PB]
        self.utm = review_tile[self.UTM]
        self.tile = review_tile[self.TILE]
        self.datum = review_tile[self.DATUM]
        self.build = review_tile['build']
        self.locality = review_tile[self.LOCALITY]
        self.resolution = review_tile[self.RESOLUTION]
        self.closing_dist = review_tile[self.CLOSING_DISTANCE]
        self.datatype = review_tile[self.DATATYPE]
        self.for_nav = review_tile[self.FOR_NAV]
        self.view_id = review_tile[self.VIEW_ID]
        self.tries = review_tile[self.TRIES]
        self.priority = review_tile[self.PRIORITY] if review_tile[self.PRIORITY] else 0  # if None, make it 0

        self.out_of_date = review_tile[self.OUT_OF_DATE]
        self.summary = review_tile[self.SUMMARY]
        self.combine = CombineOperation(review_tile)
        self.export = ExportOperation(review_tile)
        self.data_location = review_tile[self.DATA_LOCATION]

        self.epsg = review_tile['st_srid']
        self.geometry = review_tile['geometry'] if 'geometry_buffered' not in review_tile else review_tile['geometry_buffered']
        self.public = review_tile['combine_public']  # public - included "unqualified"
        self.internal = review_tile['combine_internal']  # includes sensitive
        self.navigation = review_tile['combine_navigation']  # only QC'd (qualified) data
        # for_nav = review_tile.get("for_nav", True)
        # self.for_nav = "" if for_nav else "not_for_navigation"
        # self.data_type = review_tile.get("dtype", "")

    def __repr__(self):
        return f"TileInfo:{self.pb}_{self.utm}{self.hemi}_{self.tile}_{self.locality}_{self.datatype}_{self.resolution}m_{'nav' if self.for_nav else 'NOT_NAV'}"

    def hash_id(self):
        return hash_id(self.pb, self.utm, self.hemi, self.tile, self.datum, self.resolution, self.datatype, self.for_nav)

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
        cursor.execute(f"""SELECT * from {table} WHERE b_id={pk_id}""",)
        tile_info = cursor.fetchone()
        return cls(tile_info)

    @classmethod
    def update_table(cls, connection_info: (ConnectionInfo, psycopg2.extras.DictCursor), where, database=None, table=None, **kwargs):
        """ Update a table based on a a dictionary (from kwargs) of values

        Parameters
        ----------
        cursor
            open psycopg2 cursor
        where
            dictionary of column name(s) and value(s) to match
        database
            name of the database to update, default is the tile_specifications
        table_name
            name of the table to update, default is the combine_spec_view
        kwargs
            dictionary of column name and value to update.  Make sure to wrap strings in single quotes.
        Returns
        -------
        None
        """

        if database is None:
            database = cls.SOURCE_DATABASE
        if table is None:
            table = cls.SOURCE_TABLE
        if isinstance(connection_info, ConnectionInfo):
            conn_info = ConnectionInfo(database, connection_info.username, connection_info.password, connection_info.hostname, connection_info.port, [table])
            conn, cursor = connection_with_retries(conn_info)
        else:
            cursor = connection_info
        pg_update(cursor, table, where, **kwargs)

    def _is_running(self, obj):
        return (obj.start_time is not None) and ((obj.end_time is None) or (obj.start_time > obj.end_time))

    @property
    def combine_is_running(self):
        return self._is_running(self.combine)

    @property
    def export_is_running(self):
        return self._is_running(self.export)

    def update_table_record(self, connection_info: (ConnectionInfo, psycopg2.extras.DictCursor), database=None, table=None, **kwargs):
        where = {self.VIEW_ID: self.view_id}
        self.update_table(connection_info, where, database, table, **kwargs)

    def update_table_status(self,  connection_info: (ConnectionInfo, psycopg2.extras.DictCursor), database=None, table=None):
        if database is None:
            database = self.SOURCE_DATABASE
        if table is None:
            table = self.SOURCE_TABLE
        if isinstance(connection_info, ConnectionInfo):
            conn_info = ConnectionInfo(database, connection_info.username, connection_info.password, connection_info.hostname, connection_info.port, [table])
            conn, cursor = connection_with_retries(conn_info)
        else:
            cursor = connection_info
        # trying to just save the b_id from the combine_spec_view and use that to update the combine_spec_bruty table
        # cursor.execute(
        #     f"""update {tablename} set ({self.OUT_OF_DATE},{self.SUMMARY})=(%s, %s)
        #     where ({self.PB},{self.UTM},{self.HEMISPHERE},{self.TILE},{self.DATUM},{self.LOCALITY},{self.RESOLUTION}, {self.DATATYPE},{self.NOT_FOR_NAV})=(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        #     (self.out_of_date, self.summary, self.pb, self.utm, self.hemi.upper(), self.tile, self.datum, self.locality, self.resolution, self.datatype, self.not_for_nav))
        cursor.execute(
            f"""update {table} set ({self.combine.START_TIME},{self.combine.END_TIME},{self.combine.EXIT_CODE},{self.combine.WARNINGS_LOG},{self.combine.INFO_LOG},
            {self.export.START_TIME},{self.export.END_TIME},{self.export.EXIT_CODE},{self.export.WARNINGS_LOG},{self.export.INFO_LOG},
            {self.OUT_OF_DATE},{self.SUMMARY},{self.TRIES})=(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            where ({self.VIEW_ID})=(%s)""",
            (self.combine.start_time, self.combine.end_time, self.combine.exit_code, self.combine.warnings_log, self.combine.info_log,
             self.export.start_time, self.export.end_time, self.export.exit_code, self.export.warnings_log, self.export.info_log,
             self.out_of_date, self.summary, self.tries, self.view_id))

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
    def base_name(self):
        return "_".join([self.pb, self.locality, f"utm{self.utm}{self.hemi}", self.datum])

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
    console_process: ConsoleProcessTracker
    tile_info: TileInfo
    db: WorldDatabase
    fingerprint: str
    lock: str = None

    def clear_finish_code(self):
        try:
            del self.db.completion_codes[self.fingerprint]
        except KeyError:
            pass

    def finish_code(self):
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

    def refresh_tiles_list(self, needs_combining=False):
        self.remaining_tiles = {}
        for tile_info in iterate_tiles_table(self.config, only_needs_to_combine=needs_combining, max_retries=self.max_tries):
            res = tile_info.resolution
            if self.user_res and res not in self.user_res:
                continue
            for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                if self.user_dtypes and dtype not in self.user_dtypes:
                    continue
                for nav_flag_value in (True, False):
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
        use_priority = tile.priority if not tile.is_running else TileManager.RUNNING_PRIORITY
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

            for pb in sorted(pb_counts.keys(), key=lambda x: pb_counts[x]):
                for hash, tile in self.priorities[priority][pb].items():
                    if hash not in currently_running:
                        return tile
        return None
    def remove(self, tile):
        try:
            hash = tile.hash_id()
            del self.remaining_tiles[hash]
            del self.priorities[self._revised_priority(tile)][tile.pb][hash]
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


def get_tiles_records(config, needs_to_combine=False, needs_to_export=False, ignore_combine_running=False, ignore_export_running=False, max_retries=3):
    """ Read the NBS postgres tile_specifications database for the bruty_tile table and, if not existing,
    create Bruty databases for the area of interest for the polygon listed in the postgres table.
    """
    tiles = parse_ints_with_ranges(config.get('tiles', ""))
    zones = parse_ints_with_ranges(config.get('zones', ""))
    production_branches = parse_multiple_values(config.get('production_branches', ""))
    dtypes = parse_multiple_values(config.get('dtypes', ""))
    datums = parse_multiple_values(config.get('datums', ""))
    force = config.getboolean('force_build', False)

    conn_info = connect_params_from_config(config)
    conn_info.database = "tile_specifications"
    connection, cursor = connection_with_retries(conn_info)
    # tile specs are now being held in views, so we have to read the combine_specs table
    # then read the geometry from the view of the equivalent area/records
    # grab them all at run time so that if records are changed during the run we at least know all the geometries should have been from the run time
    conditions = []
    def ignore_running(cls):
        conditions.append(f"""({cls.END_TIME} > {cls.START_TIME}  OR -- finished running previously or
                                {cls.START_TIME} IS NULL) -- never ran""")
    if ignore_combine_running:
        ignore_running(CombineOperation)
    if ignore_export_running:
        ignore_running(ExportOperation)
    def needs_processing(cls):
        conditions.append(f"""({cls.REQUEST_TIME} IS NOT NULL AND
                             (({cls.START_TIME} IS NULL OR {cls.REQUEST_TIME} > {cls.START_TIME}) OR -- not started yet
                              ({cls.EXIT_CODE} > 0 AND ({cls.TRIES} IS NULL OR {cls.TRIES} < {max_retries})))) -- failed with a positive exit code""")
    if needs_to_combine:
        needs_processing(CombineOperation)
    if needs_to_export:
        needs_processing(ExportOperation)
    if production_branches:  # put strings in single quotes
        conditions.append(f"""UPPER(production_branch) IN ({', '.join(["'"+pb.upper()+"'" for pb in production_branches])})""")
    if zones:
        conditions.append(f"utm IN ({', '.join([str(utm) for utm in zones])})")
    if tiles:
        conditions.append(f"tile IN ({', '.join([str(t) for t in tiles])})")
    if dtypes:
        conditions.append(f"""LOWER(datatype) IN ({', '.join(["'" + dt.lower() + "'" for dt in dtypes])})""")
    if datums:  # put strings in single quotes
        conditions.append(f"""datum IN ({', '.join(["'"+str(d)+"'" for d in datums])})""")
    if not force:
        conditions.append(f"""(build IS TRUE)""")
    where_clause = "\nAND ".join(conditions)  # a comment at the end was making this fail, so put a leading newline
    if where_clause:
        where_clause = " WHERE " + where_clause
    # This is very similar to the combine_spec_view except we are getting the geometry_buffered and its SRID which are not in the view because of QGIS performance
    cursor.execute(f"""SELECT B.*, R.*, ST_SRID(geometry_buffered), TI.* 
                       FROM combine_spec_bruty B JOIN combine_spec_resolutions R ON (B.res_id = R.r_id) JOIN combine_spec_tiles TI ON (R.tile_id = TI.t_id)
                       {where_clause}
                       ORDER BY priority DESC, tile ASC
                       """)
    records = cursor.fetchall()
    connection.close()
    return records

def iterate_tiles_table(config, only_needs_to_combine=False, only_needs_to_export=False,
                        ignore_combine_running=False, ignore_export_running=False, max_retries=3):
    records = get_tiles_records(config, needs_to_combine=only_needs_to_combine, ignore_combine_running=ignore_combine_running,
                                needs_to_export=only_needs_to_export, ignore_export_running=ignore_export_running, max_retries=max_retries)
    for review_tile in records:
        info = TileInfo(**review_tile)
        # determine if this tile should be processed
        # if not (force or info.build):
        #     continue
        # if production_branches and info.pb not in production_branches:
        #     continue
        # if datums and info.datum not in datums:
        #     continue
        # if tiles and int(info.tile) not in tiles:
        #     continue
        # if zones and info.utm not in zones:
        #     continue
        # yield db or do a callback function?
        yield info

