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
from nbs.bruty.nbs_postgres import NOT_NAV, get_nbs_records, get_sorting_info, get_transform_metadata, connect_params_from_config, hash_id
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
    NOT_FOR_NAV = 'not_for_navigation'
    DATATYPE = 'datatype'
    VIEW_ID = 'b_id'
    START_TIME = 'start_time'
    END_TIME = 'end_time'
    EXIT_CODE = 'exit_code'
    TRIES = 'tries'

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
        self.not_for_nav = review_tile[self.NOT_FOR_NAV]
        self.view_id = review_tile[self.VIEW_ID]
        self.tries = review_tile[self.TRIES]

        self.out_of_date = review_tile[self.OUT_OF_DATE]
        self.summary = review_tile[self.SUMMARY]
        self.start_time = review_tile[self.START_TIME]
        self.end_time = review_tile[self.END_TIME]
        self.exit_code = review_tile[self.EXIT_CODE]
        self.epsg = review_tile['st_srid']
        self.geometry = review_tile['geometry']
        self.public = review_tile['combine_public']  # public - included "unqualified"
        self.internal = review_tile['combine_internal']  # includes sensitive
        self.navigation = review_tile['combine_navigation']  # only QC'd (qualified) data
        # for_nav = review_tile.get("for_nav", True)
        # self.for_nav = "" if for_nav else "not_for_navigation"
        # self.data_type = review_tile.get("dtype", "")

    def __repr__(self):
        return f"TileInfo:{self.pb}_{self.utm}{self.hemi}_{self.tile}_{self.locality}"

    def hash_id(self, res=None):
        if res is None:
            res = self.resolution
        return hash_id(self.pb, self.utm, self.hemi, self.tile, self.datum, res)

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
            f"""update {table} set ({self.START_TIME},{self.END_TIME},{self.EXIT_CODE},{self.OUT_OF_DATE},{self.SUMMARY},{self.TRIES})=(%s, %s, %s, %s, %s, %s) 
            where ({self.VIEW_ID})=(%s)""",
            (self.start_time, self.end_time, self.exit_code, self.out_of_date, self.summary, self.tries, self.view_id))

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
            for_nav = not self.not_for_nav
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


def iterate_tiles_table(config, only_needs_to_combine=False, max_retries=3):
    """ Read the NBS postgres tile_specifications database for the bruty_tile table and, if not existing,
    create Bruty databases for the area of interest for the polygon listed in the postgres table.
    """
    tiles = parse_ints_with_ranges(config.get('tiles', ""))
    zones = parse_ints_with_ranges(config.get('zones', ""))
    production_branches = parse_multiple_values(config.get('production_branches', ""))
    datums = parse_multiple_values(config.get('datums', ""))

    force = config.getboolean('force_build', False)

    conn_info = connect_params_from_config(config)
    conn_info.database = "tile_specifications"
    # tile specs are now being held in views, so we have to read the combine_specs table
    # then read the geometry from the view of the equivalent area/records
    if only_needs_to_combine:
        fields, records = get_nbs_records("combine_spec_view", conn_info,
                                          where_clause=f"""WHERE (end_time > start_time  OR -- finished running previously or  
                                                           (start_time IS NULL AND combine_time IS NOT NULL)) -- never ran
                                                           AND 
                                                           (combine_time IS NOT NULL AND
                                                             ((start_time IS NULL OR combine_time > start_time) OR -- not started yet
                                                              (exit_code > 0 AND (tries IS NULL OR tries < {max_retries})))) -- failed with a positive exit code""",
                                          geom_name='geometry', order='ORDER BY priority DESC, tile ASC')
    else:
        fields, records = get_nbs_records(f"combine_spec_resolutions JOIN combine_spec_tiles ON t_id=res_id", conn_info, geom_name='geometry', order='ORDER BY tile ASC')
    branch_utms = {}
    for review_tile in records:
        utms = branch_utms.setdefault(review_tile['production_branch'], set())
        utms.add(str(review_tile['utm']) + review_tile['hemisphere'])
    # grab them all at run time so that if records are changed during the run we at least know all the geometries should have been from the run time
    all_records = []
    for branch, utms in branch_utms.items():
        if production_branches and branch not in production_branches:
            continue
        else:
            for utm in utms:
                wants_recs = True
                # get records is a slow operation so don't call it unless we need it or aren't sure
                if zones:
                    utmzone= int(re.search(r"\d+", utm).group())
                    if utmzone not in zones:
                        wants_recs = False
                if wants_recs:
                    if only_needs_to_combine:
                        fields, records = get_nbs_records("combine_spec_view", conn_info,
                                                          where_clause=f"""WHERE (end_time > start_time  OR -- finished running previously or  
                                                                           (start_time IS NULL AND combine_time IS NOT NULL)) -- never ran
                                                                           AND 
                                                                           (combine_time IS NOT NULL AND
                                                                             ((start_time IS NULL OR combine_time > start_time) OR -- not started yet
                                                                              (exit_code > 0 AND (tries IS NULL OR tries < {max_retries})))) -- failed with a positive exit code""",
                                                          geom_name='geometry', order='ORDER BY priority DESC, tile ASC')
                    else:
                        fields, records = get_nbs_records(f"SELECT * from combine_spec_{branch}_{utm}", conn_info, geom_name='geometry', order='ORDER BY tile ASC')
                    raise Exception("fix the combine_spec views")
                    all_records.extend(records)
    for review_tile in all_records:
        info = TileInfo(**review_tile)
        # determine if this tile should be processed
        if not (force or info.build):
            continue
        if production_branches and info.pb not in production_branches:
            continue
        if datums and info.datum not in datums:
            continue
        if tiles and int(info.tile) not in tiles:
            continue
        if zones and info.utm not in zones:
            continue
        # yield db or do a callback function?
        yield info


""" 
>>> prior = psutil.pids()
>>> p = subprocess.Popen(args[1:], creationflags=subprocess.CREATE_NEW_CONSOLE)
>>> for p in psutil.pids():
...   try:
...     if p not in prior: print(psutil.Process(p).cmdline())
...   except:
...     pass
...
['cmd.exe', '/K', 'set', 'pythonpath=&&', 'set', 'TCL_LIBRARY=&&', 'set', 'TIX_LIBRARY=&&', 'set', 'TK_LIBRARY=&&', 'C:\\PydroSVN\\trunk\\Miniconda36\\scripts\\activate', 'Pydro27', '&&', 'python']
['python']
['\\??\\C:\\WINDOWS\\system32\\conhost.exe', '0x4']
>>> for p in psutil.pids():
...   try:
...     if p not in prior: print(psutil.Process(p).cmdline())
...   except:
...     pass
...
['cmd.exe', '/K', 'set', 'pythonpath=&&', 'set', 'TCL_LIBRARY=&&', 'set', 'TIX_LIBRARY=&&', 'set', 'TK_LIBRARY=&&', 'C:\\PydroSVN\\trunk\\Miniconda36\\scripts\\activate', 'Pydro27', '&&', 'python']
['\\??\\C:\\WINDOWS\\system32\\conhost.exe', '0x4']
>>> p = psutil.Popen(args[1:], creationflags=subprocess.CREATE_NEW_CONSOLE)
>>> p.is_running()
True
"""

if __name__ == '__main__':
    # Runs the main function for each config specified in sys.argv
    run_configs(iterate_tiles_table, "Splitting Tiles", ["debug.config"], logger=LOGGER)
