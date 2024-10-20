import os
import pathlib
from dataclasses import dataclass
import logging
import re
import time

import numpy
import psycopg2
from osgeo import ogr, gdal
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


class SQLConnectionAndCursor:
    """ This is to speed up some operations by supplying a cursor rather than making a connection which cost about 3 seconds per new connection. """
    def __init__(self, conn_info_or_cursor: (ConnectionInfo, (psycopg2.extensions.connection, psycopg2.extras.DictCursor), dict)):
        """ Accepts a ConnectionInfo object or a tuple of connection and cursor or a dictionary that can be used to create a ConnectionInfo object.

        Creates attributes of conn and cursor based on the input.

        Parameters
        ----------
        conn_info_or_cursor
        """
        try:  # a dict or configparse object that acts like a dict
            conn_info = connect_params_from_config(conn_info_or_cursor)
        except (TypeError, KeyError):  # either a ConnectionInfo object or a tuple of connection and cursor
            conn_info = conn_info_or_cursor
        try:
            # @TODO allow for variable SOURCE_DATABASE locations
            conn_copy = ConnectionInfo(CombineTileInfo.SOURCE_DATABASE, conn_info.username, conn_info.password, conn_info.hostname,
                                       conn_info.port, [])
            self.conn, self.cursor = connection_with_retries(conn_copy)
        except AttributeError:  # must be a tuple
            if isinstance(conn_info_or_cursor[1], psycopg2.extras.DictCursor):
                self.conn, self.cursor = conn_info_or_cursor
            else:
                raise ValueError("The arguement was not recognized as a way to connect to a sql database\nAccepts a configparse, dictionary, (connection, cursor) or ConnectionInfo object.")

    @classmethod
    def from_info(cls, info):
        """ passes through if already a SQLConnectionAndCursor object, otherwise creates a new one from cls.__init__"""
        if isinstance(info, cls):
            return info
        else:
            return cls(info)


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
    RUNNING = 'running'

    def __init__(self, record=None):
        if record:
            self.read(record)

    def read(self, record):
        self.start_time = record.get(self.START_TIME, None)
        self.end_time = record.get(self.END_TIME, None)
        self.request_time = record.get(self.REQUEST_TIME, None)
        self.exit_code = record.get(self.EXIT_CODE, None)
        self.tries = record.get(self.TRIES, None)
        self.warnings_log = record.get(self.WARNINGS_LOG, None)
        self.info_log = record.get(self.INFO_LOG, None)
        self.data_location = record.get(self.DATA_LOCATION, None)
        self.running = record.get(self.RUNNING, None)

    def needs_processing(self):
        process = False
        if self.request_time is not None:
            if self.start_time is None or self.end_time is None:
                process = True
            elif self.request_time > self.start_time or self.start_time > self.end_time:
                process = True
            elif self.exit_code > 0:
                process = True
        return process


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
    RUNNING = OPERATION + "_" + BrutyOperation.RUNNING

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
    RUNNING = OPERATION + "_" + BrutyOperation.RUNNING

    def __init__(self, record=None):
        super().__init__(record)


class TileInfo:
    SOURCE_DATABASE = "tile_specifications"
    SOURCE_TABLE = "spec_tiles"
    JOINED_TABLE = "spec_tiles"
    IS_LOCKED = 'is_locked'
    LOCK_QUERY = f"""(SELECT (ctid NOT IN (SELECT ctid FROM {SOURCE_TABLE} FOR UPDATE SKIP LOCKED))) as {IS_LOCKED}"""

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
    PRIMARY_KEY = TILE_ID
    EPSG_QUERY = 'ST_SRID(geometry)'
    GEOMETRY = 'geometry'

    def __init__(self, **review_tile):
        # @TODO if this becomes used more, make this into a static method called "def from_combine_spec"
        #   and make a different init not based on the combine_spec table layout
        self.sql_obj = None
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
        self.geometry = review_tile.get(self.GEOMETRY, None)

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
            self.raw_geometry = geom
            self.minx = minx
            self.miny = miny
            self.maxx = maxx
            self.maxy = maxy

    def refresh_lock_status(self, sql_info):
        """ Update the self.is_locked value based on the database - if the record is not found then is_locked will be set to None

        Parameters
        ----------
        conn_info

        Returns
        -------

        """
        locks = self.check_locks(sql_info, self.pk)
        try:
            self.is_locked = locks[0][self.IS_LOCKED]
        except IndexError:
            self.is_locked = None
        return self.is_locked

    @classmethod
    def from_table(cls, sql_info, pk_id, table=None):
        """ Reads a records from the JOINED_TABLE based on the primary key and returns a TileInfo object
        Parameters
        ----------
        connection_info
        pk_id
        database
        table

        Returns
        -------

        """
        if table is None:
            table = cls.JOINED_TABLE
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
        sql_obj.cursor.execute(f"""SELECT *, {cls.EPSG_QUERY} from {table} WHERE {cls.PRIMARY_KEY}={pk_id}""",)
        record = sql_obj.cursor.fetchone()
        return cls(**record)

    @classmethod
    def check_locks(cls, sql_info, pk_vals):
        """ Check the lock status of multiple records
        Parameters
        ----------
        conn_info
            ConnectionInfo object
        pk_vals
            Primary key values to check

        Returns
        -------

        """
        records = cls._get_records(sql_info, pk_vals, f"{cls.PRIMARY_KEY}, {cls.LOCK_QUERY}", cls.JOINED_TABLE)
        return records

    @classmethod
    def _get_records(cls, sql_info, pk_vals, select, table):
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
        # This is very similar to the combine_spec_view except we are getting the geometry_buffered and its SRID which are not in the view because of QGIS performance
        # Actually it seems that when querying the whole table it's significantly faster to get less columns, so just get the critical ones then query the rest when needed.
        # The killer is the geometry, a query without geometry is taking 0.5 seconds, with geometry it's taking 25 seconds
        # Using the time comparison as SQL would take a 0.6 second query to 13 seconds
        # SELECT B.*, R.*, ST_SRID(geometry_buffered), TI.*, {lock_query}

        # to use the IN operator we need a tuple (not a list) of values
        # if pk_vals isn't iterable then make it a tuple
        if not hasattr(pk_vals, '__iter__') or isinstance(pk_vals, str):
            tuple_vals = (pk_vals,)
        else:
            tuple_vals = tuple(pk_vals)
        sql_obj.cursor.execute(f"""SELECT {select}
                           FROM {table} 
                           WHERE {cls.PRIMARY_KEY} IN %s
                           """, (tuple_vals,))
        records = sql_obj.cursor.fetchall()
        return records

    @classmethod
    def get_full_records(cls, sql_info, pk_vals):
        """ Check the lock status of multiple records
        Parameters
        ----------
        conn_info
            ConnectionInfo object
        pk_vals
            Primary key values to check

        Returns
        -------

        """
        records = cls._get_records(sql_info, pk_vals, f"*, {cls.EPSG_QUERY}, {cls.LOCK_QUERY}", cls.JOINED_TABLE)
        return [cls(**record) for record in records]


    def __repr__(self):
        return f"TileInfo:{self.pb}_{self.utm}{self.hemi}_{self.tile}_{self.locality}"

    def hash_id(self):
        """ Returns a hash of the tile info so it can be used as a dictionary key"""
        return hash_id(self.pb, self.utm, self.hemi, self.tile, self.datum)

    @property
    def base_name(self):
        return "_".join([self.pb, self.locality, f"utm{self.utm}{self.hemi}", self.datum])

    def _set_running(self, bool_val, set_all=False):
        """ When called with
        """
        if set_all:
            where = f"WHERE {self.PRIMARY_KEY} IN (SELECT {self.PRIMARY_KEY} FROM {self.SOURCE_TABLE} FOR UPDATE SKIP LOCKED)"
        else:  # this sets everyone to false except the ones that are locked -- which cleans up any errors
            where = f"WHERE {self.PRIMARY_KEY}={self.pk}"
        self.sql_obj.cursor.execute(f"""UPDATE {self.SOURCE_TABLE} SET {self.RUNNING}={bool_val} {where}""")
        # self.sql_obj.cursor.execute(f"""UPDATE {self.SOURCE_TABLE} SET {self.RUNNING}={bool_val} WHERE {self.PRIMARY_KEY}={self.pk}""")

    def has_lock(self):
        return self.sql_obj is not None

    def acquire_lock(self, conn_info):
        """ Acquire a lock on the record in the database using a NEW connection (can't pass an existing connection).
        This is because if close() or commit()) is called on the connection then the lock is released.
        If successful, it will return the connection and cursor and update the record with the full record.
        If not, it will raise a BaseLockException.
        """
        if isinstance(conn_info, ConnectionInfo):
            self.sql_obj = SQLConnectionAndCursor(conn_info)
            self._set_running(True)
            self.sql_obj.conn.set_session(autocommit=False)
        else:
            raise ValueError("The connection info must be a ConnectionInfo object")
        self.sql_obj.cursor.execute(f"""SELECT * FROM {self.SOURCE_TABLE} WHERE {self.PRIMARY_KEY}=%s FOR UPDATE SKIP LOCKED""", (self.pk,))
        record = self.sql_obj.cursor.fetchone()
        if record is None:
            self.release_lock()
            full_info = None
            raise BaseLockException(f"Failed to acquire lock for {self} where PK={self.pk}")
        else:
            self.sql_obj.cursor.execute(f"""SELECT *, {self.EPSG_QUERY} FROM {self.JOINED_TABLE} WHERE {self.PRIMARY_KEY}=%s""", (self.pk,))
            record = self.sql_obj.cursor.fetchone()
            # udpate to the full record in case it was a partial record
            self.read_record(**record)
        return self.sql_obj.conn, self.sql_obj.cursor

    def release_lock(self):
        if self.has_lock():
            self.sql_obj.conn.commit()
            self._set_running(False, set_all=True)  # this cleans up any combines that crashed and left the running flag on
            self.sql_obj.conn.commit()  # autocommit is turned off due to locking needs
            self.sql_obj.conn.close()
            self.sql_obj = None

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
        if not self.sql_obj:
            raise BaseLockException("No cursor to update the table")
        pg_update(self.sql_obj.cursor, self.SOURCE_TABLE, where, **kwargs)

    def update_table_record(self, **kwargs):
        where = {self.PRIMARY_KEY: self.pk}
        self.update_table(where, **kwargs)


class ResolutionTileInfo(TileInfo):
    SOURCE_TABLE = "spec_resolutions"
    RESOLUTION = "resolution"
    CLOSING_DISTANCE = 'closing_distance'
    RESOLUTION_ID = 'r_id'
    TILE_ID = 'tile_id'
    PRIMARY_KEY = RESOLUTION_ID
    JOINED_TABLE = f"{SOURCE_TABLE} R JOIN {TileInfo.SOURCE_TABLE} TI ON (R.{TILE_ID}=TI.{TileInfo.PRIMARY_KEY})"
    LOCK_QUERY = f"""(SELECT (R.ctid NOT IN (SELECT ctid FROM {SOURCE_TABLE} FOR UPDATE SKIP LOCKED))) as {TileInfo.IS_LOCKED}"""
    EPSG_QUERY = 'ST_SRID(geometry_buffered)'
    GEOMETRY_BUFFERED = 'geometry_buffered'

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
        self.geometry_buffered = review_tile.get(self.GEOMETRY_BUFFERED, None)

    def __repr__(self):
        return super().__repr__()+f"_{self.resolution}m"

    @property
    def RUNNING(self):
        return self.export.RUNNING

    def hash_id(self):
        return hash_id(super().hash_id(),  self.resolution)

    def get_related_combine_info(self, sql_info):
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
        combine_ids = self.get_related_combine_ids(sql_obj)
        combine_tiles = CombineTileInfo.get_full_records(sql_obj, combine_ids)
        return combine_tiles

    def get_related_combine_ids(self, sql_info):
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
        sql_obj.cursor.execute(f"""SELECT {CombineTileInfo.COMBINE_ID} FROM {CombineTileInfo.SOURCE_TABLE} WHERE {CombineTileInfo.RESOLUTION_ID}=%s""", (self.pk,))
        records = sql_obj.cursor.fetchall()
        return [r[CombineTileInfo.COMBINE_ID] for r in records]

    def acquire_lock_and_combine_locks(self, conn_info: ConnectionInfo):
        # lock the current record in the spec_resolutions table and set up a connection+cursor
        self.acquire_lock(conn_info)
        # lock all the spec_combines that this resolution would affect
        all_ids = self.get_related_combine_ids(self.sql_obj)
        self.sql_obj.cursor.execute(f"""SELECT * FROM {CombineTileInfo.SOURCE_TABLE} WHERE {CombineTileInfo.RESOLUTION_ID}=%s FOR UPDATE SKIP LOCKED""", (self.pk,))
        records = self.sql_obj.cursor.fetchall()
        if len(records) != len(all_ids):
            print(len(records),'vs', len(all_ids))
            print(all_ids)
            for n, rec in enumerate(records):
                print(rec)
            self.release_lock()
            raise BaseLockException(f"Failed to acquire all locks for {self} (PK={self.pk}) related combine records {all_ids}")
        return self.sql_obj.conn, self.sql_obj.cursor

    @property
    def geometry_buffered(self):
        return self._geom_buffered

    @geometry_buffered.setter
    def geometry_buffered(self, geom):
        # FIXME consolidate this with the TileInfo geometry setter function so only one conversion function exists
        if geom is None:
            self._geom_buffered = None
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
            self._geom_buffered = g
            self.raw_geom_buffered = geom
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

    def update_export_status(self, table=None):
        # trying to just save the c_id from the combine_spec_view and use that to update the combine_spec_bruty table
        # cursor.execute(
        #     f"""update {tablename} set ({self.OUT_OF_DATE},{self.SUMMARY})=(%s, %s)
        #     where ({self.PB},{self.UTM},{self.HEMISPHERE},{self.TILE},{self.DATUM},{self.LOCALITY},{self.RESOLUTION}, {self.DATATYPE},{self.NOT_FOR_NAV})=(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        #     (self.out_of_date, self.summary, self.pb, self.utm, self.hemi.upper(), self.tile, self.datum, self.locality, self.resolution, self.datatype, self.not_for_nav))
        self.sql_obj.cursor.execute(
            f"""update {table} set ({self.export.START_TIME},{self.export.END_TIME},{self.export.EXIT_CODE},{self.export.WARNINGS_LOG},
            {self.export.INFO_LOG},{self.export.TRIES})=(%s, %s, %s, %s, %s, %s) 
            where ({self.RESOLUTION_ID})=(%s)""",
            (self.export.start_time, self.export.end_time, self.export.exit_code, self.export.warnings_log, self.export.info_log, self.export.tries,
             self.res_tile_id))


class CombineTileInfo(ResolutionTileInfo):
    SOURCE_TABLE = "spec_combines"
    VIEW_TABLE = "view_individual_combines"

    COMBINE_ID = 'c_id'
    FOR_NAV = 'for_navigation'
    DATATYPE = 'datatype'
    RESOLUTION_ID = 'res_id'
    OUT_OF_DATE = "out_of_date"
    SUMMARY = "change_summary"
    PRIMARY_KEY = COMBINE_ID
    JOINED_TABLE = f"""{SOURCE_TABLE} C JOIN {ResolutionTileInfo.SOURCE_TABLE} R ON (C.{RESOLUTION_ID}=R.{ResolutionTileInfo.PRIMARY_KEY}) 
                        JOIN {TileInfo.SOURCE_TABLE} TI ON (R.{ResolutionTileInfo.TILE_ID}=TI.{TileInfo.PRIMARY_KEY})"""
    LOCK_QUERY = f"""(SELECT (C.ctid NOT IN (SELECT ctid FROM {SOURCE_TABLE} FOR UPDATE SKIP LOCKED))) as {TileInfo.IS_LOCKED}"""
    EPSG_QUERY = 'ST_SRID(geometry_buffered)'
    GEOMETRY = 'geometry_buffered'

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
        # self.res_foreign_key = review_tile.get(self.RESOLUTION_ID, None)  # this will be in the ResolutionTileInfo as res_tile_id

    def __repr__(self):
        return super().__repr__()+f"_{self.datatype}_{'nav' if self.for_nav else 'NOT_NAV'}"

    def hash_id(self):
        return hash_id(super().hash_id(),  self.datatype, self.for_nav)

    @property
    def RUNNING(self):
        return self.combine.RUNNING

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
    fingerprint: str


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
        self.sql_obj = None
        self.max_tries = max_tries
        self.allow_res = allow_res
        self.user_dtypes, self.user_res = None, None
        self.read_user_settings(self.allow_res)
        self.remaining_tiles = {}
        self.reconnect()

    def reconnect(self):
        # drop and reconnect to postgres as part of our refresh
        try:
            self.sql_obj.conn.close()
        except Exception:
            pass
        self.sql_obj = SQLConnectionAndCursor.from_info(self.config)

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
            for tile_info in iterate_combine_table(self.config, needs_to_process=needs_combining, max_retries=self.max_tries, sql_info=self.sql_obj):
                res = tile_info.resolution
                if self.user_res and res not in self.user_res:
                    continue
                self.remaining_tiles[tile_info.hash_id()] = tile_info
        if self.process_exports:
            for tile_info in iterate_export_table(self.config, needs_to_process=needs_exporting, max_retries=self.max_tries, sql_info=self.sql_obj):
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
        use_priority = tile.priority if not tile.is_locked else TileManager.RUNNING_PRIORITY
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
                            yield tile

    def details_str(self):
        details = []
        details.append("Remaining tiles:")
        for priority in sorted(self.priorities.keys(), reverse=True):
            details.append(f"\tPriority {priority}:")
            for pb, pb_vals in self.priorities[priority].items():
                details.append(f"\t\tProduction Branch {pb}:")
                for key, tile in pb_vals.items():
                    operation = "Combine" if isinstance(tile, CombineTileInfo) else "Export"
                    details.append(f"\t\t\t{operation} {tile}")
        return "\n".join(details)

    def remove(self, tile):
        # The processing loop removes the tile from the remaining_tiles list as part of its processing loop.
        # but if the loop finishes then it refreshes the list and the tile may reappear in the list.
        try:
            hash = tile.hash_id()
            del self.remaining_tiles[hash]
            # if the start/end times got modified then a running tile may now look finished or vice versa, so check both places in our dictionary
            # No longer modifying this - we are doing it as an iterator instead
            # try:
            #     del self.priorities[self._revised_priority(tile)][tile.pb][hash]
            # except KeyError:
            #     del self.priorities[self.RUNNING_PRIORITY][tile.pb][hash]
        except KeyError:
            pass


def create_world_db(root_path, tile_info: TileInfo, log_level=logging.INFO):
    full_path = pathlib.Path(root_path).joinpath(tile_info.bruty_db_name())

    try:  # see if there is an exising Bruty database
        db = WorldDatabase.open(full_path, log_level=log_level)
    except FileNotFoundError:  # create an empty bruty database
        epsg = tile_info.epsg
        if tile_info.geometry_buffered is not None:
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

def lock_column_query(table_name, alias, do_lookup=True, col_name=TileInfo.IS_LOCKED):
    if do_lookup:
        # the R comes from the table alias in the execute statement below

        lock_column = f"""(SELECT ({alias}.ctid NOT IN (SELECT ctid FROM {table_name} FOR UPDATE SKIP LOCKED))) as {col_name}"""
    else:
        lock_column = f"False as {col_name}"
    return lock_column

def get_export_records(config, needs_to_process=False, max_retries=3, sql_info=None):
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
    if sql_info is None:
        sql_obj = SQLConnectionAndCursor(config)
    else:
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
    conditions = basic_conditions(config)
    if needs_to_process:
        conditions.append(needs_processing_condition(ExportOperation, max_retries=max_retries))
    where_clause = "\nAND ".join(conditions)  # a comment at the end was making this fail, so put a leading newline
    if where_clause:
        where_clause = " WHERE " + where_clause
    # This is the tiles with their resolutions and the SRID is based on geometry_buffered
    # SELECT R.*, ST_SRID(geometry_buffered), TI.*, {lock_query}
    # Querying the geometry is very slow - do a query when we need that specific record
    # and doing the time check in python is faster than doing it in SQL
    # @NOTE we must use the same alias for the spec_resolutions table as the JOINED_TABLE string since that is what the LOCK_QUERY is looking for
    # Making a connection to the database is taking about 3 seconds while the query is taking 0.2 seconds -- so adding an option to supply and open cursor
    sql_obj.cursor.execute(f"""SELECT {TileInfo.TILE}, {TileInfo.UTM}, {TileInfo.PB},{TileInfo.DATUM}, {TileInfo.LOCALITY}, {TileInfo.HEMISPHERE},
                            {TileInfo.EXPORT_PUBLIC}, {TileInfo.EXPORT_INTERNAL}, {TileInfo.EXPORT_NAVIGATION}, {TileInfo.PRIORITY}, {TileInfo.BUILD},
                            {ExportOperation.START_TIME}, {ExportOperation.END_TIME}, {ExportOperation.EXIT_CODE}, {ExportOperation.TRIES}, {ExportOperation.REQUEST_TIME},
                            {ResolutionTileInfo.RESOLUTION}, {ResolutionTileInfo.CLOSING_DISTANCE}, {ResolutionTileInfo.PRIMARY_KEY}, 
                            {ResolutionTileInfo.LOCK_QUERY} 
                       FROM spec_resolutions R 
                       JOIN spec_tiles TI ON (R.tile_id = TI.t_id)
                       {where_clause}
                       ORDER BY priority DESC, tile ASC
                       """)
    records = sql_obj.cursor.fetchall()
    return records


def get_combine_records(config, needs_to_process=False, get_lock_status=True, max_retries=3, sql_info=None):
    """ Read the NBS postgres tile_specifications database for the bruty_tile table and, if not existing,
    create Bruty databases for the area of interest for the polygon listed in the postgres table.

    Parameters
    ----------
    config
    needs_to_process
        This is very slow in SQL, so it's better to do this in python
    get_lock_status
    max_retries
    conn_info_or_cursor
        If None, then the connection will be made based on the config settings.
        If a dict cursor is supplied then it will be used.
        If a ConnectionInfo object, then a connection will be made based on that object.

    Returns
    -------

    """
    dtypes = parse_multiple_values(config.get('dtypes', ""))
    if sql_info is None:
        sql_obj = SQLConnectionAndCursor(config)
    else:
        sql_obj = SQLConnectionAndCursor.from_info(sql_info)
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
    # This is very similar to the combine_spec_view except we are getting the geometry_buffered and its SRID which are not in the view because of QGIS performance
    # Actually it seems that when querying the whole table it's significantly faster to get less columns, so just get the critical ones then query the rest when needed.
    # The killer is the geometry, a query without geometry is taking 0.5 seconds, with geometry it's taking 25 seconds
    # Using the time comparison as SQL would take a 0.6 second query to 13 seconds
    # Making a connection to the database is taking about 3 seconds while the query is taking 0.2 seconds -- so adding an option to supply and open cursor
    # @NOTE we must use the same alias for the spec_resolutions table as the JOINED_TABLE string since that is what the LOCK_QUERY is looking for
    sql_obj.cursor.execute(f"""SELECT {TileInfo.TILE}, {TileInfo.UTM}, {TileInfo.PB},{TileInfo.DATUM}, {TileInfo.LOCALITY}, {TileInfo.HEMISPHERE}, 
                            {TileInfo.PRIORITY}, {TileInfo.BUILD},
                            {CombineOperation.START_TIME}, {CombineOperation.END_TIME}, {CombineOperation.EXIT_CODE}, {CombineOperation.TRIES}, {CombineOperation.REQUEST_TIME},
                            {ResolutionTileInfo.RESOLUTION}, 
                            {CombineTileInfo.DATATYPE}, {CombineTileInfo.FOR_NAV}, {CombineTileInfo.PRIMARY_KEY},  
                            {CombineTileInfo.LOCK_QUERY} 
                       FROM {CombineTileInfo.SOURCE_TABLE} C 
                       JOIN {ResolutionTileInfo.SOURCE_TABLE} R ON (C.res_id = R.r_id) 
                       JOIN {TileInfo.SOURCE_TABLE} TI ON (R.tile_id = TI.t_id)
                       {where_clause}
                       ORDER BY priority DESC, tile ASC
                       """)
    records = sql_obj.cursor.fetchall()
    return records

def iterate_export_table(config, needs_to_process=False, max_retries=3, sql_info=None):
    records = get_export_records(config, max_retries=max_retries, sql_info=sql_info)
    for review_tile in records:
        info = ResolutionTileInfo(**review_tile)
        if not needs_to_process or (info.export.needs_processing() and (info.export.tries is None or info.export.tries < max_retries)):
            yield info

def iterate_combine_table(config, needs_to_process=False, max_retries=3, sql_info=None):
    records = get_combine_records(config, max_retries=max_retries, sql_info=sql_info)
    for review_tile in records:
        info = CombineTileInfo(**review_tile)
        if not needs_to_process or (info.combine.needs_processing() and (info.combine.tries is None or info.combine.tries < max_retries)):
            yield info

