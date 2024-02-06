import json
import os
import pathlib
import shutil
import tempfile
import pickle
import logging
import sqlite3
import time
import multiprocessing
import importlib
from functools import partial
# import logging
import gc
from datetime import datetime
from collections.abc import MutableMapping

import numpy

try:
    import fiona
except ModuleNotFoundError:
    fiona = None

from osgeo import gdal, ogr  # , osr

from HSTB.drivers import bag
from nbs_utils.points_utils import mmap_from_npz, iterate_points_file
from nbs.bruty.utils import merge_arrays, get_crs_transformer, onerr, tqdm, make_gdal_dataset_area, get_epsg_or_wkt, \
    calc_area_array_params, iterate_gdal_image, transform_rect  # , add_uncertainty_layer, compute_delta_coord, merge_array, make_gdal_dataset_size
from nbs.bruty.raster_data import TiffStorage, LayersEnum, affine, inv_affine, affine_center, RasterData  # , arrays_dont_match
# noinspection PyUnresolvedReferences
from nbs.bruty.history import DiskHistory, AccumulationHistory, RasterHistory
from nbs.bruty.abstract import VABC  # , abstractmethod
from nbs.bruty.tile_calculations import TMSTilesMercator, GoogleTilesMercator, GoogleTilesLatLon, UTMTiles, LatLonTiles, TilingScheme, \
    ExactUTMTiles, ExactTilingScheme
from nbs.bruty import morton
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
from nbs.configs import get_logger, set_file_logging, make_family_of_logs  # , iter_configs, log_config, parse_multiple_values

geo_debug = False
_debug = False
NO_OVERRIDE = -1

NO_LOCK = True
VERSION = (1, 0, 0)
__version__ = '.'.join(map(str, VERSION))

# Use this when I know only one process is running.
from nbs.bruty.nbs_no_locks import LockNotAcquired, BaseLockException, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, current_address, Lock
"""
#----------------------------------------------------
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Binary
import sqlalchemy

engine = sqlalchemy.create_engine(r'sqlite+pysqlite:///C:\Data\nbs\wdb_metadata.sqlite')
Base = automap_base()
Base.prepare(engine, reflect=True)
session = Session(engine)
included = Base.classes.reinserts
res = session.query(included).first()

metadata_three = MetaData();
my_reflected_table = Table("reinserts", metadata_three, Column("tiles", sqlalchemy.PickleType), autoload_with=engine)
with engine.begin() as conn:
    res= conn.execute(my_reflected_table.select())
    for row in res:
        print(row.nbs_id, row.tiles)
#----------------------------------------------------
from sqlalchemy import create_engine
engine = create_engine('sqlite:///:memory:', echo=True)
from sqlalchemy import Table, Column, Integer, String, MetaData, ForeignKey, Binary
metadata_obj = MetaData()
users = Table('users', metadata_obj,
    Column('id', Integer, primary_key=True),
    Column('name', String),
    Column('fullname', String),
    Column('test', Binary),
)
metadata_obj.create_all(engine)
#----------------------------------------------------
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
import sqlalchemy

engine = sqlalchemy.create_engine(f'postgresql+psycopg2://{user}:{password}!@{hostname}:{port}/metadata')
Base = automap_base()
Base.prepare(engine, reflect=True)

pbc19 = Base.classes.pbc_utm19n_mllw
session = Session(engine)

res = session.query(pbc19).first()
res.decay_score
res.script_to_filename
res.from_filename
"""


def use_locks(port):
    global LockNotAcquired, BaseLockException, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, NO_LOCK, current_address, Lock
    if port is None or port == "":
        from nbs.bruty.nbs_no_locks import LockNotAcquired, BaseLockException, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, \
            SqlLock, NameLock, current_address, Lock
        NO_LOCK = True
    else:
        from nbs.bruty.nbs_locks import LockNotAcquired, BaseLockException, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, \
            SqlLock, NameLock, start_server, current_address, Lock
        start_server(int(port))
        NO_LOCK = False


class SQLDataField:
    def __init__(self, name, sql_type, conv_to_python=None, conv_to_sql=None):
        self.name = name
        self.sql_type = sql_type
        self.conv_to_python = conv_to_python
        self.conv_to_sql = conv_to_sql

    def to_python(self, val):
        return val if self.conv_to_python is None else self.conv_to_python(val)

    def to_sql(self, val):
        return val if self.conv_to_sql is None else self.conv_to_sql(val)

    def sql_create_str(self):
        return self.name + " " + self.sql_type


class SQLDataRecord:
    pass


def make_data_class(fields):
    class SQLDataRecordType(SQLDataRecord):
        """ This class allows accessing a list of data from an sql call by attribute notation.
        So instead of rec[1] you would use rec.tiles or rec._data['tiles']
        """
        _fields = fields
        _name_to_index = {fld.name: ind for ind, fld in enumerate(fields)}
        _index_to_name = {ind: fld.name for ind, fld in enumerate(fields)}

        def __init__(self, data=None):
            if data:
                self._data = data
            else:
                self._data = [None] * len(self._fields)

        def __getattr__(self, key):
            if key in self.__dict__:
                return self.__dict__[key]
            else:
                try:
                    index = self._name_to_index[key]
                    val = self._data[index]
                except KeyError as e:
                    raise AttributeError(str(e))
                except IndexError as e:
                    raise AttributeError(str(e))
                return val

        def __setattr__(self, key, value):
            # translate self.whatever to self['whatever'], if it's not already an attribute (like self.data needs to be)
            if key[0] != "_" and key in self._name_to_index:
                self._data[self._name_to_index[key]] = value
            else:
                return super().__setattr__(key, value)

        def to_python(self):
            data = [sql_data_field.to_python(v) for v, sql_data_field in zip(self._data, self._fields)]
            return self.__class__(data)

        def to_sql(self):
            return [sql_data_field.to_sql(v) for v, sql_data_field in zip(self._data, self._fields)]
    return SQLDataRecordType


class SqliteSurveyMetadata(MutableMapping):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    fields = tuple()  # override this with data structures, probably using the SQLDataField class

    def __init__(self, path_to_sqlite_db, tablename, primary_key, timeout=120):
        self.tablename = tablename
        if primary_key.lower() in ("rowid", "oid", "_rowid_"):
            self._uses_oid = True
            self._primary_index = 0
        else:
            self._uses_oid = False
            for idx, fld in enumerate(self.fields):
                if fld.name == primary_key:
                    self._primary_index = idx
                    break
        try:
            self._primary_index
        except AttributeError:
            raise NameError(f"{primary_key} not found in field names")
        self.field_str = ", ".join([f"{fld.name}" for fld in self.fields])
        self.question_mark_str = ",".join(["?" for fld in self.fields])
        self.primary_key = primary_key
        self.path_to_sqlite_db = path_to_sqlite_db
        # Connection to a SQLite database
        self.conn = sqlite3.connect(path_to_sqlite_db, timeout=timeout)
        # Cursor object
        self.cur = self.conn.cursor()
        # sqliteDropTable = "DROP TABLE IF EXISTS parts"
        if not self.cur.execute(f'''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{self.tablename}' ''').fetchone()[0]:
            self.create_table()
        self.data_class = make_data_class(self.fields)

    def create_table(self):
        # Create a SQLite table with primary key
        # sqlite_create_table = "CREATE TABLE included(nbs_id integer PRIMARY KEY, survey_path text, tiles bytes, epsg integer, reverse_z integer, "\
        # "survey_score float, flag integer, dformat bytes, mtime real)"
        sqlite_create_table = ", ".join([f"{fld.name} {fld.sql_type}" for fld in self.fields])
        # Execute the create table statement
        self.cur.execute(f"CREATE TABLE {self.tablename} ({sqlite_create_table})")

    def to_python(self, rec):
        if not isinstance(rec, SQLDataRecord):
            rec = self.data_class(rec)
        converted = rec.to_python()
        # data = [sql_data_field.to_python(v) for v, sql_data_field in zip(rec, self.fields)]
        return converted
        # return (rec[0], rec[1], pickle.loads(rec[2]), rec[3], bool(rec[4]), rec[5], rec[])

    def to_sql(self, rec):
        if not isinstance(rec, SQLDataRecord):
            rec = self.data_class(rec)
        return rec.to_sql()
        # return [sql_data_field.to_sql(v) for v, sql_data_field in zip(rec, self.fields)]

    def get_records(self, field, val):
        if self._uses_oid:
            raw_records = self.cur.execute(f"select oid, * from {self.tablename} where {field} = (?)", [val]).fetchall()
            proc_records = []
            for record in raw_records:
                val = self.to_python(record[1:])
                val.oid = record[0]
                proc_records.append(val)
        else:
            raw_records = self.cur.execute(f"select * from {self.tablename} where {field} = (?)", [val]).fetchall()
            proc_records = [self.to_python(record) for record in raw_records]
        return proc_records

    def __getitem__(self, key):
        recs = self.cur.execute(f"select * from {self.tablename} where {self.primary_key} = (?)", [key]).fetchall()
        if len(recs) > 1:
            raise IndexError(f"More than one value found for {key} in sqlite metadata")
        elif len(recs) == 0:
            raise KeyError(f"No key {key}")
        else:
            rec = self.to_python(recs[0])
            # @TODO return the whole thing?  why remvoe the primary key unless it's a hidden (implicit) oid
            # if not self._uses_oid:
            #     retval = (*rec[:self._primary_index], *rec[self._primary_index + 1:])  # remove the key from the record
            # else:
            #     retval = tuple(rec)
            # return retval
            return rec

    def __setitem__(self, key, val):
        # self.conn.isolation_level = 'EXCLUSIVE'
        # self.conn.execute('BEGIN EXCLUSIVE')
        with SqlLock(self.conn):
            record = self.to_sql(val)  # converts to list from SQLDataRecord class
            if not self._uses_oid:
                # replace the key in the data with the passed in key
                record[self._primary_index] = key
                # @todo - see if there is a significant performance penalty to doing update and insert if that fails,
                #   if not then switch to always doing that
                # insert first and update if it fails
                try:
                    # cur.execute("insert into included values (?, ?, ?, ?, ?, ?, ?, ?, ?)", rec)
                    self.cur.execute(f"insert into {self.tablename} values ({self.question_mark_str})", record)
                except sqlite3.IntegrityError:
                    # update
                    record.append(key)
                    # cur.execute("""update included set (survey_path, tiles, epsg, reverse_z, survey_score, flag, dformat, mtime)=(?,?,?,?,?,?,?,?) where nbs_id=(?)""",
                    self.cur.execute(
                        f"""update {self.tablename} set ({self.field_str})=({self.question_mark_str}) where {self.primary_key}=(?)""",
                        record)
            else:
                # update first and insert if the update fails.  Insert would always succeed and make new IDs
                uprecord = record + [key]
                # cur.execute("""update included set (survey_path, tiles, epsg, reverse_z, survey_score, flag, dformat, mtime)=(?,?,?,?,?,?,?,?) where nbs_id=(?)""",
                worked = self.cur.execute(
                    f"""update {self.tablename} set ({self.field_str})=({self.question_mark_str}) where {self.primary_key}=(?)""",
                    uprecord).rowcount
                if worked < 1:
                    # cur.execute("insert into included values (?, ?, ?, ?, ?, ?, ?, ?, ?)", rec)
                    self.cur.execute(f"insert into {self.tablename} values ({self.question_mark_str})", record)
            # self.conn.commit()
            # self.conn.isolation_level = 'DEFERRED'

    def add_oid_record(self, val):
        self.__setitem__(None, val)
        return self.cur.lastrowid

    def __delitem__(self, key):
        # self.conn.isolation_level = 'EXCLUSIVE'
        # self.conn.execute('BEGIN EXCLUSIVE')
        with SqlLock(self.conn):
            self.cur.execute(f"""delete from {self.tablename} where {self.primary_key}=(?)""", [key])
        # self.conn.commit()
        # self.conn.isolation_level = 'DEFERRED'

    def __iter__(self):
        for key in self.cur.execute(f"""select {self.primary_key} from {self.tablename}""").fetchall():
            yield key[0]

    def __len__(self):
        return self.cur.execute(f"""SELECT COUNT(*) from {self.tablename}""").fetchone()[0]


class IncludedMetadata(SqliteSurveyMetadata):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    fields = (SQLDataField('nbs_id', 'integer PRIMARY KEY'),
              SQLDataField('survey_path', 'text', conv_to_sql=str),  # pathlib.Path needs conversion to str
              SQLDataField('tiles', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
              SQLDataField('sorting_metadata', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
              SQLDataField('epsg', 'integer'),
              SQLDataField('reverse_z', 'integer', bool, int),
              SQLDataField('survey_score', 'real'),
              SQLDataField('flag', 'integer'),
              SQLDataField('dformat', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
              SQLDataField('mtime', 'real'),
              SQLDataField('transaction_id', 'integer'),
              )

    def __init__(self, path_to_sqlite_db, tablename, *args, **kwargs):
        super().__init__(path_to_sqlite_db, tablename, *args, **kwargs)
        if not self.cur.execute(f"SELECT COUNT(*) AS CNTREC FROM pragma_table_info('{tablename}') WHERE name='transaction_id'").fetchone()[0]:
            self.cur.execute(f'ALTER TABLE {tablename} ADD column transaction_id integer;')

    def create_table(self):
        super().create_table()
        # Create a secondary key on the name column
        create_secondary_index = f"CREATE INDEX index_{self.tablename}_survey_path ON {self.tablename}(survey_path)"
        self.cur.execute(create_secondary_index)


class IncludedIds(IncludedMetadata):
    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "included", "nbs_id")


class IncludedSurveys(IncludedMetadata):
    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "included", "survey_path")


class StartedMetadata(SqliteSurveyMetadata):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    fields = (SQLDataField('nbs_id', 'integer PRIMARY KEY'),
              SQLDataField('survey_path', 'text', conv_to_sql=str),  # converts pathlib.Path to str
              SQLDataField('tiles', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
              SQLDataField('mtime', 'real'),
              SQLDataField('transaction_id', 'integer'),
              )

    def __init__(self, path_to_sqlite_db, tablename, *args, **kwargs):
        super().__init__(path_to_sqlite_db, tablename, *args, **kwargs)
        # update the table structure if an old schema is found
        if not self.cur.execute(f"SELECT COUNT(*) AS CNTREC FROM pragma_table_info('{tablename}') WHERE name='transaction_id'").fetchone()[0]:
            self.cur.execute(f'ALTER TABLE {tablename} ADD column transaction_id integer;')


class StartedIds(StartedMetadata):
    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "started", "nbs_id")


class StartedSurveys(StartedMetadata):
    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "started", "survey_path")


class TransactionGroups(SqliteSurveyMetadata):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    # use the sqlite built in oid
    fields = (
        # SQLDataField('oid', 'int PRIMARY KEY'),  # this would actually be auto-supplied by sqlite but sqlalchemy needs an explicit alias
        SQLDataField('ttype', 'TEXT'),
        SQLDataField('ttime', 'TEXT', datetime.fromisoformat, datetime.isoformat),
        SQLDataField('process_id', 'integer'),
        SQLDataField('finished', 'integer'),
        SQLDataField('user_quit', 'integer'),
    )

    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "transaction_groups", "oid")
        # update the table structure if an old schema is found
        if not self.cur.execute(f"SELECT COUNT(*) AS CNTREC FROM pragma_table_info('transaction_groups') WHERE name='process_id'").fetchone()[0]:
            self.cur.execute(f'ALTER TABLE transaction_groups ADD column process_id integer;')
            self.cur.execute(f'ALTER TABLE transaction_groups ADD column finished integer;')
            self.cur.execute(f'ALTER TABLE transaction_groups ADD column user_quit integer;')
            self.cur.execute("update transaction_groups set process_id =  0;")
            self.cur.execute("update transaction_groups set finished =  0;")
            self.cur.execute("update transaction_groups set user_quit =  0;")

    def set_finished(self, oid):
        with SqlLock(self.conn):
            self.cur.execute(f"""update {self.tablename} set (finished)=(1) where {self.primary_key}=(?)""", [oid])

    def set_quit(self, oid):
        with SqlLock(self.conn):
            self.cur.execute(f"""update {self.tablename} set (user_quit)=(1) where {self.primary_key}=(?)""", [oid])


class CompletionCodes(SqliteSurveyMetadata):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    # use the sqlite built in oid
    fields = (
        SQLDataField('fingerprint', 'TEXT PRIMARY KEY'),
        SQLDataField('ttime', 'TEXT', datetime.fromisoformat, datetime.isoformat),
        SQLDataField('code', 'integer'),
    )

    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "completion_codes", "fingerprint")


class StartFinishRecords(SqliteSurveyMetadata):  # fixme - rename one of these transaction classes so it's clearer
    def unfinished_records(self):
        return self.get_records("finished", 0)

    def finished_records(self):
        return self.get_records("finished", 1)

    def is_finished(self, oid):
        return self.cur.execute(f"select finished from {self.tablename} where oid = (?)", [oid]).fetchone()[0]

    def set_finished(self, oid):
        with SqlLock(self.conn):
            self.cur.execute(f"""update {self.tablename} set (finished)=(1) where {self.primary_key}=(?)""", [oid])

    def set_started(self, oid):
        with SqlLock(self.conn):
            self.cur.execute(f"""update {self.tablename} set (started)=(1) where {self.primary_key}=(?)""", [oid])


class ReinsertInstructions(StartFinishRecords):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    # use the sqlite built in oid
    fields = (
        # SQLDataField('oid', 'int PRIMARY KEY'),  # this would actually be auto-supplied by sqlite byt sqlalchemy needs an explicit alias
        SQLDataField('nbs_id', 'integer'),
        SQLDataField('tiles', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
        SQLDataField('started', 'integer'),
        SQLDataField('finished', 'integer'),
    )

    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "reinserts", "oid")


class AreaOfInterest(StartFinishRecords):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    # use the sqlite built in oid
    fields = (
        # SQLDataField('oid', 'int PRIMARY KEY'),  # this would actually be auto-supplied by sqlite byt sqlalchemy needs an explicit alias
        SQLDataField('x', 'real'),
        SQLDataField('y', 'real'),
        SQLDataField('poly', 'integer'),
    )

    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "aoi", "oid")


class SurveyRemovals(StartFinishRecords):
    """ Class to store survey metadata into an sqlite3 database.  This is to replace the pickle file that replaced the json.
    The file will have two tables to store the dictionaries from included_ids, started_ids, included_surveys, started_surveys
    """
    # use the sqlite built in oid
    fields = (
        # SQLDataField('oid', 'int PRIMARY KEY'),  # this would actually be auto-supplied by sqlite byt sqlalchemy needs an explicit alias
        SQLDataField('nbs_id', 'integer'),
        SQLDataField('tiles', 'BLOB', pickle.loads, pickle.dumps),  # sqlalchemy doesn't accept 'bytes' so use SQL type of BLOB
        SQLDataField('affects', 'text'),
        SQLDataField('transaction_id', 'integer'),
        SQLDataField('started', 'integer'),
        SQLDataField('finished', 'integer'),
    )

    def __init__(self, path_to_sqlite_db):
        super().__init__(path_to_sqlite_db, "removals", "oid")


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
            self._make_logger()
        self.to_file()  # store parameters so it can be loaded back from disk

    def _make_logger(self):
        log_name = __name__ + "." + self.data_path.name
        self.LOGGER = get_logger(log_name)
        make_family_of_logs(log_name, self.data_path.joinpath("log"), remove_other_file_loggers=False,
                            log_format=f'[%(asctime)s] {__name__} %(levelname)-8s: %(message)s')

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
        self._make_logger()
        # do any version updates here
        self._loaded_from_version = json_dict['version']
        self._version = json_dict['version']

    def iterate_filled_tiles(self):
        for tx_dir in tqdm(list(os.scandir(self.data_path)), desc='X directories', mininterval=.7, leave=False):
            if tx_dir.is_dir():
                for ty_dir in tqdm(list(os.scandir(tx_dir)), desc='Y directories', mininterval=.7, leave=False):
                    if ty_dir.is_dir():
                        try:
                            tx, ty = self.str_to_tile_index(tx_dir.name, ty_dir.name)
                        except ValueError:
                            print(f"non-tile directory in the database {tx_dir.name}, {ty_dir.name}")
                        else:
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
        # Use the same tile_scheme (which is just geographic parameters) with an AccumulationHistory to allow multiple passes
        # on the same dataset to be added to the main database at one time
        use_history = AccumulationHistory
        new_db = WorldTilesBackend(self.tile_scheme, use_history, self.storage_class, self.data_class, data_path)
        return new_db

    def remove_accumulation_db(self, storage_db):
        path_to_del = storage_db.data_path
        # need to release the logger that writes into the accumulation db before deleting the files
        for h in storage_db.LOGGER.handlers:
            if isinstance(h, logging.FileHandler):
                h.close()
        shutil.rmtree(path_to_del, onerror=onerr)

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

    def get_tile_history_by_index(self, tx, ty, no_create=False):
        hist_path = self.get_history_path_by_index(tx, ty)
        if no_create and not self.storage_class.exists(hist_path):
            # raise FileNotFoundError(f"{hist_path} was not found")
            history = None
        else:
            history = self.history_class(self.storage_class(self.data_class, hist_path))
            if history.min_x is None:  # "min_x" not in history.get_metadata():
                lx, ly, ux, uy = self.tile_scheme.tile_index_to_xy(tx, ty)
                history.set_corners(lx, ly, ux, uy)
            # @ todo remove this after rebuild/check of existing PBG14,15,16 data
            if "epsg" not in history.get_metadata():
                history.set_epsg(self.epsg)
        return history

    def iter_tiles(self, x, y, x2, y2, no_create=False):
        for tx, ty in self.get_tiles_indices(x, y, x2, y2):
            history = self.get_tile_history_by_index(tx, ty, no_create=no_create)
            if history is not None:
                yield tx, ty, history

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
    def __init__(self, res_x, res_y, utm_epsg, history_class, storage_class, data_class, data_path, zoom_level=13, offset_x=0, offset_y=0):
        tile_scheme = ExactUTMTiles(res_x, res_y, zoom=zoom_level, epsg=utm_epsg, offset_x=offset_x, offset_y=offset_y)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class GoogleMercatorTileBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = GoogleTilesMercator(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class TMSMercatorTileBackend(WorldTilesBackend):
    def __init__(self, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = TMSTilesMercator(zoom=zoom_level)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


def poly_from_pts(area_of_interest, transformer=None):
    """ Pass a list or array of points in order to create a polygon OR two points representing two corners of a rectangle.
    returns an ogr.Geometry
    """
    if len(area_of_interest) == 2:  # two corners
        # build corners from the provided bounds
        x1, y1 = area_of_interest[0][:2]
        x2, y2 = area_of_interest[1][:2]
        ul = (x1, y1)
        ur = (x2, y1)
        lr = (x2, y2)
        ll = (x1, y2)
        area_of_interest = [ul, ur, lr, ll]
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for pt in area_of_interest:
        x, y = pt[:2]
        if transformer:
            x, y = transformer.transform(x, y)
        ring.AddPoint(float(x), float(y))  # float conversion to prevent type error if numpy int is passed in.
    # list() used to make list, tuple or numpy.array compares all work
    if list(area_of_interest[0][:2]) != list(area_of_interest[-1][:2]):  # connect the last point to the first to have a completed polygon
        x, y = area_of_interest[0][:2]
        if transformer:
            x, y = transformer.transform(x, y)
        ring.AddPoint(float(x), float(y))
    geom = ogr.Geometry(ogr.wkbPolygon)
    geom.AddGeometry(ring)
    return geom


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
    METACLASS_NAME = "wdb_metadata.class"

    def __init__(self, backend, area_of_interest: (ogr.Geometry, list, numpy.array)=None):
        """
        Parameters
        ----------
        backend
            The desired storage and coordinate system backend, example:
            UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y, zoom_level=zoom)
        area_of_interest
            If not None, then incoming data will be checked against the area_of_interest and skipped if its bounding rectangle
            does not intersect the area_of_interest.

            Pass in: the two corners of a rectangle OR a list of points to make a polygon OR a premade ogr.Geometry
        """
        self.db = backend
        self.area_of_interest = area_of_interest
        pth = self.metadata_filename().with_suffix(".sqlite")
        self.write_aoi(pth)
        self.included_surveys = IncludedSurveys(pth)
        self.included_ids = IncludedIds(pth)
        self.started_surveys = StartedSurveys(pth)
        self.started_ids = StartedIds(pth)
        self.transaction_groups = TransactionGroups(pth)
        self.removed_ids = SurveyRemovals(pth)
        self.reinserts = ReinsertInstructions(pth)
        self.completion_codes = CompletionCodes(pth)
        self.to_file()

    @property
    def res_x(self):
        return self.db.tile_scheme.res_x

    @property
    def res_y(self):
        return self.db.tile_scheme.res_y

    @property
    def area_of_interest(self):
        return self._aoi

    @area_of_interest.setter
    def area_of_interest(self, val):
        if isinstance(val, ogr.Geometry):
            self._aoi = val
        elif not val:
            self._aoi = None
        else:
            self._aoi = poly_from_pts(val)
        if self._aoi:
            x1, x2, y1, y2 = self._aoi.GetEnvelope()
            self.tiles_of_interest = self.db.get_tiles_indices(x1, y1, x2, y2)
        else:
            self.tiles_of_interest = None

    @staticmethod
    def open(data_dir):
        """
        Parameters
        ----------
        data_dir

        Returns
        -------

        """
        data_path = WorldDatabase.metadata_filename_in_dir(data_dir)
        mode = 'rb'
        # if no_locks was used then this always succeeds and a different lock should be used earlier like in combine_tiles.py script
        with FileLock(data_path, mode, SHARED) as infile:  # this will wait for the file to be available
            data = pickle.load(infile)
        data['data_path'] = pathlib.Path(data_dir)  # override in case the user copied the data to a different path
        cls = eval(data['class'])
        obj = cls.__new__(cls)
        obj.from_dict(data)
        return obj

    def search_for_accum_db(self):
        possible_accum = []
        for tx_dir in tqdm(list(os.scandir(self.db.data_path)), desc='accum directories', mininterval=.7, leave=False):
            if tx_dir.is_dir():
                if "_accum" in str(tx_dir):
                    print("accumulation directory found at", tx_dir)
                    possible_accum.append(tx_dir)
        return possible_accum

    def search_for_bad_positioning(self, tolerance=50):
        possible_errors = {}
        for sid, val in tqdm(self.included_ids.items(), total=len(self.included_ids), desc='Surveys', mininterval=.7, leave=False):
            tiles = numpy.array(val.tiles)
            if len(tiles) > 1:
                xs = tiles[:, 0]
                xs.sort()
                max_x = numpy.diff(xs).max()
                ys = tiles[:, 1]
                ys.sort()
                max_y = numpy.diff(ys).max()
                if max_y > tolerance or max_x > tolerance:
                    possible_errors[sid] = (max_x, max_y, val.survey_path, val.tiles)
            elif not self.area_of_interest and len(tiles) == 0:  # if there is an area of interest then many surveys won't list tiles.
                print("survey had no tiles listed (no data)?", sid)
        for sid, (max_x, max_y, pth, tiles) in possible_errors.items():
            print("Tolerance exceeded", sid, max_x, max_y)  # , val[1])
            print("    " + pth)
        return possible_errors

    def validate(self):
        """
        Returns
        -------

        """
        # check each tile history for internal consistency - overall metadata vs the individual deltas
        tile_missing = {}
        tile_extra = {}
        contributor_missing = {}
        per_tile_contributors = {}
        for tx, ty, raster, meta in self.db.iterate_filled_tiles():
            tile_history = self.db.get_tile_history_by_index(tx, ty)
            missing, extra, duplicates, expected, actual = tile_history.validate()
            if missing:
                tile_missing[(tx, ty)] = missing
            if extra:
                tile_extra[(tx, ty)] = extra
            per_tile_contributors[(tx, ty)] = expected
        # check each survey in the included list against the tiles that are listed to make sure they not are missing
        for contrib in tqdm(self.included_ids.keys(), desc='Surveys', mininterval=.7, leave=False):
            for tx, ty in self.included_ids[contrib].tiles:
                try:
                    if str(contrib) not in per_tile_contributors[(tx, ty)]:  # use str since tiles are using json and keys come back as strings
                        contributor_missing.setdefault((tx, ty), [])
                        contributor_missing[(tx, ty)].append(contrib)
                except KeyError as e:
                    if not self.area_of_interest:  # if area of interest there are still some tiles being listed that are not in the AOI
                        raise e
        if tile_missing:
            print("The following contributors are listed in the tile's metadata but don't have a delta in the commit history"
                  "So they were added but and are in metadata.json but have no _00000_.tif data in the directory")
            for tile_xy, missing in tile_missing.items():
                print(f"Tile {tile_xy} was missing contributors from the history which were listed in that tile metadata {missing}")
        if tile_extra:
            print("The following contributors have _00000_.tif data in the directory but are NOT listed in the tile's metadata.json")
            for tile_xy, extra in tile_extra.items():
                print(f"Tile {tile_xy} had extra contributors in the history {extra}")
        if contributor_missing:
            print("The following contributors are listed in the world database metadata as being applied to a tile "
                  "but don't appear in that Tile's metadata.json, most likely a remove happened and they were not reinserted correctly.")
            for tile_xy, missing in contributor_missing.items():
                print(f"Based on the global insertion metadata, Tile {tile_xy} was missing contributors from the history {missing}")
        if not tile_missing and not tile_extra and not contributor_missing:
            print("Global contributors match individual tiles and tiles are internally consistent too")
        return tile_missing, tile_extra, contributor_missing

    @classmethod
    def metadata_filename_in_dir(cls, data_dir):
        data_path = pathlib.Path(data_dir).joinpath(cls.METACLASS_NAME)
        return data_path

    def metadata_filename(self):
        return self.metadata_filename_in_dir(self.db.data_path)

    def metadata_mode(self):
        return 'rb+'

    def add_transaction_group(self, ttype, ttime):
        with FileLock(self.metadata_filename(), self.metadata_mode(), EXCLUSIVE) as metadata_file:
            new_id = self.transaction_groups.add_oid_record((ttype, ttime))
        return new_id

    def to_file(self, data_path=None, locked_file=None):
        local_lock = None
        if locked_file is None:
            if not data_path:
                data_path = self.metadata_filename()
            # outfile = open(data_path, 'w')
            mode = 'wb'
            print('locking metadata for exclusive at ', datetime.now().isoformat())
            local_lock = FileLock(data_path, mode, EXCLUSIVE)  # this will wait for the file to be available
            local_lock.acquire()
            outfile = local_lock.fh
        else:
            outfile = locked_file
        pickle.dump(self.for_json(), outfile)
        if local_lock is not None:
            print('unlocking metadata ', datetime.now().isoformat())
            local_lock.release()

    def for_json(self):
        # @todo make a base class or mixin that auto converts to json,
        #    maybe list the attributes desired with a callback that handles anything not auto converted to json

        # for all formats save the type of class and module path
        json_dict = {'class': self.__class__.__name__,
                     'module': self.__class__.__module__,
                     }
        return json_dict

    def from_dict(self, json_dict):
        """Build the entire object from json"""
        self.db = WorldTilesBackend.from_file(json_dict['data_path'])
        self.update_metadata_from_dict(json_dict)

    def write_aoi(self, pth=None):
        if not pth:
            pth = self.metadata_filename().with_suffix(".sqlite")
        aoi = AreaOfInterest(pth)
        for j in aoi:
            del aoi[j]
        if self.area_of_interest:
            for npoly in range(self.area_of_interest.GetGeometryCount()):
                poly = self.area_of_interest.GetGeometryRef(npoly)
                for pt in poly.GetPoints():
                    aoi.add_oid_record([*pt[:2], npoly])

    def update_metadata_from_dict(self, json_dict):
        update = False
        try:
            if str(self.included_surveys.path_to_sqlite_db) != str(json_dict['data_path']):
                update = True
        except AttributeError:
            update = True
        if update:
            pth = self.metadata_filename().with_suffix(".sqlite")
            aoi = AreaOfInterest(pth)
            pts = numpy.array([(pt.x, pt.y, pt.poly) for pt in aoi.values()])
            if len(pts) > 0:
                if len(numpy.unique(pts[:, 2])) > 1:
                    raise ValueError("Multiple polygon areas are not supported yet")
                self.area_of_interest = poly_from_pts(pts)
            else:
                self.area_of_interest = None
            self.included_surveys = IncludedSurveys(pth)
            self.included_ids = IncludedIds(pth)
            self.started_surveys = StartedSurveys(pth)
            self.started_ids = StartedIds(pth)
            self.removed_ids = SurveyRemovals(pth)
            self.reinserts = ReinsertInstructions(pth)
            self.transaction_groups = TransactionGroups(pth)
            self.completion_codes = CompletionCodes(pth)

    def insert_survey(self, path_to_survey_data, override_epsg=NO_OVERRIDE, contrib_id=numpy.nan, compare_callback=None, reverse_z=False,
                      limit_to_tiles=None, force=False, survey_score=100, flag=0, dformat=None, transaction_id=-1, sorting_metadata=None, crop=False):
        """
        Parameters
        ----------
        path_to_survey_data
        override_epsg
        contrib_id
        compare_callback
        reverse_z
        limit_to_tiles
        force
        survey_score
        flag
        dformat
        transaction_id
        sorting_metadata

        Returns
        -------

        """
        self.db.LOGGER.debug(f"attempt to insert {path_to_survey_data}")
        done = False
        extension = pathlib.Path(str(path_to_survey_data).lower()).suffix
        if extension == '.bag':
            try:
                vr = bag.VRBag(path_to_survey_data, mode='r')
            except (bag.BAGError, ValueError):  # value error for single res of version < 1.6.  Maybe should change to BAGError in bag.py
                pass  # allow gdal to read single res
            else:
                self.insert_survey_vr(vr, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback, reverse_z=reverse_z,
                                      limit_to_tiles=limit_to_tiles, force=force, survey_score=survey_score, flag=flag,
                                      transaction_id=transaction_id, sorting_metadata=sorting_metadata)
                done = True
        if not done:
            if extension in ['.bag', '.tif', '.tiff']:
                self.insert_raster_survey(path_to_survey_data, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback,
                                        reverse_z=reverse_z, limit_to_tiles=limit_to_tiles, force=force, survey_score=survey_score, flag=flag,
                                        transaction_id=transaction_id, sorting_metadata=sorting_metadata)
                done = True
        if not done:
            if extension in ['.txt', '.npy', '.csv', '.npz', '.gpkg']:
                self.insert_points_survey(path_to_survey_data, override_epsg=override_epsg, contrib_id=contrib_id, compare_callback=compare_callback,
                                       reverse_z=reverse_z, limit_to_tiles=limit_to_tiles, force=force, survey_score=survey_score, flag=flag,
                                       dformat=dformat, transaction_id=transaction_id, sorting_metadata=sorting_metadata, crop=crop)
                done = True
        if not done:
            if extension in ['.csar', ]:
                # export to xyz
                # FIXME -- export points from csar and support LAS or whatever points file is decided on.
                # self.insert_txt_survey(path_to_survey_data, override_epsg=override_epsg, contrib_id=contrib_id,
                # compare_callback=compare_callback, reverse_z=reverse_z)
                done = True
                raise ValueError("Filename ends in csar which needs to be eported to a different format first")

    def _get_transformer(self, srs, override_epsg, path_to_survey_data, inv=False):
        try:
            epsg = get_epsg_or_wkt(srs)
        except Exception as e:
            if override_epsg == NO_OVERRIDE:
                raise e
            else:
                epsg = None

        if override_epsg != NO_OVERRIDE:
            if str(epsg) != str(override_epsg):
                self.db.LOGGER.warning(f"override {epsg} with {override_epsg} in {path_to_survey_data}")
                epsg = override_epsg
        # fixme - do coordinate transforms correctly
        if inv:
            transformer = get_crs_transformer(self.db.epsg, epsg)
        else:
            transformer = get_crs_transformer(epsg, self.db.epsg)
        if transformer:
            print("@todo - do transforms correctly with proj/vdatum etc")
        return transformer

    def iterate_pts_gpkg(self, path_to_survey_data, block_size=30000000):
        """ Reads a geopackage for layers that are point data (currently ignores rasters or other geometries) with an uncertainty field per point.

        Parameters
        ----------
        path_to_survey_data
        block_size : int
            maximum number of points to read at once

        Returns
        -------
        generator yielding (wkt, x, y, depth, uncertainty)

        """
        self.db.LOGGER.debug(f"insert geopackage {path_to_survey_data}")
        gpkg = gdal.OpenEx(path_to_survey_data)
        # geopackages can have both raster and vector layers, so we will iterate through them but currently only supporting point data from a gpkg
        for ilyr in range(gpkg.GetLayerCount()):
            lyr = gpkg.GetLayer(ilyr)
            if lyr.GetGeomType() == ogr.wkbPoint:
                srs = lyr.GetSpatialRef()
                wkt = srs.ExportToWkt()
                attr_depth[i] = feat['elevation']  # test what happens with attribute name not existing

                total_points = lyr.GetFeatureCount()
                for i, feat in enumerate(lyr):
                    if i % block_size == 0:  # allocate space for next block of data
                        npts = min(total_points - i, block_size)
                        x = numpy.zeros(npts, dtype=numpy.float64)
                        y = numpy.zeros(npts, dtype=numpy.float64)
                        depth = numpy.zeros(npts, dtype=numpy.float64)
                        uncertainty = numpy.zeros(npts, dtype=numpy.float64)
                    # read the point data and get the depth from the point Z or override with the elevation attribute
                    x[i], y[i], depth[i] = feat.GetGeometryRef().GetPoint()
                    try:
                        attr_depth = feat['elevation']
                        depth[i] = attr_depth
                    except KeyError:
                        pass
                    uncertainty[i] = feat['uncertainty']
                    # yield the block of data if it is full or we are at the end of the file
                    if i % block_size == block_size - 1 or i == total_points - 1:  # end of block or end of file
                        yield wkt, x, y, depth, uncertainty

    def insert_survey_as_outside_area_of_interest(self, path_to_survey_data, survey_score=100, flag=0, dformat=None, override_epsg: int = NO_OVERRIDE,
                                                  contrib_id=numpy.nan, reverse_z: bool = False, transaction_id=-1, sorting_metadata=None):
        """ Insert a survey into the metadata so it is known to have been evaluated but had no data of interest.
        Tiles will be stored as an empty list.
        """
        self.start_survey_insertion(path_to_survey_data, [], contrib_id, transaction_id)
        self.finished_survey_insertion(path_to_survey_data, [], contrib_id, override_epsg, reverse_z, survey_score, flag,
                                       dformat, transaction_id, sorting_metadata=sorting_metadata)

    # noinspection PyUnboundLocalVariable
    def insert_points_survey(self, path_to_survey_data, survey_score=100, flag=0, dformat=None, override_epsg: int = NO_OVERRIDE,
                          contrib_id=numpy.nan, compare_callback=None, reverse_z: bool = False, limit_to_tiles=None, force=False, transaction_id=-1,
                          sorting_metadata=None, block_size=30000000, crop=False):
        """ Reads a text file and inserts into the tiled database.
        The format parameter is passed to numpy.loadtxt and needs to have names of x, y, depth, uncertainty.

        Parameters
        ----------
        path_to_survey_data
            full path filename to read using numpy
        survey_score
            score to apply to data, if not a column in the data
        flag
            flags to apply to data, if not a column in the data
        dformat
            numpy dtype format to pass to numpy.loadtxt, default is [('y', 'f8'), ('x', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')]
        override_epsg
            epsg to use instead of what is in the file, use default of NO_OVERRIDE to use the file SpatialReferenceSystem
        contrib_id
            an integer of the contributor_id to store or numpy.nan if contributor is not being used
        compare_callback
            function to call for scoring or None to use the default comparison
        reverse_z
            will multiply the depth values by -1 before combining
        limit_to_tiles
        force
        transaction_id
        sorting_metadata
        block_size
        crop
            forces any tile indices that would be greater than the extents to fall on the edges
            aded so ENCs with too large an area will not raise an error, assuming an area of interest has been specified which will crop the data.

        Returns
        -------
        None
        """
        has_out_of_bounds_points = False
        self.db.LOGGER.debug(f"insert survey points {path_to_survey_data}")
        if force or contrib_id is None or contrib_id not in self.included_ids:
            skip_as_disjoint = False
            if str(path_to_survey_data).lower().endswith(".npz"):
                data = numpy.load(path_to_survey_data)
                wkt = str(data['wkt'])
                if wkt is not None and override_epsg == NO_OVERRIDE:
                    epsg = wkt
                    transformer = get_crs_transformer(epsg, self.db.epsg)
                else:
                    transformer = None
                try:
                    corners = data['minmax']
                except KeyError:
                    pass
                else:
                    bounds = poly_from_pts(corners, transformer)
                    if self.area_of_interest and not self.area_of_interest.Intersects(bounds):
                        skip_as_disjoint = True
            elif str(path_to_survey_data).lower().endswith(".gpkg"):
                # @TODO if we want to support geopackages with raster data we will have to split the point processing out of here
                gpkg = gdal.OpenEx(path_to_survey_data)
                # geopackages can have both raster and vector layers, so we will iterate through them but currently only supporting point data from a gpkg
                disjoints = []
                point_lyr_count = 0
                for ilyr in range(gpkg.GetLayerCount()):
                    lyr = gpkg.GetLayer(ilyr)
                    if lyr.GetGeomType() == ogr.wkbPoint:
                        point_lyr_count += 1
                        srs = lyr.GetSpatialRef()
                        wkt = srs.ExportToWkt()
                        if wkt is not None and override_epsg == NO_OVERRIDE:
                            epsg = wkt
                            transformer = get_crs_transformer(epsg, self.db.epsg)
                        else:
                            transformer = None
                        lx, ux, ly, uy = lyr.GetExtent()
                        bounds = poly_from_pts(numpy.array(((lx, ly), (ux, uy))), transformer)
                        lyr_is_disjoint = self.area_of_interest and not self.area_of_interest.Intersects(bounds)
                        disjoints.append(lyr_is_disjoint)
                if len(lyr_is_disjoint) == point_lyr_count and all(disjoints):
                    skip_as_disjoint = True

            if skip_as_disjoint:
                self.insert_survey_as_outside_area_of_interest(path_to_survey_data, survey_score, flag, dformat, override_epsg,
                                                               contrib_id, reverse_z, transaction_id, sorting_metadata)
            else:
                if limit_to_tiles is None and self.area_of_interest:
                    limit_to_tiles = self.tiles_of_interest
                # npy and csv don't have coordinate system, so set up a default.  npz will set this later if needed
                if override_epsg == NO_OVERRIDE:
                    epsg = self.db.epsg
                else:
                    epsg = override_epsg
                transformer = get_crs_transformer(epsg, self.db.epsg)

                if not dformat:
                    dformat = [('x', 'f8'), ('y', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')]
                start = 0
                done = False
                tile_list = numpy.zeros([0,2], dtype=numpy.int32)  # make an empty list to start
                # find the tiles to lock
                with tqdm(desc="geocode/process points", total=2) as top_progress_bar:
                    with tqdm(desc="geocode+split", total=0, leave=False) as progress_bar:
                        if str(path_to_survey_data).lower().endswith(".gpkg"):
                            pts_iterator = self.iterate_pts_gpkg(path_to_survey_data, block_size=block_size)
                        else:
                            pts_iterator = iterate_points_file(path_to_survey_data, dformat=dformat, block_size=block_size)
                        for wkt, x, y, depth, uncertainty in pts_iterator:
                            # this shold only need to happen once but we can reset the transformer on each loop as it's cheap to do
                            progress_bar.reset(progress_bar.total + 1)
                            progress_bar.refresh()
                            progress_bar.update(progress_bar.total - 1)
                            progress_bar.refresh()
                            if wkt is not None and override_epsg == NO_OVERRIDE:
                                epsg = wkt
                                transformer = get_crs_transformer(epsg, self.db.epsg)
                            # transformer
                            #     Optional function used to transform from x,y in the file to the coordinate system of the database.
                            #     It will be called as new_x, new_y = func( x, y ).
                            if transformer:
                                x, y = transformer.transform(x, y)
                            try:
                                txs, tys = self.db.tile_scheme.xy_to_tile_index(x, y)
                            except IndexError as e:
                                if crop:
                                    self.db.LOGGER.warning(f"Points out of range for insertion, cropping data {path_to_survey_data}\n{str(e)}")
                                    has_out_of_bounds_points = True
                                    too_big = numpy.logical_or(x > self.db.tile_scheme.max_x, y > self.db.tile_scheme.max_y)
                                    too_small = numpy.logical_or(x < self.db.tile_scheme.min_x, y < self.db.tile_scheme.min_y)
                                    bad_indices = numpy.logical_or(too_big, too_small)
                                    x = x[~bad_indices]
                                    y = y[~bad_indices]
                                    txs, tys = self.db.tile_scheme.xy_to_tile_index(x, y)
                                else:
                                    raise e
                            new_tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
                            tile_list = numpy.unique(numpy.vstack([tile_list, new_tile_list]), axis=0)

                        # process the data into an accumulation db then insert into the main db
                        top_progress_bar.update(1)
                        progress_bar.set_description("processing")
                        progress_bar.reset()
                        with AreaLock(tile_list, EXCLUSIVE | NON_BLOCKING, self.db.get_history_path_by_index) as lock:
                            self.start_survey_insertion(path_to_survey_data, tile_list, contrib_id, transaction_id)
                            # @TODO for small text data (like ENCs) there is a lot of overhead in making an accumulation DB and then merging it in.
                            #   Could optionally determine/supply the point count (for npz files) and
                            #   then choose to directly insert or make an accumulation DB
                            temp_path = tempfile.mkdtemp(suffix="_accum", dir=self.db.data_path)
                            storage_db = self.db.make_accumulation_db(temp_path)
                            for wkt, x, y, depth, uncertainty in iterate_points_file(path_to_survey_data, dformat=dformat, block_size=block_size):
                                progress_bar.update(1)
                                if has_out_of_bounds_points:  # must have specified crop and failed the tile computation, so remove the data
                                    too_big = numpy.logical_or(x > self.db.tile_scheme.max_x, y > self.db.tile_scheme.max_y)
                                    too_small = numpy.logical_or(x < self.db.tile_scheme.min_x, y < self.db.tile_scheme.min_y)
                                    bad_indices = numpy.logical_or(too_big, too_small)
                                    x = x[~bad_indices]
                                    y = y[~bad_indices]
                                    depth = depth[~bad_indices]
                                    uncertainty = uncertainty[~bad_indices]
                                # transformer
                                #     Optional function used to transform from x,y in the file to the coordinate system of the database.
                                #     It will be called as new_x, new_y = func( x, y ).
                                if transformer:
                                    x, y = transformer.transform(x, y)
                                if reverse_z:
                                    depth = depth * -1
                                score = numpy.full(x.shape, survey_score)
                                flags = numpy.full(x.shape, flag)
                                # we already computed all tiles that should be filled, so we can ignore this
                                tiles = self.insert_survey_array(numpy.array((x, y, depth, uncertainty, score, flags)), path_to_survey_data,
                                                                 contrib_id=contrib_id, compare_callback=compare_callback, limit_to_tiles=limit_to_tiles,
                                                                 accumulation_db=storage_db)
                            self.db.append_accumulation_db(storage_db)
                            self.finished_survey_insertion(path_to_survey_data, tile_list, contrib_id, override_epsg, reverse_z, survey_score, flag,
                                                           dformat, transaction_id, sorting_metadata=sorting_metadata)
                            self.db.remove_accumulation_db(storage_db)
                            del storage_db
        else:
            raise Exception(f"Survey Exists already in database {contrib_id}")

    def _insert_xyz(self, x, y, depth, uncertainty, survey_score, flag, contrib_id, path_to_survey_data, compare_callback, override_epsg, reverse,
                    limit_to_tiles=None, force=False, dformat=None, transaction_id=-1, sorting_metadata=None):
        """Convenience function to handle locking tiles and making flag/score arrays for readers that make arrays of points"""
        if force or contrib_id is None or contrib_id not in self.included_ids:
            score = numpy.full(x.shape, survey_score)
            flags = numpy.full(x.shape, flag)
            txs, tys = self.db.tile_scheme.xy_to_tile_index(x, y)
            tile_list = numpy.unique(numpy.array((txs, tys)).T, axis=0)
            with AreaLock(tile_list, EXCLUSIVE | NON_BLOCKING, self.db.get_history_path_by_index) as lock:
                self.start_survey_insertion(path_to_survey_data, tile_list, contrib_id, transaction_id)
                tiles = self.insert_survey_array(numpy.array((x, y, depth, uncertainty, score, flags)), path_to_survey_data,
                                                 contrib_id=contrib_id, compare_callback=compare_callback, limit_to_tiles=limit_to_tiles)
                self.finished_survey_insertion(path_to_survey_data, tiles, contrib_id, override_epsg, reverse, survey_score, flag, dformat,
                                               transaction_id, sorting_metadata)
        else:
            raise Exception(f"Survey Exists already in database {contrib_id}")

    # @todo - make the survey_ids and survey_paths into properties that load from disk when called by user so that they stay in sync.
    #    Otherwise use postgres to hold that info so queries are current.
    def finished_survey_insertion(self, path_to_survey_data, tiles, contrib_id=numpy.nan,
                                  override_epsg=NO_OVERRIDE, reverse_z=False, survey_score=100,
                                  flag=0, dformat=None, transaction_id=-1, sorting_metadata=None,
                                  override_time = None):
        """ Stores all the information needed to reinsert or tell how a survey was inserted EXCEPT for the comparison callback which is a function.
        Two dictionaries are made, self.included_ids keyed on the IDs and self.included_surveys keys on the file paths.

        Parameters
        ----------
        transaction_id
        dformat
        path_to_survey_data
        tiles
        contrib_id
        override_epsg
        reverse_z
        survey_score
        flag
        sorting_metadata

        Returns
        -------

        """
        # store the tiles filled by this survey as a convenience lookup for the future when removing or querying.
        # print('locking metadata for exclusive at ', datetime.now().isoformat())
        with FileLock(self.metadata_filename(), self.metadata_mode(), EXCLUSIVE) as metadata_file:
            self.db.LOGGER.debug(f"finishing {path_to_survey_data}")
            # backup_path1 = self.metadata_filename().parent.joinpath("wdb_metadata.bak1")
            # self.to_file(backup_path1)

            # update the metadata in case another process wrote to it since the last time we updated/loaded
            # self.update_metadata_from_disk(locked_file=metadata_file)  # removed with switch to sqlite

            # add the new survey to the metadata and store to disk
            # json doesn't like pathlib.Paths to be stored -- convert to strings
            if override_time is None:
                mtime = pathlib.Path(path_to_survey_data).stat().st_mtime
            else:
                mtime = override_time

            # fixme probably should Get the union of existing tiles plus whatever is passed in.  What cases should we overwrite if ever?
            #    in the meantime we'll assume the data is from full insert and not let a re-insert on limited tiles overwrite the data
            if str(path_to_survey_data) not in self.included_surveys:
                data = self.included_surveys.data_class()
                data.survey_path = path_to_survey_data
                data.nbs_id = contrib_id
                data.tiles = [(tx, ty) for tx, ty in tiles]  # json doesn't like sets, convert to list
                data.sorting_metadata = sorting_metadata
                data.epsg = override_epsg
                data.reverse_z = reverse_z
                data.survey_score = survey_score
                data.flag = flag
                data.dformat = dformat
                data.mtime = mtime
                data.transaction_id = transaction_id
                self.included_surveys[str(path_to_survey_data)] = data
                # (contrib_id, [(tx, ty) for tx, ty in tiles], sorting_metadata, override_epsg, reverse_z, survey_score, flag, dformat, mtime,
                # transaction_id)
                if contrib_id is not None:
                    self.included_ids[contrib_id] = data
                    # (str(path_to_survey_data), [(tx, ty) for tx, ty in tiles], sorting_metadata, override_epsg, reverse_z, survey_score, flag, dformat, mtime,
                    # transaction_id)
            # Sqlite file updates immediately, so no longer need to call to_file() which only stores the class setup.
            # # rather than overwrite, since we have it locked, truncate the file and then write new data to it
            # metadata_file.seek(0)
            # metadata_file.truncate(0)
            # self.to_file(locked_file=metadata_file)
            # print('unlocking metadata at ', datetime.now().isoformat())

    def start_survey_insertion(self, path_to_survey_data, tiles, contrib_id=numpy.nan, transaction_id=-1):
        """
        Parameters
        ----------
        path_to_survey_data
        tiles
        contrib_id
        transaction_id

        Returns
        -------

        """
        # store the tiles filled by this survey as a convenience lookup for the future when removing or querying.
        # print('locking metadata for exclusive at ', datetime.now().isoformat())
        with FileLock(self.metadata_filename(), self.metadata_mode(), EXCLUSIVE) as metadata_file:
            self.db.LOGGER.info(f"starting {contrib_id} {path_to_survey_data}")
            # update the metadata in case another process wrote to it since the last time we updated/loaded
            # self.update_metadata_from_disk(locked_file=metadata_file)  # removed with switch to sqlite

            # add the new survey to the metadata and store to disk
            # json doesn't like pathlib.Paths to be stored -- convert to strings
            mtime = pathlib.Path(path_to_survey_data).stat().st_mtime
            # if a survey was started but not finished it will be listed here, make sure the modtime or hash matches or else the
            # already inserted data could be erroneous and should be removed first
            if str(path_to_survey_data) in self.started_surveys:
                existing_mtime = self.started_surveys[path_to_survey_data].mtime
            elif contrib_id in self.started_ids:
                existing_mtime = self.started_ids[contrib_id].mtime
            else:
                existing_mtime = mtime
            if mtime != existing_mtime:
                # @todo To remove a survey from here we would need to call remove_and_recompute and make sure we don't have the tiles locked?
                #   but we must have the tiles locked or another process could start operating in the same area
                raise ValueError("There was a mismatched mod time already listed in the world DB, need to remove it first(?)")
            # fixme probably should Get the union of existing tiles plus whatever is passed in.  What cases should we overwrite if ever?
            #    in the meantime we'll assume the data is from full insert and not let a re-insert on limited tiles overwrite the data
            if str(path_to_survey_data) not in self.started_surveys:
                data = self.started_surveys.data_class()
                data.nbs_id = contrib_id
                data.survey_path = path_to_survey_data
                data.tiles = [(tx, ty) for tx, ty in tiles]  # json doesn't like sets, convert to list
                data.mtime = mtime
                data.transaction_id = transaction_id
                self.started_surveys[str(path_to_survey_data)] = data
                # (contrib_id, [(tx, ty) for tx, ty in tiles], mtime, transaction_id)  # json doesn't like sets, convert to list
                if contrib_id is not None:
                    self.started_ids[contrib_id] = data
                    # (str(path_to_survey_data), [(tx, ty) for tx, ty in tiles], mtime, transaction_id)
            # Sqlite file updates immediately, so no longer need to call to_file() which only stores the class setup.
            # # rather than overwrite, since we have it locked, truncate the file and then write new data to it
            # metadata_file.seek(0)
            # metadata_file.truncate(0)
            # self.to_file(locked_file=metadata_file)
            # print('unlocking metadata at ', datetime.now().isoformat())

    def insert_survey_array(self, input_survey_data, contrib_name, accumulation_db=None, contrib_id=numpy.nan, compare_callback=None,
                            limit_to_tiles=None):
        """ Insert a numpy array (or list of lists) of data into the database.

        This function is typically used by other methods that are inserting data from a file.
        If data is directly inserted via insert_survey_array then you must call finished_survey_insertion or
        the contributor will not be removable later using the remove/recompute functions.
        The remove functions depend on the tile lists to avoid having to read/search the entire dataset.
        They might call this function multiple times in order to not load all the data at one time,
        which is why the finished_survey_insertion is called separately.


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
        list
            tiles indices that data was added to
        """
        # fixme for survey_data - use pandas?  structured arrays?

        # @todo Allow for pixel sizes?  right now treating as point rather than say 8m or 4m coverages
        if not isinstance(contrib_name, str):
            contrib_name = str(contrib_name)
        if not accumulation_db:
            accumulation_db = self.db
        if contrib_id is None:
            contrib_id = numpy.nan
        # print('convert the contributor integer to store inside a float buffer')
        # a = numpy.array([1, 3, 5, 1234567890], numpy.int32)
        # f = numpy.frombuffer(a.tobytes(), numpy.float32)
        # b = numpy.frombuffer(f.tobytes(), numpy.int32)
        # b
        # array([1, 3, 5, 1234567890])
        # Test that float 32 doesn't get truncated weirdly when taken to float64 and back
        # i32 = numpy.arange(0, 100000000).astype(numpy.int32)
        # f32 = numpy.frombuffer(i32.tobytes(), numpy.float32)
        # f32.shape
        # (100000000,)
        # f64 = f32.astype(numpy.float64)
        # i = numpy.frombuffer(f64.astype(numpy.float32).tobytes(), numpy.int32)
        # numpy.all(i == i32)
        # True
        # FIXME - HACK -- encoding integer contributor number as float to fit in the tiff which is float32.
        #  The raster_data classes should return recarrays (structured arrays) but
        #  that will require some rework on the slicing and concatenation elsewhere.
        #  Due to this, the export routine should convert the float back to int and also get fixed later.
        #  Also have to modify the sorting routine to accommodate the difference
        #  (translate the ints to floats there too)
        try:
            float_contributor = numpy.frombuffer(numpy.int32(contrib_id).tobytes(), numpy.float32)[0]
        except ValueError:
            float_contributor = numpy.float32(numpy.nan)

        contributor = numpy.full(input_survey_data[0].shape, float_contributor)

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

        # # remove any tile that aren't in the allowed list
        # if limit_to_tiles is not None:
        #     keep_rows = numpy.zeros(tile_list.shape[0], numpy.bool8)
        #     # could expand the dimensions of the limit_to_tiles and compare all the tiles at once but
        #     # by looping the tiles and running logical_or this supports sets which don't turn into numpy arrays as easily
        #     for tx, ty in limit_to_tiles:
        #         a = numpy.all(tile_list == (tx, ty), axis=1)
        #         keep_rows = numpy.logical_or(a, keep_rows)
        #     tile_list = numpy.delete(tile_list, ~keep_rows, axis=0)

        # sort by TileX then split into groups based on TX
        # Next sort each group by TileY.
        # This should result in a dictionary of (TX,TY)=survey_data for just that tile.
        # this should be much faster on a large geographic dataset with a lot of data (GMRT)
        # which had been re-comparing to (tx, ty) for all data on each loop.
        # The initial assumption of geographic limited tiles was not good enough so extra logic was needed.

        # sort by x so the split works
        tiles_data = {}
        sorted_tx_ind = txs.argsort()
        survey_data = survey_data[:, sorted_tx_ind]
        txs = txs[sorted_tx_ind]
        tys = tys[sorted_tx_ind]
        # find the indices where each tx (column) starts -- had to be sorted for this to work
        grouped_tx, tx_split_indices = numpy.unique(txs, return_index=True)
        # split the data into lists of points based on the tx (column) it fell in
        grouped_tx_survey_data = numpy.split(survey_data, tx_split_indices[1:], axis=1)
        # also split the ty values so we can sort by it next
        grouped_tys = numpy.split(tys, tx_split_indices[1:])
        # for each tx column of data, sort by ty and make a dictionary of the tx,ty data points
        for tx, group_survey_data, group_tys in zip(grouped_tx, grouped_tx_survey_data, grouped_tys):
            # like above, sort by ty now
            sorted_ty_ind = group_tys.argsort()
            # rearrange the tx group to be in ty ascending order
            group_survey_data = group_survey_data[:, sorted_ty_ind]
            group_tys = group_tys[sorted_ty_ind]
            # find the indices where the ty value changes
            grouped_ty, ty_split_indices = numpy.unique(group_tys, return_index=True)
            # group the tx points into a list split by ty (row)
            grouped_tx_ty_survey_data = numpy.split(group_survey_data, ty_split_indices[1:], axis=1)
            # make a dictionary value for each tx, ty pair
            for ty, tile_survey_data in zip(grouped_ty, grouped_tx_ty_survey_data):
                tiles_data[(int(tx), int(ty))] = tile_survey_data  # force int so we don't accidentally get numpy.int in the keys

        # remove any tile that aren't in the allowed list
        if limit_to_tiles is not None:
            # make sure the limits are ints, not floats or numpy.ints etc so compare works
            limit_ints = set([(int(tx), int(ty)) for tx, ty in limit_to_tiles])
            for tx, ty in list(tiles_data.keys()):
                if (tx, ty) not in limit_ints:
                    del tiles_data[(tx, ty)]
        tile_list = list(tiles_data.keys())

        # itererate each tile that was found to have data
        # @FIXME - confirm that points outside bounds (less than lower left and greater than upper right) don't crash this
        for i_tile, (tx, ty) in enumerate(tqdm(tile_list, desc='survey array tiles', mininterval=.7, leave=False)):
            if _debug:
                pass
                # print("debug skipping tiles")
                # if tx != 3325 or ty != 3207:  # utm 16, US5PLQII_utm, H13196 -- gaps in DB and exported enc cells
                # if tx != 3325 or ty != 3207:  # utm 16, US5MSYAF_utm, H13193 (raw bag is in utm15 though) -- gaps in DB and exported enc cells  217849.73 (m), 3307249.86 (m)
                # if tx != 4614 or ty != 3227:  # utm 15 h13190 -- area with res = 4.15m (larger than the 4m output)
                # if tx != 4615 or ty != 3227:  # utm 15 h13190 -- area with res = 4.15m (larger than the 4m output)
                # if tx != 3500 or ty != 4143:
                #     continue
            # print(f'processing tile {i_tile + 1} of {len(tile_list)}')
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
            # pts2 = survey_data[:, numpy.logical_and(txs == tx, tys == ty)]
            pts = tiles_data[(tx, ty)]
            # for j in range(pts.shape[1]):
            #     for i in range(pts2.shape[1]):
            #         if (pts2[:, i] == pts[:, j]).all():
            #             print(j, 'found', i)
            #             break

            # replace x,y with row, col for the points
            i, j = raster_data.xy_to_rc_using_dims(new_arrays.shape[1], new_arrays.shape[2], pts[0], pts[1])
            # if there is a contributor ID and callback,
            # pass the existing contributors to the callback function and genenerate as many comparison matrices as needed
            # also allow for depth (elevation) to be placed into the comparison matrices.
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

        return [tuple((int(tx), int(ty))) for tx, ty in tile_list]  # convert to a vanilla python int for compatibility with json

    def init_tile(self, tx, ty, tile_history):
        """
        Parameters
        ----------
        tx
        ty
        tile_history

        Returns
        -------

        """
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
            return int(0.00001 + (uy - ly) / self.db.tile_scheme.res_y), int(0.00001 + (ux - lx) / self.db.tile_scheme.res_x)
        else:
            return 512, 512

    def insert_survey_vr(self, vr, survey_score=100, flag=0, override_epsg=NO_OVERRIDE, contrib_id=numpy.nan, compare_callback=None, reverse_z=False,
                         limit_to_tiles=None, force=False, transaction_id=-1, sorting_metadata=None):
        """
        Parameters
        ----------
        vr
        survey_score
        flag
        override_epsg
        contrib_id
        compare_callback
        reverse_z
        limit_to_tiles
        force
        transaction_id
        sorting_metadata

        Returns
        -------

        """
        # raise Exception("limit tiles to area of interest, if applicable")
        if not isinstance(vr, bag.VRBag):
            vr = bag.VRBag(vr, mode='r')
        self.db.LOGGER.debug(f"insert VR bag {vr.filename}")

        crs_transformer = self._get_transformer(vr.srs, override_epsg, vr.filename)

        # adjust for the VR returning center of supercells and not the edge.
        supercell_half_x = vr.cell_size_x / 2.0
        supercell_half_y = vr.cell_size_y / 2.0
        x1 = vr.minx - supercell_half_x
        y1 = vr.miny - supercell_half_y
        # noinspection PyUnresolvedReferences  -- nbs needs to update version of HSTP
        x2 = vr.maxx + supercell_half_x
        # noinspection PyUnresolvedReferences
        y2 = vr.maxy + supercell_half_y
        # convert to target reference system if needed
        bounds = poly_from_pts(((x1, y1), (x2, y2)), crs_transformer)

        needs_processing = force or contrib_id is None or contrib_id not in self.included_ids
        skip_as_disjoint = False
        if needs_processing and self.area_of_interest and not self.area_of_interest.Intersects(bounds):
            skip_as_disjoint = True
        if skip_as_disjoint:
            self.insert_survey_as_outside_area_of_interest(vr.filename, survey_score, flag, 'vr', override_epsg,
                                                           contrib_id, reverse_z, transaction_id, sorting_metadata)
        else:
            # @todo adjust tile_list for area_of_interest
            tile_list = self.db.get_tiles_indices(x1, y1, x2, y2)
            with AreaLock(tile_list, EXCLUSIVE | NON_BLOCKING, self.db.get_history_path_by_index) as lock:
                if needs_processing:
                    self.start_survey_insertion(vr.filename, tile_list, contrib_id, transaction_id)
                    refinement_list = numpy.argwhere(vr.get_valid_refinements())
                    if self.area_of_interest:
                        refinements = refinement_list.tolist()
                        for r in range(len(refinements) - 1, -1, -1):
                            i, j = refinements[r]
                            bounds = poly_from_pts(vr.refinement_extents(i, j), crs_transformer)
                            if not self.area_of_interest.Intersects(bounds):
                                del refinements[r]
                        refinement_list = numpy.array(refinements, dtype=refinement_list.dtype)

                    if len(refinement_list) == 0:
                        all_tiles = []
                    else:
                        # in order to speed up the vr processing, which would have narrow strips being processed
                        # use a morton ordering on the tile indices so they are more closely processed in geolocation
                        # and fewer loads/writes are requested of the db tiles (which are slow tiff read/writes)
                        mort = morton.interleave2d_64(refinement_list.T)
                        sorted_refinement_indices = refinement_list[numpy.lexsort([mort])]

                        temp_path = tempfile.mkdtemp(suffix="_accum", dir=self.db.data_path)
                        storage_db = self.db.make_accumulation_db(temp_path)
                        x_accum, y_accum, depth_accum, uncertainty_accum, scores_accum, flags_accum = None, None, None, None, None, None
                        max_len = 500000
                        all_tiles = set()

                        for ti, tj in tqdm(sorted_refinement_indices, desc='refinement', mininterval=.7, leave=False):
                            # get an individual refinement and convert it to x,y from the row column system it was in.
                            refinement = vr.read_refinement(ti, tj)
                            # todo replace this with
                            r, c = numpy.indices(refinement.depth.shape)  # make indices into array elements that can be converted to x,y coordinates
                            pts = numpy.array([r, c, refinement.depth, refinement.uncertainty]).reshape(4, -1)
                            pts = pts[:, pts[2] != vr.fill_value]  # remove nodata points

                            x, y = affine_center(pts[0], pts[1],
                                                 *refinement.geotransform)  # refinement_llx, resolution_x, 0, refinement_lly, 0, resolution_y)
                            ptsnew = refinement.get_xy_pts_arrays()
                            xnew, ynew = ptsnew[:2]
                            ptsnew = ptsnew[2:]
                            # noinspection PyUnresolvedReferences
                            if not ((x == xnew).all() and (y == ynew).all() and (pts == ptsnew).all()):
                                raise Exception("mismatch")

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
                                                                              uncertainty_accum[:last_index], scores_accum[:last_index],
                                                                              flags_accum[:last_index])),
                                                                 vr.filename, accumulation_db=storage_db, contrib_id=contrib_id,
                                                                 compare_callback=compare_callback, limit_to_tiles=limit_to_tiles)
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
                                                                          uncertainty_accum[:last_index], scores_accum[:last_index],
                                                                          flags_accum[:last_index])),
                                                             vr.filename, accumulation_db=storage_db, contrib_id=contrib_id,
                                                             compare_callback=compare_callback)
                            all_tiles.update(tiles)
                        self.db.append_accumulation_db(storage_db)
                        self.db.remove_accumulation_db(storage_db)
                        del storage_db
                    self.finished_survey_insertion(vr.filename, all_tiles, contrib_id, override_epsg, reverse_z, survey_score, flag, dformat="vr",
                                                   transaction_id=transaction_id, sorting_metadata=sorting_metadata)
                else:
                    raise Exception(f"Survey Exists already in database {contrib_id}")

    def insert_raster_survey(self, path_to_survey_data, survey_score=100, flag=0, override_epsg=NO_OVERRIDE, data_band=1, uncert_band=2,
                           contrib_id=numpy.nan, compare_callback=None, reverse_z=False, limit_to_tiles=None, force=False,
                           transaction_id=-1, sorting_metadata=None):
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
        override_epsg
        data_band
        uncert_band
        contrib_id
        compare_callback
        reverse_z
        limit_to_tiles
        force
        transaction_id
        sorting_metadata

        Returns
        -------

        """
        # @fixme rasterdata on disk is not storing the survey_ids?  Forgot to update/write metadata?

        # metadata = {**dataset.GetMetadata()}
        # pixel_is_area = True if 'AREA_OR_POINT' in metadata and metadata['AREA_OR_POINT'][:-2] == 'Area' else False
        self.db.LOGGER.debug(f"insert gdal dataset {path_to_survey_data}")
        ds = gdal.Open(str(path_to_survey_data))
        drv = ds.GetDriver()
        driver_name = drv.ShortName
        geotransform = ds.GetGeoTransform()
        crs_transformer = self._get_transformer(ds.GetSpatialRef(), override_epsg, path_to_survey_data)

        x1, y1 = affine_center(0, 0, *geotransform)
        x2, y2 = affine_center(ds.RasterYSize, ds.RasterXSize, *geotransform)
        bounds = poly_from_pts(((x1, y1), (x2, y2)), crs_transformer)

        needs_processing = force or contrib_id is None or contrib_id not in self.included_ids
        skip_as_disjoint = False
        if needs_processing and self.area_of_interest and not self.area_of_interest.Intersects(bounds):
            skip_as_disjoint = True
        if skip_as_disjoint:
            self.insert_survey_as_outside_area_of_interest(path_to_survey_data, survey_score, flag, 'gdal', override_epsg,
                                                           contrib_id, reverse_z, transaction_id, sorting_metadata)
        else:
            tile_list = self.db.get_tiles_indices(x1, y1, x2, y2)
            with AreaLock(tile_list, EXCLUSIVE | NON_BLOCKING, self.db.get_history_path_by_index) as lock:
                if needs_processing:
                    self.start_survey_insertion(path_to_survey_data, tile_list, contrib_id, transaction_id)
                    temp_path = tempfile.mkdtemp(suffix="_accum", dir=self.db.data_path)
                    storage_db = self.db.make_accumulation_db(temp_path)
                    all_tiles = set()
                    # read the data array in blocks
                    # if geo_debug and False:
                    #     col_block_size = col_size
                    #     row_block_size = row_size

                    if self.area_of_interest:
                        aoi_x1, aoi_x2, aoi_y1, aoi_y2 = self.area_of_interest.GetEnvelope()
                        inv_crs_transform = self._get_transformer(ds.GetSpatialRef(), override_epsg, path_to_survey_data, inv=True)
                        if inv_crs_transform is not None:  # convert all four corners of the envelope as it won't be square in reprojected CRS
                            (xs, ys) = inv_crs_transform.transform([aoi_x1, aoi_x1, aoi_x2, aoi_x2], [aoi_y1, aoi_y2, aoi_y2, aoi_y1])
                            aoi_x1, aoi_x2 = min(xs), max(xs)
                            aoi_y1, aoi_y2 = min(ys), max(ys)
                        ([r1, r2], [c1, c2]) = inv_affine([aoi_x1, aoi_x2], [aoi_y1, aoi_y2], *geotransform)
                        start_row = int(min(r1, r2))
                        start_row = max(start_row, 0)
                        start_col = int(min(c1, c2))
                        start_col = max(start_col, 0)
                        end_row = int(max(r1, r2) + 1)
                        end_col = int(max(c1, c2) + 1)
                    else:
                        start_row, start_col = 0, 0
                        end_row, end_col = None, None
                    # @fixme -- bands 1,2 means bag works but single band will fail
                    for ic, ir, nodata, (data, uncert) in iterate_gdal_image(ds, band_nums=(data_band, uncert_band), min_block_size=4096,
                                                                             max_block_size=8192, start_col=start_col, start_row=start_row,
                                                                             end_col=end_col, end_row=end_row):
                        # if _debug:
                        #     if ic > 1:
                        #         break

                        # read the uncertainty as an array (if it exists)
                        r, c = numpy.indices(data.shape)  # make indices into array elements that can be converted to x,y coordinates
                        r += ir  # adjust the block r,c to the global raster r,c
                        c += ic
                        # pts = numpy.dstack([r, c, data, uncert]).reshape((-1, 4))
                        pts = numpy.array([r, c, data, uncert]).reshape(4, -1)
                        # @todo consider changing this to use the where_not_nodata function
                        if nodata is None or numpy.isnan(nodata):
                            pts = pts[:, ~numpy.isnan(pts[2])]
                        else:
                            # found in a 32bit float that gdal is returning a double precision number inf (1.7976931348623157e+308)
                            # which wouldn't match the data which had INF so we'll remove data by numpy.inf and the value itself
                            if not numpy.isinf(nodata) and nodata in (numpy.finfo(numpy.float32).max, numpy.finfo(numpy.float64).max,):
                                pts = pts[:, pts[2] != numpy.inf]  # remove nodata points
                            elif not numpy.isinf(nodata) and nodata in (numpy.finfo(numpy.float32).min, numpy.finfo(numpy.float64).min):
                                pts = pts[:, pts[2] != -numpy.inf]  # remove nodata points
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
                                                             accumulation_db=storage_db, contrib_id=contrib_id, compare_callback=compare_callback,
                                                             limit_to_tiles=limit_to_tiles)
                            all_tiles.update(tiles)
                    self.db.append_accumulation_db(storage_db)
                    self.db.remove_accumulation_db(storage_db)
                    del storage_db
                    # @fixme -- turn all_tiles into one consistent, unique list.  Is list of lists with duplicates right now
                    self.finished_survey_insertion(path_to_survey_data, all_tiles, contrib_id, override_epsg, reverse_z, survey_score, flag,
                                                   dformat="gdal", transaction_id=transaction_id, sorting_metadata=sorting_metadata)
                else:
                    raise Exception(f"Survey Exists already in database {contrib_id}")

    def export_area(self, fname, x1, y1, x2, y2, res, target_epsg=None, driver="GTiff",
                    layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
                    gdal_options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"),
                    compare_callback=None, align=True):
        """ Retrieves an area from the database at the requested resolution.

        # 1) Create a single tif tile that covers the area desired
        # 2) Get the master db tile indices that the area overlaps and iterate them
        # 3) Read the single tif sub-area as an array that covers this tile being processed
        # 4) Use the db.tile_scheme function to convert points from the tiles to x,y
        # 5) Make sure the tiles aren't locked, and put in a read lock so the data doesn't get changed while we are reading
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
        gdal_options
        compare_callback
        align
        Returns
        -------
        int, dataset
            number of database tiles that supplied data into the export area, open gdal dataset for the file location specified

        """
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg
        # 1) Create a single tif tile that covers the area desired
        # probably won't export the score layer but we need it when combining data into the export area
        dataset, dataset_score = self.make_export_rasters(fname, x1, y1, x2, y2, res,
                                                          target_epsg=target_epsg, driver=driver, layers=layers,
                                                          gdal_options=gdal_options, align=align)

        tile_count = self.export_into_raster(dataset, dataset_score, target_epsg=target_epsg, layers=layers, compare_callback=compare_callback)
        score_name = dataset_score.GetDescription()
        del dataset_score
        try:
            os.remove(score_name)
        except PermissionError:
            gc.collect()
            try:
                os.remove(score_name)
            except PermissionError:
                print(f"Failed to remove {score_name}, permission denied (in use?)")
        return tile_count, dataset

    def make_export_rasters(self, fname, x1, y1, x2, y2, res, target_epsg=None, driver="GTiff",
                            layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
                            gdal_options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"),
                            align=True):
        """
        Parameters
        ----------
        fname
        x1
        y1
        x2
        y2
        res
        target_epsg
        driver
        layers
        gdal_options
        align

        Returns
        -------

        """
        # @TODO change align flag to be None/center/edge for no alignment, cell centers match (like bag format) or cell edge (like geotiff)
        try:
            dx, dy = res
        except TypeError:
            dx = dy = res
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg

        fname = pathlib.Path(fname)
        score_name = fname.with_suffix(".score" + fname.suffix)
        if align:
            align_y, align_x = self.db.tile_scheme.min_y, self.db.tile_scheme.min_x
        else:
            align_x = align_y = None
        dataset = make_gdal_dataset_area(fname, len(layers), x1, y1, x2, y2, dx, dy, target_epsg, driver,
                                         gdal_options, align_x=align_x, align_y=align_y)
        for index, band_info in enumerate(layers):
            band = dataset.GetRasterBand(index + 1)
            band.SetDescription(LayersEnum(band_info).name)
            del band
        # probably won't export the score layer but we need it when combining data into the export area
        dataset_score = make_gdal_dataset_area(score_name, 3, x1, y1, x2, y2, dx, dy, target_epsg, driver, align_x=align_x, align_y=align_y)
        return dataset, dataset_score

    def export_into_raster(self, dataset, dataset_score, target_epsg=None,
                           layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
                           compare_callback=None):
        """
        Parameters
        ----------
        dataset
        dataset_score
        target_epsg
        layers
        compare_callback

        Returns
        -------

        """
        if not target_epsg:
            target_epsg = self.db.tile_scheme.epsg
        crs_transform = get_crs_transformer(self.db.tile_scheme.epsg, target_epsg)
        inv_crs_transform = get_crs_transformer(target_epsg, self.db.tile_scheme.epsg)

        affine_transform = dataset.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy

        max_cols, max_rows = dataset.RasterXSize, dataset.RasterYSize

        # 2) Get the master db tile indices that the area overlaps and iterate them
        x1, y1 = affine(0, 0, *affine_transform)
        x2, y2 = affine(max_rows - 1, max_cols - 1, *affine_transform)
        if crs_transform:
            overview_x1, overview_y1, overview_x2, overview_y2 = transform_rect(x1, y1, x2, y2, inv_crs_transform.transform)
        else:
            overview_x1, overview_y1, overview_x2, overview_y2 = x1, y1, x2, y2
        tile_count = 0

        cnt_histories = 0
        # @todo change functions in backends to iterate tile paths and then build on that for iterating tiles (instead of just tiles)
        for tx, ty in self.db.get_tiles_indices(overview_x1, overview_y1, overview_x2, overview_y2):
            hist_path = self.db.get_history_path_by_index(tx, ty)
            if self.db.storage_class.exists(hist_path):
                cnt_histories += 1

        # for txi, tyi, tile_history in self.db.iter_tiles(overview_x1, overview_y1, overview_x2, overview_y2, no_create=True):
        for txi, tyi, tile_history in tqdm(self.db.iter_tiles(overview_x1, overview_y1, overview_x2, overview_y2, no_create=True),
                                            total=cnt_histories, desc='bruty tile', mininterval=.7, leave=False):
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
        return tile_count

    def fast_extract(self, x1, y1, x2, y2, layers=(LayersEnum.ELEVATION,)):
        """ Quickly extract data.  This can only be used by databases that used an ExactTilingScheme so the indexing is uniform.

        This function will basically grab the data and just insert into an output array at the same resolution as the underlying data.
        This is faster because the scoring comparison is not needed and the CRS transforms are not applicable.
        The only thing needed is to figure out the database indices mapping to the output array.

        Returns
        -------

        """
        if not isinstance(self.db.tile_scheme, ExactTilingScheme):
            raise TypeError("The tile scheme for this database is not derived from ExactTilingScheme "
                            "and this function can not be used since the tile sizes aren't uniform")

        txs, tys = self.db.tile_scheme.xy_to_tile_index(numpy.array([x1, x2]), numpy.array([y1, y2]))
        txs = list(range(min(txs), max(txs) + 1))
        tys = list(range(min(tys), max(tys) + 1))
        cols = numpy.zeros(len(txs), numpy.int)
        rows = numpy.zeros(len(tys), numpy.int)

        # # @todo figure out the starting position and what its row/column is and make that the origin
        # nr, nc = self.init_tile(txs[0], tys[0], None)
        # start_row, start_col = xy_to_rc_using_dims
        # # @todo figure out the amount to trim at the end also
        # nr, nc = self.init_tile(txs[-1], tys[-1], None)
        # end_row, end_col = xy_to_rc_using_dims

        total_cols = 0
        for itx, tx in enumerate(txs):
            cols[itx] = total_cols
            nr, nc = self.init_tile(tx, tys[0], None)
            total_cols += nc

        total_rows = 0
        for ity, ty in enumerate(tys):
            rows[ity] = total_rows
            nr, nc = self.init_tile(txs[0], ty, None)
            total_rows += nr
        rr, cc = numpy.meshgrid(rows, cols)
        indices = numpy.transpose([rr, cc])

        # @FIXME is float32 going to work with contributor being a large integer value?
        output_array = numpy.full([len(layers), total_rows, total_cols], numpy.nan, numpy.float32)

        for icol, tx in enumerate(txs):
            for irow, ty in enumerate(tys):
                tile_history = self.db.get_tile_history_by_index(tx, ty)
                try:
                    raster_data = tile_history[-1]
                    arrays = raster_data.get_arrays(layers)
                    start_row, start_col = indices[irow, icol]
                    output_array[:, start_row:start_row + arrays.shape[1], start_col:start_col + arrays.shape[2]] = arrays
                except IndexError:
                    # if the db is empty then nothing to do
                    pass
        return output_array, txs, tys, indices

    @staticmethod
    def merge_rasters(tile_layers, tile_scoring, raster_data,
                      crs_transform, affine_transform, start_col, start_row, block_cols, block_rows,
                      dataset, layers, dataset_score, reverse_sort=(False, False, False)):
        """
        Parameters
        ----------
        tile_layers
        tile_scoring
        raster_data
        crs_transform
        affine_transform
        start_col
        start_row
        block_cols
        block_rows
        dataset
        layers
        dataset_score
        reverse_sort

        Returns
        -------

        """
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
            score_band = dataset_score.GetRasterBand(i + 1)
            score_band.WriteArray(sort_key_scores[i], start_col, start_row)

    def extract_soundings(self):
        # this is the same as extract area except score = depth
        raise NotImplementedError

    def soundings_from_caris_combined_csar(self):
        # Either fill a world database and then extract soundings or
        # generalize the inner loop of extract area then use it against csar data directly
        raise NotImplementedError

    def export_at_date(self, area, date):
        raise NotImplementedError

    def clean(self, removals, compare_callback=None, transaction_id=-1, subprocesses=5):
        """
        Parameters
        ----------
        invalid_score_surveys
        unfinished_surveys
        out_of_sync
        compare_callback
        transaction_id
        subprocesses

        Returns
        -------

        """
        # removals = set()
        # removals.update(invalid_score_surveys)
        # removals.update(unfinished_surveys)
        # removals.update(out_of_sync)
        self.remove_and_recompute(removals, compare_callback=compare_callback, transaction_id=transaction_id, subprocesses=subprocesses)

    def remove_survey(self, contributor, transaction_id=-1):
        """ Will remove a survey from all the tiles it had been appled to.
        Returns a list of the tiles affected and the IDs that had been processed after the removed contributor.
        Those contributors that came after are also removed from the listed tiles.
        The caller should then re-insert those contributors or remove them from the database also.
        See reinsert_surveys_in_tiles and remove_and_recompute functions.

        Imagine a high accuracy/score survey was inserted and then some low accuracy/score surveys are added.
        The high accuracy data would prevent the low accuracy from ever appearing in the history.
        Then it is discoverd that the high scoring survey had errors in it (say, bad vessel offsets) and needs to be removed.
        The low accuracy data would never have appeared in the history so removing the high accuracy survey would leave a hole.
        This is why the caller makes the decision to remove or re-insert the data.

        Will raise nbs_locks.LockNotAcquired if the tiles are already in use and should be tried again later.

        Parameters
        ----------
        contributor
        transaction_id

        Returns
        -------
        dict
            keys are survey ids affected, values are sets of the tiles affected for that survey
        """
        # 1) Find all the tiles the ID falls in and make a master list of tile numbers (this is in master metadata record?)
        # 1a) Lock all the tiles for write access
        # 2) For each Tile, find where in the history of the tile the ID was
        # 3) Find all the IDs that came after it's insertion and add to a master list of IDs
        # 4) Revert all the tiles back to the state before the ID was inserted
        # 5) Remove the ID from the master list of IDs that are included in the database

        # 1) Find all the tiles the ID falls in and make a master list of tile numbers (this is in master metadata record?)
        try:
            rec = self.started_ids[contributor]
        except KeyError:
            rec = self.included_ids[contributor]
        path_to_file, tile_list = rec.survey_path, rec.tiles
        self.db.LOGGER.debug(f"Trying to remove contributor {contributor} - try lock for:")
        self.db.LOGGER.debug(f"  path to file was {path_to_file} in tiles {tile_list}")
        # 1a) Lock all the tiles for write access
        with AreaLock(tile_list, EXCLUSIVE | NON_BLOCKING, self.db.get_history_path_by_index) as lock:
            self.db.LOGGER.info(f"removing contributor {contributor}")
            # add to the removed listing before we take it out of the overall metadata
            data = self.removed_ids.data_class()
            data.nbs_id = contributor
            data.tiles = tile_list
            data.affects = "TBD"
            data.transaction_id = transaction_id
            data.started = 1
            data.finished = 0
            new_id = self.removed_ids.add_oid_record(data)  # (contributor, tile_list, "TBD", transaction_id, 1, 0))
            contributor_tiles = {}  # master list of other contributors that will be affected by removing the requested contributor
            for tx, ty in tile_list:
                tile_history = self.db.get_tile_history_by_index(tx, ty, no_create=True)
                if tile_history is not None:
                    remove_index = None
                    for t in range(len(tile_history)):
                        # meta = tile_history.get_metadata()['contributors']
                        current_contrib = tile_history.history[t].get_metadata()['contrib_id']
                        # 2) For each Tile, find where in the history of the tile the ID was
                        if current_contrib == contributor:
                            if remove_index is None:  # if the contributor is listed twice (perhaps a crash during load) then ignore the second occurrence
                                remove_index = t
                        # 3) Find all the IDs that came after it's insertion and add to a master list of IDs
                        if remove_index is not None:  # we'd found the desired contributor already
                            if current_contrib != contributor:
                                contributor_tiles.setdefault(current_contrib, set())
                                contributor_tiles[current_contrib].add((int(tx), int(ty)))
                    # 4) Revert all the tiles back to the state before the ID was inserted
                    # remove the history entries from the requested contributor to the end
                    if remove_index is not None:
                        del tile_history[remove_index:]
            data.affects = ",".join([str(c) for c in contributor_tiles])
            data.finished = 1
            self.removed_ids[new_id] = data
            # (contributor, tile_list, ",".join([str(c) for c in contributor_tiles]), transaction_id, 1, 1)

            # 5) Remove the ID from the master list of IDs that are included in the database
            # remove the contributor from the master metadata list
            # (the contributor list holds contributors that are edited and either need to be removed or re-inserted)
            with FileLock(self.metadata_filename(), self.metadata_mode(), EXCLUSIVE) as metadata_file:
                # backup_path1 = self.metadata_filename().parent.joinpath("wdb_metadata.bak1")
                # self.to_file(backup_path1)

                # update the metadata in case another process wrote to it since the last time we updated/loaded
                # self.update_metadata_from_disk(locked_file=metadata_file)  # removed with switch to sqlite

                # add the new survey to the metadata and store to disk
                # json doesn't like pathlib.Paths to be stored -- convert to strings
                try:
                    del self.included_surveys[path_to_file]
                except KeyError:
                    pass  # must not have finished this survey (crashed in processing)
                del self.started_surveys[path_to_file]

                try:
                    del self.included_ids[int(contributor)]
                except KeyError:
                    pass  # must not have finished this survey (crashed in processing)
                try:
                    del self.started_ids[contributor]
                except KeyError:
                    try:
                        del self.started_ids[str(contributor)]
                    except KeyError:
                        try:
                            del self.started_ids[int(contributor)]
                        except KeyError:
                            self.db.LOGGER.warning(f"While removing {contributor}, it was not found in started_ids table.  The database may still be corrupt - a full scan of contributor layers would have to be done to be certain." )

                # Sqlite file updates immediately, so no longer need to call to_file() which only stores the class setup.
                # # rather than overwrite, since we have it locked, truncate the file and then write new data to it
                # metadata_file.seek(0)
                # metadata_file.truncate(0)
                # self.to_file(locked_file=metadata_file)

        self.db.LOGGER.debug(f"removing contributor {contributor} will affect {len(contributor_tiles)} contributors: {contributor_tiles}")
        return contributor_tiles

    def reinsert_from_sqlite(self, comp_callback=None):
        """ Use the metadata to try and reinsert contributors from disk.
        This is useful when removing a survey which then requires all surveys after it to be reprocessed.
        Optionally can restrict insert to only certain tile indices,
        useful when a survey is removed and not the entire database area needs to be reprocessed.

        Parameters
        ----------

        Returns
        -------

        """
        recs = True
        skips = 0
        while len(self.reinserts.unfinished_records()) > 0:
            # this stops a race condition where the process that did the work on the last survey gets shut out by the other processes overwhelming the lock server
            time.sleep(2)
            for reinsert_rec in self.reinserts.unfinished_records():  # if we use the generator it may have trouble when records are removed by other processes
                oid = reinsert_rec.oid
                contributor = reinsert_rec.nbs_id
                try:
                    included_rec = self.included_ids[contributor]
                except KeyError:
                    try:
                        included_rec = self.included_ids[str(contributor)]
                    except KeyError:
                        try:
                            included_rec = self.included_ids[int(contributor)]
                        except KeyError:
                            # if the contributor to reinsert is not in the included_ids list then remove the reinsert as either:
                            # 1) There was an error during insert, or
                            # 2) there was a change in the metadata tables or other bug.
                            # Removing it from the reinsert list should then cause it to get cleaned up in one of the QC runs.
                            # Either as data in a tile but not in the DB or an unfinished insert (in 'started_ids' but not 'inserted_ids')
                            self.db.LOGGER.warning(
                                f"{contributor} was found in the reinsert list but not in the included_ids\nThis should get automatically cleaned up but PLEASE CONFIRM the DB is not otherwise corrupt.")
                            for tx, ty in reinsert_rec.tiles:
                                tile_history = self.db.get_tile_history_by_index(tx, ty)
                                contribs = tile_history.get_metadata()['contributors'].keys()
                                if contributor in contribs or contributor in map(int, contribs):
                                    self.db.LOGGER.warning(f"{contributor} was found in the reinsert list but not in the included_ids yet in the bruty internal tiles at ({tx},{ty})!!!")
                            del self.reinserts[oid]
                            continue
                fname = included_rec.survey_path
                override_epsg = included_rec.epsg
                reverse_z = included_rec.reverse_z
                dformat = included_rec.dformat
                trans_id = included_rec.transaction_id

                try:
                    lock = FileLock(fname)  # this doesn't work with the file lock - just the multiprocessing locks
                    if lock.acquire():
                        rec = self.reinserts[oid]
                        if not rec.finished:  # still in the database, did another process remove it?
                            self.reinserts.set_started(oid)
                            self.insert_survey(fname, contrib_id=rec.nbs_id, compare_callback=comp_callback, override_epsg=override_epsg,
                                               reverse_z=reverse_z,
                                               limit_to_tiles=rec.tiles, force=True, dformat=dformat, transaction_id=trans_id)
                            self.reinserts.set_finished(oid)
                        lock.release()
                    else:
                        # print(f"{path} was locked - probably another process is working on it")
                        raise LockNotAcquired()
                except LockNotAcquired:
                    skips += 1
                    if skips % 20 == 0:
                        print(os.getpid(), 'files in use for ', contributor, fname)
                        print(f'skipped {skips} times so far')

                # Added limit_to_tiles to address this --
                # fixme surveys can be inserted into a tile multiple times - Imagine inserting survey numbers 1,2,3,4 in order
                #    If two surveys (1,4) are removed and they both overlap a survey (2) then the overlapped survey will be re-inserted into
                #    all tiles (1,4) fell in, even if #2 wasn't removed in the areas that had tile 4 since #2 had been inserted first.
                #    So, this tile in question may have had surveys 2,3,4 in it's contributor then #4 is removed which would not list #2 as needing removal
                #    In a different tile surveys 1,2 were listed - and when #1 is removed then #2 is listed for reinsertion.
                #    This means that the 2,3,4 tile will reprocess #2 and it's tile history will show 2,3,2  (duplicate #2 entries)
                #    Either remove one at a time (which has some efficiency issues or check on re-insert if the survey is already in the history,
                #    or just live with the duplicate (it doesn't hurt much).
    def add_reinserts(self, contributors:dict):
        existing_unfinished = {rec.nbs_id: rec for rec in self.reinserts.unfinished_records()}

        for contrib, tiles in contributors.items():
            try:
                existing = existing_unfinished[contrib]
                updated = set(existing.tiles)
                updated.update(tiles)
                existing.tiles = updated
                self.reinserts[existing.oid] = existing  # set the data back into the file
            except KeyError:
                self.reinserts.add_oid_record((contrib, tuple(tiles), 0, 0))

    def remove_and_recompute(self, contributors: (int, float, list, tuple), compare_callback=None, subprocesses=5, transaction_id=-1):
        """ Remove the given contributor IDs and then attempt to reinsert the contributors that would have been affected by the surveys being removed.

        Parameters
        ----------
        contributors
        compare_callback
        subprocesses
        transaction_id

        Returns
        -------

        """
        # if not isinstance(contributors, collections.abc.Iterable):
        try:
            iter(contributors)
        except TypeError:
            contributors = [contributors]
        self.db.LOGGER.info(f"removing and recomputing for contributors: {contributors}")
        affected_contributors = {}
        unfinished_removals = []
        for contributor in contributors:
            try:
                contributors_tiles = self.remove_survey(contributor, transaction_id=transaction_id)
                for contrib, tiles in contributors_tiles.items():
                    affected_contributors.setdefault(contrib, set())
                    affected_contributors[contrib].update(tiles)
            except LockNotAcquired:
                print("Unable to remove survey, lock not acquired", contributor)
                unfinished_removals.append(contributor)
        for contrib in contributors:
            if contrib in affected_contributors:
                del affected_contributors[contrib]  # don't reinsert any surveys listed for full removal

        # @todo - figure out if this is right.  If a removal was locked then we are reinserting everything else then raising an error.
        #    This seems best as otherwise we could remove data but not replace it so we'd have partially inserted data and not know it.
        # 1) Make a sqlite database of the contibutors and tiles to work on
        self.add_reinserts(affected_contributors)
        if len(self.reinserts.unfinished_records()) > 0:
            if NO_LOCK or subprocesses == 1:
                if subprocesses > 1:
                    print("Warning, using more than one process would require a lock server.\nOnly one process will be used.")
                self.reinsert_from_sqlite(comp_callback=compare_callback)
            else:  # a server must be running so we can multi-process
                # 3) start multiple processes to reinsert the data based on the file
                if compare_callback:
                    cb_module = compare_callback.func.__module__
                    cb_name = compare_callback.func.__name__
                    cb_args = compare_callback.args
                    cb_kwargs = compare_callback.keywords
                else:
                    cb_module = None
                    cb_name = None
                    cb_args = ()
                    cb_kwargs = {}
                # to call the insert function we need to pickle all the parameters that were used in the scoring callback
                # fixme this will probably fail if the callback is just a function and not a partial or the parameters get complex (non-picklable)
                all_args = [str(self.db.data_path), current_address()[1], cb_module, cb_name]
                all_args.extend(cb_args)
                process_list = []
                # pickle.dump((all_args, cb_kwargs), open(self.reinsert_filename().with_suffix(".test.reinsert.pickle"), "wb"))
                for p in range(subprocesses):  # wait for them to all finish
                    proc = multiprocessing.Process(target=start_reinsert_process, args=all_args, kwargs=cb_kwargs)
                    proc.start()
                    process_list.append(proc)
                for proc in process_list:  # wait for them to all finish
                    proc.join()
        if unfinished_removals:
            raise RuntimeError(
                f"some removal were locked, make sure all bruty combines are closed and lock server is restarted {unfinished_removals}")

    def revise_survey(self, survey_id, path_to_survey_file):
        # we are not going to trust that the data was only edited for height/depth but that the scoring may have been adjusted too
        # so this becomes a simple two step process.
        # If we want to add a modify the data we could find all the nodes that applied but this would violate the NBS combine method
        # which includes depth as one of the parameters to compute the supercession values, even though rarely used.
        # 1) Remove the existing survey
        # 2) Re-add the edited survey
        raise NotImplementedError

    def find_contributing_surveys(self, area):
        raise NotImplementedError

    def find_area_affected_by_survey(self, survey_data):
        raise NotImplementedError

    def change_survey_score(self, survey_id, new_score):
        raise NotImplementedError

    def __cleanup_disk(self):
        # could both recompress the tiffs and search for broken files?
        raise NotImplementedError


def start_reinsert_process(db_path, lock_port, comp_callback_module, comp_callback_name, *args, **opts):
    """
    Parameters
    ----------
    db_path
    lock_port
    comp_callback_module
    comp_callback_name
    args
    opts

    Returns
    -------

    """
    if lock_port:
        use_locks(lock_port)
    db = WorldDatabase.open(db_path)
    if comp_callback_module:
        im = importlib.import_module(comp_callback_module)
        comp_func = eval("im." + comp_callback_name)
        comp_callback = partial(comp_func, *args, **opts)
    elif comp_callback_name:
        comp_func = eval(comp_callback_name)
        comp_callback = partial(comp_func, *args, **opts)
    else:
        comp_callback = None
    # print(db_path, len(db.included_ids))
    # print(comp_callback_module, comp_callback_name)
    # print(comp_callback, len(args), len(opts))
    db.reinsert_from_sqlite(comp_callback)


class CustomBackend(WorldTilesBackend):
    def __init__(self, utm_epsg, res_x, res_y, x1, y1, x2, y2, history_class, storage_class, data_class, data_path, zoom_level=13):
        tile_scheme = ExactTilingScheme(res_x, res_y, min_x=x1, min_y=y1, max_x=x2, max_y=y2, zoom=zoom_level, epsg=utm_epsg)
        super().__init__(tile_scheme, history_class, storage_class, data_class, data_path)


class CustomArea(WorldDatabase):
    """ This class allows for only processing a limited area using an exact coordinate bound and resolution.
    The default implementation will use an accumulation history (so remove doesn't work) and saves to disk as geotiffs.
    """

    def __init__(self, epsg, x1, y1, x2, y2, res_x, res_y, storage_directory, history_class=None, storage_class=None, data_class=None):
        min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
        shape = max(shape_x, shape_y)
        tiles = shape / 512  # this should result in tiles max sizes between 512 and 1024 pixels
        zoom = int(numpy.log2(tiles))
        if zoom < 0:
            zoom = 0
        if history_class is None:
            history_class = AccumulationHistory
        if storage_class is None:
            storage_class = DiskHistory
        if data_class is None:
            data_class = TiffStorage
        super().__init__(CustomBackend(epsg, res_x, res_y, min_x, min_y, max_x, max_y, history_class, storage_class, data_class,
                                       storage_directory, zoom_level=zoom))

    def export(self, fname, driver="GTiff", layers=(LayersEnum.ELEVATION, LayersEnum.UNCERTAINTY, LayersEnum.CONTRIBUTOR),
               gdal_options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES")):
        """Export the full area of the 'single file database' in the epsg the data is stored in"""
        y1 = self.db.tile_scheme.min_y
        y2 = self.db.tile_scheme.max_y - self.res_y
        x1 = self.db.tile_scheme.min_x
        x2 = self.db.tile_scheme.max_x - self.res_x
        # we already aligned the data with minx, miny so we'll tell export_area that align=false
        return super().export_area(fname, x1, y1, x2, y2, (self.res_x, self.res_y), driver=driver,
                                   layers=layers, gdal_options=gdal_options, align=False)


