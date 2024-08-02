import pathlib
import pickle
from functools import partial
import random
import re
import os
from dataclasses import dataclass
from hashlib import blake2b

from osgeo import gdal, ogr
import numpy
import psycopg2

from nbs.bruty import nbs_locks
from nbs.bruty.raster_data import TiffStorage, LayersEnum
from nbs.configs import parse_multiple_values, iter_configs
from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review.meta_review import database_has_table, split_URL_port

_debug = False

REVIEWED = "qualified"
NOT_NAV = "not_for_navigation"
PREREVIEW = "unqualified"
SENSITIVE = "sensitive"
ENC = "enc"
GMRT = "gmrt"

PUBLIC = "Public"
NAVIGATION = 'Navigation'
INTERNAL = 'Internal'

NBS_ID_STR = "nbs_id"

"""
SELECT nbs_id, from_filename, script_to_filename, manual_to_filename, for_navigation, never_post, decay_score,script_resolution,manual_resolution,script_point_spacing,manual_point_spacing 
FROM public.pbg_gulf_utm14n_mllw
WHERE nbs_id = 566122

SELECT nbs_id, from_filename, script_to_filename, manual_to_filename, for_navigation, never_post, decay_score,script_resolution,manual_resolution,script_point_spacing,manual_point_spacing 
FROM public.pbc_utm19n_mllw
WHERE from_filename like '%H12299%'
"""

@dataclass
class ConnectionInfo:
    database: str
    username: str
    password: str
    hostname: str = 'OCS-VS-NBS05'
    port: str = '5434'
    tablenames: tuple = ()


def hash_id(product_branch: str, zone: int, hemi: str, tile: int, datum: str, res: int):
    """ Get a 64 bit integer to use with the pg_try_advisory_lock_shared or pg_try_advisory_lock functions

    Parameters
    ----------
    product_branch
        any string, but for NBS would be PBB, PBG or PBC etc
    zone
        UTM zone number, for NBS 18, 19 are US east coast
    tile
        The review tile being operated on, for NBS ranging from 1 to 150 in general
    datum
        datum being operated on, MLLW or MSL etc
    res
        resolution of the output product, for NBS will normally be 2, 4, 8 or 16 meter

    Returns
    -------
    int
        the 64 bit hash value as an integer for easy use with postgres locks

    """
    lock_string = f"{product_branch.lower()}_{zone}{hemi}_{tile}_{datum.lower()}_{res}".encode()
    digest = blake2b(lock_string, digest_size=8).hexdigest()
    id = int(digest, 16)
    return id


# @TODO ?? switch to cursor_factory=NamedTupleCursor by default so usage is rec.name instead of rec['name']
def connection_with_retries(conn_info:ConnectionInfo, cursor_factory=psycopg2.extras.DictCursor):
    connection = connect_with_retries(database=conn_info.database, user=conn_info.username, password=conn_info.password,
                                      host=conn_info.hostname, port=conn_info.port)
    cursor = connection.cursor(cursor_factory=cursor_factory)
    return connection, cursor


def connect_params_from_config(config):
    with open(os.path.expanduser(config['URL_FILENAME'])) as hostname_file:
        url = hostname_file.readline()
    hostname, port = split_URL_port(url)

    with open(os.path.expanduser(config['CREDENTIALS_FILENAME'])) as database_credentials_file:
        username, password = [line.strip() for line in database_credentials_file][:2]
    tablenames, database = config.get('tablenames', ""), config['database']
    tablename_list = parse_multiple_values(tablenames)
    return ConnectionInfo(database, username, password, hostname, port, tablename_list)


# switched to Identity column - also made values as one million * utm zone and add 800k or 900k for the prereview and sensitive tables.
# since the tiffs are floats, we only have 7 digits of precision for the identity so make them 40k * utm zone instead of a million
# can fix this later by storing a local translation of contributor to unique number in the database but trying to keep a direct lookup for now.
# *update - made a hack to store int32 in the tiff float32 so full range of int32 available
#   (except for 1232348160 which maps to the 1000000 bag nodata)
# So use UTM * 10000000 + product branch letter (A-G at least) * 1000000 + datum shift (mllw=0, prereview=200000, sensitive=300000, enc=400000)
#   ex: PBG18_prereview = [18680000 : 18688999]
def start_integer(table_name):
    exp = r"pb(?P<pb>\w)_(?P<name>.*)utm(?P<utm>\d+)(?P<hemi>\w)_(?P<datum>\w+)_(?P<type>\w+)$"
    m = re.match(exp, table_name, re.IGNORECASE)
    if m:
        zone_size = 1e7
        utm = int(m.group('utm'))
        utm_offset = utm * zone_size
        if m.group('hemi').lower() == 's':
            utm_offset += zone_size * 60
        product_branch_offset = (ord(m.group('pb').lower()) - ord('a')) * 1e6
        datum = m.group('datum').lower()
        dtype = m.group('type').lower()
        if datum == "mllw" and dtype == "qualified":
            type_offset = 0
        elif datum == "mllw" and dtype == "sensitive":
            type_offset = 200000
        elif datum == "mllw" and dtype == "unqualified":
            type_offset = 300000
        elif datum == "mllw" and dtype == "enc":
            type_offset = 400000
        elif datum == "mllw" and dtype == "gmrt":
            type_offset = 450000
        elif dtype == "modeling":
            type_offset = 500000
        elif datum in ('hrd', 'mld', 'lwrp') and dtype == "qualified":
            type_offset = 600000
        elif datum in ('hrd', 'mld', 'lwrp') and dtype == "enc":
            type_offset = 700000
        else:
            raise ValueError(f"no offset found for {datum}_{dtype}")

        start_val = int(utm_offset + product_branch_offset + type_offset)
    else:
        raise ValueError("Didn't parse tablename to standard nbs naming of pbX_name_utmXX_TYPE")
    return start_val


def last_nbs_id(table_name, cursor):
    """ Return the last value used by an nbs_id Identity column or the max value in the column (in case the column wasn't made right or modified)

    Parameters
    ----------
    table_name
        string of the table name to work on
    cursor
        and open psycopg2 cursor to call the database with
    Returns
    -------
    int
        The max nbs_id found.  If not an Identity column and no rows exist then None is returned.
    """
    cursor.execute(f"""select pg_get_serial_sequence('{table_name}', '{NBS_ID_STR}')""")
    (seq_name, ) = cursor.fetchone()
    if seq_name is not None:
        # get the last value from the sequence table, if it exists,
        # since this would also account for possibly deleted rows after the max value in the nbs_id
        cursor.execute(f"""select last_value from {seq_name}""")
    else:
        # Since the sequence didn't exist then we'll just return the max value in the column
        cursor.execute(f"""select max({NBS_ID_STR}) from {table_name}""")
    (last_num, ) = cursor.fetchone()
    return last_num


def create_identity_column(table_name, start_val, conn_info: ConnectionInfo, col_name=NBS_ID_STR, force_restart=False, drop_add=False):
    """ Ensures there is a column which is a non-null identity.
    Will create the column if it does not exist.
    Will fill any NULL records with an appropriate value after the max(col_name).

    Parameters
    ----------
    table_name
        table to operate on
    start_val
        if the column does not exist, is empty or drop_add is True then sets the starting value of the column
    conn_info
        object containing username, password, network path etc
    col_name
        column to create or modify
    force_restart
        override existing values in the column
    drop_add
        drop and add the column as opposed to modifying in place - this was the original behavior but now modifys the column by default.
        drop_add fails if there are views attached which prevent a drop column from working.
    Returns
    -------
    None

    """
    connection, cursor = connection_with_retries(conn_info)
    # used admin credentials for this
    # cursor.execute("create table serial_19n_mllw as (select * from pbc_utm19n_mllw)")
    # connection.commit()
    needs_id_column = False
    cursor.execute(
        f"""select * from 
            (SELECT column_name, is_identity FROM information_schema.columns WHERE table_name = '{table_name}') as t1 
            WHERE column_name='{col_name}';""")
    ret = cursor.fetchone()
    if ret is None:  # no column named 'col_name' in the table
        needs_id_column = True
    else:
        nm, is_ident = ret
        if is_ident.upper() == "NO":
            if drop_add:  # replace the column from scratch
                cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN {col_name}")
                connection.commit()
                needs_id_column = True
            else:  # change the column in place.  Works with existing views which stop drop from working.
                next_val = last_nbs_id(table_name, cursor)
                # use the start_val if there are no records but the column already exists
                if next_val is None:
                    next_val = start_val
                else:
                    next_val += 1
                # see if there are any NULLs which won't be allowed by a non-null identity
                cursor.execute(f"""select count(*) from {table_name} WHERE {col_name} IS NULL""")
                (nulls, ) = cursor.fetchone()
                # fill any nulls that exist with numbers at the end of the range
                if nulls:
                    # find the rows where nbs_id is missing, ctid is a magic postgres row id which is not long term stable so don't store it.
                    cursor.execute(f"""select ctid from {table_name} WHERE {col_name} IS NULL""")
                    ids = cursor.fetchall()
                    # create a list of (ctid, new column value)
                    revised_nbs_ids = [(ctid[0], nbsid) for ctid, nbsid in zip(ids, range(next_val, next_val + nulls))]
                    # FIXME having trouble with execute_many syntax - this rarely runs so let it loop slowly for now.
                    # update_query = f"""UPDATE {table_name} SET {col_name} = e.val FROM (VALUES %s) AS e(ct, val) WHERE {table_name}.ctid = e.ct AND {col_name} is NULL"""
                    # psycopg2.extras.execute_values(cursor, update_query, revised_nbs_ids, template=None, page_size=nulls)
                    for ct, val in revised_nbs_ids:
                        cursor.execute(f"""UPDATE {table_name} SET {col_name} = {val} WHERE ctid = '{ct}'""")
                    next_val += nulls  # update the value that future numbers should be added from
                    connection.commit()
                # now set the column to be non-null identity
                cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} SET NOT NULL")
                cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} ADD GENERATED BY DEFAULT AS IDENTITY (START WITH {next_val})")
                connection.commit()
        else:
            next_val = last_nbs_id(table_name, cursor) + 1  # the last_value return the prior number assigned then add one on the postgres nextval() call
            cursor.execute(f"""select max({NBS_ID_STR}) from {table_name}""")
            (largest_val, ) = cursor.fetchone()
            if largest_val is not None and largest_val >= next_val:  # fix manual edits that went past the auto-add number
                print(f"Fixing:{table_name} which has next sequential at {next_val} but the table has existing id at {largest_val}")
                cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} RESTART WITH {largest_val + 1}")
                connection.commit()

    if needs_id_column:
        # create nbs_id if doesn't exist
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} bigint GENERATED BY DEFAULT AS IDENTITY")
        connection.commit()
        force_restart = True

    if force_restart:
        cursor.execute(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} RESTART WITH {start_val}")
        connection.commit()
        cursor.execute(f"UPDATE {table_name} SET {col_name} = DEFAULT")
        connection.commit()
        # cursor.execute(f"update {table_name} set sid=sid+{start_val}")


def get_tablenames(cursor):
    cursor.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';""")
    tablenames = list(cursor.fetchall())
    return tablenames


def make_all_serial_columns(conn_info: ConnectionInfo, print_results: bool = False):
    """ Checks all tables that match the nbs pattern of PBx_locality_UTMxx_datum_datatype.
    Adds an nbs_id column if missing.  Modifies the nbs_id to be a unique sequence if it's not already.
    Fills the column with values starting after the last filled value or none are filled then the start value computed for the table.
    Parameters
    ----------
    conn_info
        ConnectionInfo object with database, username, password, hostname, port, tablenames
    print_results
        print the last nbs_id for each table
    Returns
    -------

    """
    connection, cursor = connection_with_retries(conn_info)
    tablenames = get_tablenames(cursor)
    for (tablename,) in tablenames:
        try:
            start = start_integer(tablename)
        except ValueError as e:
            print(f'{tablename} was not recognized as an nbs_id capable table\n    {str(e)}')
        else:
            create_identity_column(tablename, start, conn_info)
    if print_results:
        show_last_ids(cursor, tablenames)


def show_last_ids(cursor, tablenames):
    for (tablename,) in tablenames:
        cursor.execute(
            f"""select * from 
                (SELECT column_name, is_identity FROM information_schema.columns WHERE table_name = '{tablename}') as t1 
                WHERE column_name='{NBS_ID_STR}';""")
        ret = cursor.fetchone()
        if ret is not None:
            try:
                last_id = last_nbs_id(tablename, cursor)
            except Exception:
                print('failed to read for last id', tablename)
            # cursor.execute(f"""SELECT nbs_id FROM {tablename}""")
            # last_id = cursor.fetchone()
            else:
                print(tablename, last_id)
        else:
            print(tablename, "doesn't have nbs_id")

    """
    spatial_ref_sys doesn't have nbs_id
    pbg_missrvr_utm15n_lwrp (156600000,)
    pbg_missrvr_utm16n_lwrp (166600000,)
    pbg_navassa_utm18n_mllw (186000000,)
    pbc_utm19n_mllw (192003461,)
    pbg_puertorico_utm19n_mllw_sensitive (196200000,)
    pbc_utm19n_mllw_prereview (192300000,)
    pbg_gulf_utm14n_mllw_prereview (146300000,)
    pbc_utm19n_mllw_sensitive (192200000,)
    pbg_puertorico_utm20n_mllw (206000160,)
    pbg_puertorico_utm20n_mllw_sensitive (206200000,)
    pbc_utm20n_mllw (202000000,)
    pbg_utm15n_mllw_prereview (156300000,)
    pbc_utm20n_mllw_prereview (202300000,)
    pbg_gulf_utm16n_mllw_sensitive (166200000,)
    pbd_utm11n_mllw_lalb (113600017,)
    pbg_utm16n_mllw_prereview (166300000,)
    pbc_utm18n_mllw_sensitive (182200001,)
    pbd_utm11n_mllw_lalb_manual None
    scrape_tracking_pbc doesn't have nbs_id
    scrape_tracking_pbg doesn't have nbs_id
    scrape_tracking_pba_modeling doesn't have nbs_id
    scrape_tracking_pbb doesn't have nbs_id
    scrape_tracking_pbe doesn't have nbs_id
    scrape_tracking_lalb doesn't have nbs_id
    pbd_utm11n_mllw_lalb_prereview (113800000,)
    pbe_midatlantic_utm17n_mllw (174000000,)
    pbc_utm19n_mld (192600000,)
    pbe_midatlantic_utm18n_mllw (184000239,)
    pbe_midatlantic_utm19n_mllw (194000000,)
    pbg_puertorico_utm19n_mllw (196000291,)
    pbg_gulf_utm14n_mllw (146002787,)
    pbc_utm18n_mllw_enc (182400008,)
    pbg_gulf_utm14n_mllw_sensitive None
    pbb_southeast_utm16n_mllw (161000051,)
    pbg_gulf_utm15n_mllw (156008550,)
    pbg_gulf_utm15n_mllw_prereview (156300000,)
    pbb_southeast_utm17n_mllw (171000004,)
    pbb_southeast_utm18n_mllw (181000044,)
    pbg_gulf_utm15n_mllw_sensitive (156200000,)
    pbc_utm18n_hrd (182600003,)
    pbg_gulf_utm16n_mllw (166026137,)
    pbc_utm18n_mllw (182000233,)
    pbg_gulf_utm16n_mllw_prereview (166300000,)
    pbc_utm18n_mllw_prereview (182300002,)
    """
    #
    # for utm in (18, 19):
    #    for offset, extension in ([0,''], [800000, '_prereview'], [900000, '_sensitive']):
    #       tablename = f"pbc_utm{utm}n_mllw{extension}"
    #       cursor.execute(f"ALTER TABLE {tablename} ADD COLUMN {col_name} bigint GENERATED BY DEFAULT AS IDENTITY")
    #       cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} RESTART WITH {1000000*utm + offset}")
    #       cursor.execute(f"UPDATE {tablename} SET {col_name} = DEFAULT")
    # for utm in (14, 15, 16):
    #     for offset, extension in ([0, ''], [35000, '_prereview'], [38000, '_sensitive']):
    #         tablename = f"pbg_gulf_utm{utm}n_mllw{extension}"
    #         print(tablename)
    #         cursor.execute(f"create table preserial_{tablename} as (select * from {tablename})")
    #         connection.commit()
    #         try:
    #             cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} DROP IDENTITY IF EXISTS")
    #             cursor.execute(f"ALTER TABLE {tablename} DROP COLUMN {col_name}")
    #         except psycopg2.errors.UndefinedColumn:
    #             print("no pre-existing nbs_id column")
    #         cursor.execute(f"ALTER TABLE {tablename} ADD COLUMN {col_name} bigint GENERATED BY DEFAULT AS IDENTITY")
    #         cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} RESTART WITH {40000 * utm + offset}")
    #         cursor.execute(f"UPDATE {tablename} SET {col_name} = DEFAULT")
    #         connection.commit()
    #
    #
    # for utm in (15, 16):
    #     for offset, extension in ([34000, ''],):
    #         tablename = f"pbg_missrvr_utm{utm}n_lwrp{extension}"
    #         cursor.execute(f"create table preserial_{tablename} as (select * from {tablename})")
    #         connection.commit()
    #         try:
    #             cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} DROP IDENTITY IF EXISTS")
    #             cursor.execute(f"ALTER TABLE {tablename} DROP COLUMN {col_name}")
    #         except psycopg2.errors.UndefinedColumn:
    #             print("no pre-existing nbs_id column")
    #         cursor.execute(f"ALTER TABLE {tablename} ADD COLUMN {col_name} bigint GENERATED BY DEFAULT AS IDENTITY")
    #         cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} RESTART WITH {40000 * utm + offset}")
    #         cursor.execute(f"UPDATE {tablename} SET {col_name} = DEFAULT")
    #         connection.commit()

# Note - if column is already made and has NULL values, must set all data to non-null before modifying the column definition
# in psql of pgadmin -- "update pbd_utm11n_mllw_lalb set nbs_id =  0;"

# connection = connect_with_retries(database=database, user=username, password=password, host=hostname, port=port)
# cursor = connection.cursor()
# utm = 18
# col_name = 'nbs_id'
# for offset, extension in ([0, ''], [35000, '_prereview'], [38000, '_sensitive']):
#     tablename = f"pbc_utm{utm}n_mllw{extension}"
#     cursor.execute(f"ALTER TABLE {tablename} ALTER COLUMN {col_name} RESTART WITH {40000 * utm + offset}")
#     cursor.execute(f"UPDATE {tablename} SET {col_name} = DEFAULT")
#     connection.commit()

def get_nbs_records(table_name, conn_info, geom_name=None, order="", query_fields=None, exclude_fields=None):
    """ Supply a geom_name to get a value for ST_SRID({geom_name}) at the end of each sql record,
     which can be used to determine the spatial reference system used.
    """
    if _debug and conn_info.hostname is None:
        import pickle
        f = open(fr"C:\data\nbs\{table_name}.pickle", 'rb')
        records = pickle.load(f)
        fields = pickle.load(f)
        ## trim back the records to a few for testing
        # print("Thinning survey records for debugging!!!!")
        # filename_col = fields.index('from_filename')
        # id_col = fields.index('nbs_id')
        # thinned_records = []
        # for rec in records:
        #     if rec[id_col] in (12657, 12203, 12772, 10470, 10390):
        #         thinned_records.append(rec)
        # records = thinned_records
    else:
        connection, cursor = connection_with_retries(conn_info)
        if not query_fields:
            cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'")
            cols = cursor.fetchall()
            if not cols:  # even though the table name is mixed case the information_schema may have lower case in the table_name column
                cursor.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name.lower()}'")
                cols = cursor.fetchall()
            cols_names = [name for lcol in cols for name in lcol]
        else:
            cols_names = list(query_fields)
        if exclude_fields is not None:
            for exclude in exclude_fields:  # there is a "BigQuery" sql that allows for SELECT * EXCEPT X but we can't use that here yet
                for n in range(cols_names.count(exclude)):
                    cols_names.remove(exclude)
        field_str = ",".join(cols_names)
        srs = f",ST_SRID({geom_name})" if geom_name else ""
        cursor.execute(f'SELECT {field_str}{srs} FROM {table_name} {order}')
        records = cursor.fetchall()
        # the DictCursor makes an _index object that is shared by all rows which describes the mapping of name to index.
        # We will add a tablename entry at the end of the row and add the table_name to every record so it can be accessed later.
        if records:  # Fixme -- this should have a unittest to make sure this behavior doesn't break in the future
            records[0]._index['tablename'] = max(list(records[0]._index.values())) + 1
            for r in records:
                r.append(table_name)
        fields = [desc[0] for desc in cursor.description]
    return fields, records


class SurveyInfo:
    def __init__(self, from_filename, sid, data_path, decay, resolution):
        self.from_filename = from_filename
        self.sid = sid
        self.data_path = data_path
        self.decay = decay
        self.resolution = resolution


def id_to_scoring(records_lists, for_navigation_flag=(True, True), never_post_flag=(True, False), exclude=None):
    # @todo finish docs and change fields/records into class instances
    """
    Parameters
    ----------
    fields : list of lists
        list of list of field names in each respective field, record pair
    records : list of lists
        list of record lists -- matched to the fields lists
    for_navigation_flag
        tuple of two booleans, first is if navigation_flag should be checked and second is the value that is desired to be processed.
        default is (True, True) meaning use the navigation flag and only process surveys that are "for navigation"
    never_post_flag
        tuple of two booleans, first is if never_post should be checked and second is the value that is desired to be processed.
        default is (True, False) meaning use the never_post flag and only process surveys that have False (meaning yes, post)

    Returns
    -------
    (list, list, dict)
        sorted_recs, names_list, sort_dict
    """
    rec_list = []
    names_list = []
    use_for_navigation_flag, require_navigation_flag_value = for_navigation_flag
    use_never_post_flag, require_never_post_flag_value = never_post_flag
    for records in records_lists:
        # Create a dictionary that converts from the unique database ID to an ordering score
        # Basically the standings of the surveys,
        # First place is the highest decay score with a tie breaker of lowest resolution.  If both are the same they will have the same score
        # Alphabetical will have no duplicate standings (unless there is duplicate names) and first place is A (ascending alphabetical)
        # get the columns that have the important data

        # make lists of the dacay/res with survey if and also one for name vs survey id
        for rec in records:
            if use_for_navigation_flag and bool(rec['for_navigation']) != require_navigation_flag_value:
                continue
            if use_never_post_flag and bool(rec['never_post']) != require_never_post_flag_value:
                continue
            decay = rec['decay_score']
            sid = rec['nbs_id']
            if exclude and sid in exclude:
                continue
            if decay is not None:
                res = rec['manual_resolution']
                if res is None:
                    res = rec['script_resolution']
                    if res is None:
                        res = rec['manual_point_spacing']
                        if res is None:
                            res = rec['script_point_spacing']
                            if res is None:
                                print("missing res on record:", sid, rec['from_filename'])
                                continue
                path = rec['manual_to_filename']
                # A manual string can be an empty string (not Null) and also protect against it looking empty (just a space " ")
                if path is None or not path.strip():
                    path = rec['script_to_filename']
                    if path is None or not path.strip():
                        print("skipping missing to_path", sid, rec['from_filename'])
                        continue
                rec_list.append((sid, res, decay))
                # Switch to lower case, these were from filenames that I'm not sure are case sensitive
                names_list.append(SurveyInfo(rec['from_filename'].lower(), sid, path, decay, res))  # sid would be the next thing sorted if the names match
    if names_list:
        # do an ordered 2 key sort on decay then res (lexsort likes them backwards)
        rec_array = numpy.array(rec_list, dtype=numpy.float64)  # use double so not to lose precision on SID
        sorted_indices = numpy.lexsort([-rec_array[:, 1], rec_array[:, 2]])  # resolution, decay (flip the res so lowest score and largest res is first)
        sorted_recs = rec_array[sorted_indices]
        sort_val = 0
        prev_res, prev_decay = None, None
        sort_dict = {}
        # set up a dictionary that has the sorted value of the decay followed by resolution
        for n, (sid, res, decay) in enumerate(sorted_recs):
            # don't incremenet when there was a tie, this allows the next sort criteria to be checked
            if res != prev_res or decay != prev_decay:
                sort_val += 1
            prev_res = res
            prev_decay = decay
            try:
                sort_dict[int(sid)] = [sort_val]
            except ValueError:
                raise ValueError(f"Could not convert {sid} to int, was nbs_id set correctly?")
        # the NBS sort order then uses depth after decay but before alphabetical, so we can't merge the name sort with the decay+res
        # add a second value for the alphabetical naming which is the last resort to maintain constistency of selection
        #
        # sort the names so we can use an integer to use for sorting by name
        names_list.sort(key=lambda x:[x.from_filename, x.sid])
        for n, info in enumerate(names_list):
            sort_dict[int(info.sid)].append(n)
    else:
        sorted_recs, names_list, sort_dict = [], [], {}
    return sorted_recs, names_list, sort_dict


def nbs_survey_sort(id_to_score, pts, existing_arrays, pts_col_offset=0, existing_col_offset=0):
    return nbs_sort_values(id_to_score, pts[LayersEnum.CONTRIBUTOR + pts_col_offset], pts[LayersEnum.ELEVATION + pts_col_offset],
                           existing_arrays[LayersEnum.CONTRIBUTOR+existing_col_offset], existing_arrays[LayersEnum.ELEVATION+existing_col_offset])


def nbs_sort_values(id_to_score, new_contrib, new_elev, accum_contrib, accum_elev):
    # return arrays that merge_arrays will use for sorting.
    # basically the nbs sort is 4 keys: Decay Score, resolution, depth, alphabetical.
    # Decay and resolution get merged into one array since they are true for all points of the survey while depth varies with position.
    # alphabetical is a final tie breaker to make sure the same contributor is picked in the cases where the first three tie.

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

    # find all the contributors to look up
    unique_contributors = numpy.unique(accum_contrib[~numpy.isnan(accum_contrib)])
    # make arrays to store the integer scores in
    existing_decay_and_res = accum_contrib.copy()
    existing_alphabetical = accum_contrib.copy()
    # for each unique contributor fill with the associated decay/resolution score and the alphabetical score
    for contrib in unique_contributors:
        int_contrib = numpy.frombuffer(numpy.array(contrib, dtype=numpy.float32).tobytes(), dtype=numpy.int32)[0]
        try:
            existing_decay_and_res[accum_contrib == contrib] = id_to_score[int_contrib][0]
            existing_alphabetical[accum_contrib == contrib] = id_to_score[int_contrib][1]
        except KeyError as e:
            raise KeyError(f"failing to find {int_contrib} in id_to_score, the database is likely corrupt")
    # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
    unique_pts_contributors = numpy.unique(new_contrib[~numpy.isnan(new_contrib)])
    decay_and_res_score = new_contrib.copy()
    alphabetical = new_contrib.copy()
    for contrib in unique_pts_contributors:
        int_contrib = numpy.frombuffer(numpy.array(contrib, dtype=numpy.float32).tobytes(), dtype=numpy.int32)[0]
        decay_and_res_score[new_contrib == contrib] = id_to_score[int_contrib][0]
        alphabetical[new_contrib == contrib] = id_to_score[int_contrib][1]

    return numpy.array((decay_and_res_score, new_elev, alphabetical)), \
           numpy.array((existing_decay_and_res, accum_elev, existing_alphabetical)), \
           (False, False, False)


def make_contributor_csv(filename, band_num, conn_info):
    # nbs_postgres.make_contributor_csv(pathlib.Path(root).joinpath("4_utm.tif"), 3, "pbc19_mllw_metadata", 'metadata', None, None, None, None)
    # tile_name = r"C:\data\nbs\test_data_output\test_pbc_19_exact_multiprocesing_locks\exports\4m\4_utm.tif"
    # fields, records = get_nbs_records("pbc19_mllw_metadata", 'metadata', None, None, None, None)
    filename = pathlib.Path(filename)
    all_fields = []
    all_records = []
    # FIXME -- this will fail if fields is not the same for all record groups, make a class that encapsulates a record
    for table_name in conn_info.tablenames:
        fields, records = get_nbs_records(table_name, conn_info)
        all_records.append(records)
        all_fields.append(fields)
    sorted_recs, names_list, sort_dict = id_to_scoring(all_records)

    ds = gdal.Open(str(filename))
    cb = ds.GetRasterBand(band_num)
    contributors = cb.ReadAsArray()
    unique_contributors = numpy.unique(contributors[~numpy.isnan(contributors)])
    records_dict = {}
    for records in all_records:
        records_dict.update({rec[-1]: rec for rec in records})
    res_decay_dict = {sid: (res, decay) for sid, res, decay in sorted_recs}

    csv = open(filename.with_name(filename.stem+"_contrib.csv"), "w")
    for n, contrib_number in enumerate(unique_contributors):
        rec = records_dict[int(contrib_number)]
        man_date = rec['manual_start_date']
        if man_date is not None:
            survey_date = man_date
        else:
            survey_date = rec['script_start_date']
        man_catzoc = rec['manual_catzoc']
        if man_catzoc is not None and man_catzoc.strip():
            survey_catzoc = man_catzoc
        else:
            survey_catzoc = rec['script_catzoc']
        res, decay = res_decay_dict[int(contrib_number)]
        score_pos, alpha_pos = sort_dict[int(contrib_number)]
        s = f'{int(contrib_number)},res:{res};sort:{score_pos};alpha:{alpha_pos},{rec["from_filename"]},{survey_date.strftime("%Y%m%d")},{decay},{survey_catzoc}\n'
        csv.write(s)


def choose_record_value(rec, indices):
    val = None
    for index in indices:
        temp = rec[index]
        if temp is None:
            continue
        if isinstance(temp, str):
            if not temp.strip():
                continue
        val = temp
        break
    return val


def get_transform_metadata(fields_lists, records_lists):
    """ Looks at the script and manual fields for horizontal and vertical spatial reference systems.
    Chooses the proper values and returns a dictionary of nbs_id to reference system information.
    """
    transform_metadata = {}

    for fields, records in zip(fields_lists, records_lists):
        if fields:
            # Create a dictionary that converts from the unique database ID to an ordering score
            # make lists of the dacay/res with survey if and also one for name vs survey id
            for rec in records:
                metadata = {
                    'to_horiz_frame': choose_record_value(rec, ('manual_to_horiz_frame', 'script_to_horiz_frame')),
                    'to_horiz_type': choose_record_value(rec, ('manual_to_horiz_type', 'script_to_horiz_type')),
                    'to_horiz_key': choose_record_value(rec, ('manual_to_horiz_key', 'script_to_horiz_key')),
                    'vert_uncert_fixed': choose_record_value(rec, ('manual_vert_uncert_fixed', 'script_vert_uncert_fixed')),
                    'vert_uncert_vari': choose_record_value(rec, ('manual_vert_uncert_vari', 'script_vert_uncert_vari')),
                }
                transform_metadata[rec['nbs_id']] = metadata
    return transform_metadata

def get_records(conn_info, cache_dir=None, query_fields=None, exclude_fields=('provided_coverage',)):
    """ Open multiple NBS metadata tables and return combined lists/dictionaries of records and a scoring compare_callback function
    to be used with WorldDatabase insert functions.

    Parameters
    ----------
    conn_info
    cache_dir

    Returns
    -------
    sorted_recs, names_list, sort_dict, comp
    """
    all_fields = []
    all_records = []
    for table_name in conn_info.tablenames:
        try:
            fields, records = get_nbs_records(table_name, conn_info, query_fields=query_fields, exclude_fields=exclude_fields)
            if cache_dir:
                cache_fname = pathlib.Path(cache_dir).joinpath(f"last_used_{conn_info.database}_{table_name}.pickle")
                try:
                    lock = nbs_locks.Lock(cache_fname, mode="wb", timeout=180, fail_when_locked=False, flags=nbs_locks.LockFlags.EXCLUSIVE)
                    outfile = lock.acquire(check_interval=1 + random.randrange(0, 100) / 100.0)
                    pickle.dump(fields, outfile)
                    pickle.dump(records, outfile)
                    lock.release()
                except nbs_locks.AlreadyLocked:
                    print("failed to cache the metadata table")
        except psycopg2.errors.UndefinedTable:
            print(f"table {table_name} not found")
            records, fields = [], []
        all_records.append(records)
        all_fields.append(fields)
    return all_fields, all_records


def get_sorting_records(conn_info, for_navigation_flag=(True, True), cache_dir=None):
    all_fields, all_records = get_records(conn_info, cache_dir=cache_dir)
    return get_sorting_info(all_fields, all_records, for_navigation_flag=for_navigation_flag)


def get_sorting_info(all_fields, all_records, for_navigation_flag=(True, True), exclude=None):
    if any([bool(table_recs) for table_recs in all_records]):
        sorted_recs, names_list, sort_dict = id_to_scoring(all_records, for_navigation_flag=for_navigation_flag, exclude=exclude)
        comp = partial(nbs_survey_sort, sort_dict)
    else:
        sorted_recs, names_list, sort_dict, comp = [], [], {}, None
    return sorted_recs, names_list, sort_dict, comp


