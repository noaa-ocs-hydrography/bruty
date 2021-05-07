# import sys
# sys.path.append(r"C:\Git_Repos\nbs")
# sys.path.append(r"C:\Git_Repos\bruty")
from functools import partial

import numpy

from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review.meta_review import database_has_table, split_URL_port
from nbs.bruty.raster_data import TiffStorage, LayersEnum, RasterData, affine, inv_affine, affine_center, arrays_dont_match
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.bruty.world_raster_database import WorldDatabase, UTMTileBackend

_debug = True

def make_serial_column(table_id, table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434', big=False):
    connection = connect_with_retries(database=database, user=username, password=password, host=hostname, port=port)
    cursor = connection.cursor()
    # used admin credentials for this
    # cursor.execute("create table serial_19n_mllw as (select * from pbc_utm19n_mllw)")
    # connection.commit()
    if big:
        serial_str = "bigserial"  # use bigserial if 64bit is needed
        bit_shift = 32  # allows 4 billion survey IDs and 4 billion table IDs
    else:
        serial_str = "serial"  # 32bit ints
        bit_shift = 20  # allows for one million survey IDs and 4096 table IDs
    start_val = table_id << bit_shift
    cursor.execute(f'ALTER TABLE {table_name} ADD column sid {serial_str};')
    cursor.execute(f'ALTER SEQUENCE {table_name}_sid_seq RESTART WITH 10000')
    cursor.execute(f"update {table_name} set sid=sid+{start_val}")
    connection.commit()

def get_nbs_records(table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    if _debug and hostname is None:
        import pickle
        f = open(r"C:\data\nbs\pbc19_mllw_metadata.pickle", 'rb')
        records = pickle.load(f)
        fields = pickle.load(f)
    else:
        connection = connect_with_retries(database=database, user=username, password=password, host=hostname, port=port)
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM {table_name}')
        records = cursor.fetchall()
        fields = [desc[0] for desc in cursor.description]
    return fields, records

def survey_sort(id_to_score, pts, existing_arrays):

    return (decay_and_res_score, elevation, alphabetical), (existing_decay_and_res, existing_elevation, existing_alpabetical), (False, False, False)

def id_to_scoring(fields, records):
    # get the columns that have the important data
    decay_col = fields.index("decay_score")
    script_res_col = fields.index('script_resolution')
    manual_res_col = fields.index('manual_resolution')
    filename_col = fields.index('from_filename')
    id_col = fields.index('sid')
    rec_list = []
    names_list = []
    # make lists of the dacay/res with survey if and also one for name vs survey id
    for rec in records:
        if rec[decay_col] is not None:
            res = rec[manual_res_col]
            if res is None:
                res = rec[script_res_col]
            decay = rec[decay_col]
            sid = rec[id_col]
            rec_list.append((sid, res, decay))
            names_list.append((rec[filename_col], sid))
    # sort the names so we can use an integer to use for sorting by name
    names_list.sort()
    # do an ordered 2 key sort on decay then res (lexsort likes them backwards)
    rec_array = numpy.array(rec_list)
    sorted_indices = numpy.lexsort(rec_array[1], rec_array[2])
    sorted_recs = rec_array[sorted_indices]
    sort_val = 0
    prev_res, prev_decay = None, None
    sort_dict = {}
    # set up a dictionary that has the sorted value of the decay followed by resolution
    for n, (sid, res, decay) in enumerate(sorted_recs):
        # don't incremenet when there was a tie, this allows the next sort criteria to be checked
        if res != prev_res or decay != prev_decay:
            sort_val+=1
        sort_dict[sid] = [sort_val]
    # the NBS sort order then uses depth after decay but before alphabetical, so we can't merge the name sort with the decay+res
    # add a second value for the alphabetical naming which is the last resort to maintain constistency of selection
    for n, (filename, sid) in enumerate(names_list):
        sort_dict[sid].append(n)
    return sort_dict

def process_nbs_database(self, world_db_path, table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    fields, records = get_nbs_records(table_name, database, username, password, hostname=hostname, port=port)
    sort_dict = id_to_scoring(fields, records)
    try:
        db = WorldDatabase.open(world_db_path)
    except FileNotFoundError:
        epsg
        db = WorldDatabase(UTMTileBackend(epsg, RasterHistory, DiskHistory, TiffStorage, world_db_path))  # NAD823 zone 19.  WGS84 would be 32619
    path_col = fields.index('script_to_filename')
    id_col = fields.index('sid')
    comp = partial(survey_sort, sort_dict)
    for rec in records:
        survey_path = rec[path_col]
        sid = rec[id_col]
        db.insert_survey(survey_path, contrib_id=sid, compare_func=comp)



if __name__ == '__main__':
    db_path = r"c:\data\nbs\test_db"
    db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, db_path))  # NAD823 zone 19.  WGS84 would be 32619



    URL_FILENAME = r"c:\data\nbs\postgres_hostname.txt"
    CREDENTIALS_FILENAME = r"c:\data\nbs\postgres_scripting.txt"
    if _debug:
        hostname = None
        port = None
        username = None
        password = None
    else:
        with open(URL_FILENAME) as hostname_file:
            url = hostname_file.readline()
        hostname, port = split_URL_port(url)

        with open(CREDENTIALS_FILENAME) as database_credentials_file:
            username, password = [line.strip() for line in database_credentials_file][:2]

    table_name = 'serial_19n_mllw'
    database = 'metadata'
    process_nbs_database(db_path, table_name, database, username, password, hostname, port)