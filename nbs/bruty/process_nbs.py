# import sys
# sys.path.append(r"C:\Git_Repos\nbs")
# sys.path.append(r"C:\Git_Repos\bruty")
import os
import pathlib
import shutil
from functools import partial

import numpy

from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review.meta_review import database_has_table, split_URL_port
from nbs.bruty.raster_data import TiffStorage, LayersEnum
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.bruty.world_raster_database import WorldDatabase, UTMTileBackend
from nbs.bruty.utils import onerr

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


def nbs_survey_sort(id_to_score, pts, existing_arrays):
    # return arrays that merge_arrays will use for sorting.
    # basically the nbs sort is 4 keys: Decay Score, resolution, depth, alphabetical.
    # Decay and resolution get merged into one array since they are true for all points of the survey while depth varies with position.
    # alphabetical is a final tie breaker to make sure the same contributor is picked in the cases where the first three tie.

    # find all the contributors to look up
    all_contributors = existing_arrays[LayersEnum.CONTRIBUTOR]
    unique_contributors = numpy.unique(all_contributors[~numpy.isnan(all_contributors)])
    # make arrays to store the integer scores in
    existing_decay_and_res = all_contributors.copy()
    existing_alphabetical = all_contributors.copy()
    # for each unique contributor fill with the associated decay/resolution score and the alphabetical score
    for contrib in unique_contributors:
        existing_decay_and_res[all_contributors == contrib] = id_to_score[contrib][0]
        existing_alphabetical[all_contributors == contrib] = id_to_score[contrib][1]
    existing_elevation = existing_arrays[LayersEnum.ELEVATION]
    # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
    pts_contributors = pts[LayersEnum.CONTRIBUTOR + 2]
    unique_pts_contributors = numpy.unique(pts_contributors[~numpy.isnan(pts_contributors)])
    decay_and_res_score = pts_contributors.copy()
    alphabetical = pts_contributors.copy()
    for contrib in unique_pts_contributors:
        decay_and_res_score[pts_contributors == contrib] = id_to_score[contrib][0]
        alphabetical[pts_contributors == contrib] = id_to_score[contrib][1]
    elevation = pts[LayersEnum.ELEVATION + 2]

    return numpy.array((decay_and_res_score, elevation, alphabetical)), \
           numpy.array((existing_decay_and_res, existing_elevation, existing_alphabetical)), \
           (False, False, False)


def id_to_scoring(fields, records):
    # Create a dictionary that converts from the unique database ID to an ordering score
    # Basically the standings of the surveys,
    # First place is the highest decay score with a tie breaker of lowest resolution.  If both are the same they will have the same score
    # Alphabetical will have no duplicate standings (unless there is duplicate names) and first place is A (ascending alphabetical)
    # get the columns that have the important data
    decay_col = fields.index("decay_score")
    script_res_col = fields.index('script_resolution')
    manual_res_col = fields.index('manual_resolution')
    filename_col = fields.index('from_filename')
    path_col = fields.index('script_to_filename')
    manual_path_col = fields.index('manual_to_filename')
    id_col = fields.index('sid')
    rec_list = []
    names_list = []
    # make lists of the dacay/res with survey if and also one for name vs survey id
    for rec in records:
        decay = rec[decay_col]
        sid = rec[id_col]
        if decay is not None:
            res = rec[manual_res_col]
            if res is None:
                res = rec[script_res_col]
                if res is None:
                    print("missing res on record:", sid, rec[filename_col])
                    continue
            path = rec[manual_path_col]
            if path is None:
                path = rec[path_col]
                if path is None:
                    print("skipping missing to_path", sid, rec[filename_col])
                    continue
            rec_list.append((sid, res, decay))
            # Switch to lower case, these were from filenames that I'm not sure are case sensitive
            names_list.append((rec[filename_col].lower(), sid, path))  # sid would be the next thing sorted if the names match
    # sort the names so we can use an integer to use for sorting by name
    names_list.sort()
    # do an ordered 2 key sort on decay then res (lexsort likes them backwards)
    rec_array = numpy.array(rec_list)
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
        sort_dict[sid] = [sort_val]
    # the NBS sort order then uses depth after decay but before alphabetical, so we can't merge the name sort with the decay+res
    # add a second value for the alphabetical naming which is the last resort to maintain constistency of selection
    for n, (filename, sid, path) in enumerate(names_list):
        sort_dict[sid].append(n)
    return sorted_recs, names_list, sort_dict


def process_nbs_database(world_db_path, table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    fields, records = get_nbs_records(table_name, database, username, password, hostname=hostname, port=port)
    sorted_recs, names_list, sort_dict = id_to_scoring(fields, records)
    try:
        db = WorldDatabase.open(world_db_path)
    except FileNotFoundError:
        epsg
        db = WorldDatabase(UTMTileBackend(epsg, RasterHistory, DiskHistory, TiffStorage, world_db_path))  # NAD823 zone 19.  WGS84 would be 32619
    path_col = fields.index('script_to_filename')
    id_col = fields.index('sid')
    comp = partial(nbs_survey_sort, sort_dict)
    total = 0
    print('------------   changing paths !!!!!!!!!!')
    for i, (filename, sid, path) in enumerate(names_list):
        copy_data = False
        path_e = path.lower().replace('\\\\nos.noaa\\ocs\\hsd\\projects\\nbs\\nbs_data\\pbc_northeast_utm19n_mllw',
                                      r'E:\Data\nbs\PBC_Northeast_UTM19N_MLLW')
        path_c = path.lower().replace('\\\\nos.noaa\\ocs\\hsd\\projects\\nbs\\nbs_data\\pbc_northeast_utm19n_mllw',
                                      r'C:\Data\nbs\PBC_Northeast_UTM19N_MLLW')
        path = path_c
        if copy_data:
            try:
                total += os.stat(path_e).st_size
                # os.makedirs(os.path.dirname(path_c), exist_ok=True)
                # shutil.copy(path_e, path_c)
                if path_e.endswith("csar"):
                    total += os.stat(path_e + "0").st_size
                    # os.makedirs(os.path.dirname(path_c+"0"), exist_ok=True)
                    shutil.copy(path_e + "0", path_c + "0")
            except FileNotFoundError:
                print("File missing", sid, path)
        # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
        if _debug:
            if i>10:
                break
        db.insert_survey(path, override_epsg=db.db.epsg, contrib_id=sid, compare_callback=comp)
        print('inserted', path)

    print('data MB:', total / 1000000)


if __name__ == '__main__':
    data_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('tests').joinpath("test_data_output")

    def make_clean_dir(name):
        use_dir = data_dir.joinpath(name)
        if os.path.exists(use_dir):
            shutil.rmtree(use_dir, onerror=onerr)
        os.makedirs(use_dir)
        return use_dir

    db_path = data_dir.joinpath(r"test_pbc_19_db")
    # db_path = make_clean_dir(r"test_pbc_19_db")  # reset the database

    if not os.path.exists(db_path.joinpath("wdb_metadata.json")):
        db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, db_path))  # NAD823 zone 19.  WGS84 would be 32619
        del db

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
