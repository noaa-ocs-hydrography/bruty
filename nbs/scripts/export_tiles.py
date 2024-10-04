""" Run tile_export.py on each tile listed in the combine_spec tables (views) as defined by the build flag and the config parameters specified
"""
import datetime
import multiprocessing
import tempfile
import sqlite3
import traceback

import psutil

import os
import time
import re
import sys
import subprocess
import pathlib
from shapely import wkt, wkb
from osgeo import ogr, osr, gdal
import pickle
from functools import partial

import psycopg2

import fuse_dev
import nbs.bruty

from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review import meta_review
from nbs.bruty.nbs_locks import Lock, AlreadyLocked, LockFlags
from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, ConnectionInfo, connect_params_from_config, make_contributor_csv
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.exceptions import UserCancelled, BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
# import locks from world_raster_database in case we are in debug mode and have turned locks off
from nbs.bruty.world_raster_database import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
from nbs.bruty.utils import remove_file, user_action, popen_kwargs, ConsoleProcessTracker, QUIT, HELP
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config
from nbs.bruty.tile_export import combine_and_export, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db, TileToProcess, TileProcess

LOGGER = get_logger('nbs.bruty.export')
CONFIG_SECTION = 'export'

interactive_debug = False
profiling = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")

def launch(config_path, cache_file, tile_cache, export_time,
           new_console=True, env_path=r'D:\languages\miniconda3\Scripts\activate', env_name='NBS', tile_id=(0, 0),
           decimals=None, minimized=False, remove_cache=False, fingerprint=""):

    """ for_navigation_flag = (use_nav_flag, nav_flag_value)
    """
    # either launch in a new console with subprocess.popen or use multiprocessing.Process.  Could also consider dask.
    if new_console:
        # spawn a new console, activate a python environment and run the combine.py script with appropriate arguments
        bruty_code = nbs.bruty.__path__._path[0]
        bruty_root = str(pathlib.Path(bruty_code).parent.parent)
        nbs_code = fuse_dev.__path__._path[0]
        nbs_code = str(pathlib.Path(nbs_code).parent)
        restore_dir = os.getcwd()
        # os.chdir(pathlib.Path(__file__).parent.joinpath("..", "bruty"))  # change directory so we can run the tile_export script
        # FIXME - hack overriding nbs and bruty paths
        # looks like activate is overwriting the pythonpath, so specify it after the activate command
        args = ['cmd.exe', '/K', 'set', f'pythonpath=&', 'set', 'TCL_LIBRARY=&', 'set',
                'TIX_LIBRARY=&', 'set', 'TK_LIBRARY=&', env_path, env_name, '&',
                'set', f'pythonpath={nbs_code};{bruty_root}&', 'python']
        if profiling:
            args.extend(["-m", "cProfile", "-o", str(pathlib.Path(config_path).parent) + f"\\timing{tile_id[0]}_{tile_id[1]}.profile"])
        script_path = os.path.join(bruty_code, "tile_export.py")
        args.extend([script_path, "-b", config_path, "-t", export_time, "-c", cache_file, "-i", tile_cache])
        if remove_cache:
            args.extend(["--remove_cache"])
        if decimals is not None:
            args.extend(["-d", str(decimals)])
        if fingerprint:
            args.extend(['-f', fingerprint])
        # exiter closes the console if there was no error code, keeps it open if there was an error
        args.extend(["&", r"..\scripts\exiter.bat", f"{SUCCEEDED}", f"{TILE_LOCKED}"])  # note && only joins commands if they succeed, so just use one ampersand
        # because we are launching separate windows we can't use the subprocess.poll and returncode.
        # maybe we should switch to just logs and not leave a window on the screen for the user to see
        # that would make it easier to check the returncode
        proc = subprocess.Popen(args, **popen_kwargs(activate=False, minimize=minimized))
        # os.chdir(restore_dir)
        ret = proc.pid, script_path
    else:
        raise Exception("same console multiprocessing is untested")
        # could actually do the same subprocess.Popen in the same console, or multiprocessing which would remove need for psutil, or use dask(?)
        multiprocessing.Process(target=process_nbs_database, args = (db, conn_info), kwargs={'for_navigation_flag':(use_nav_flag, nav_flag_value),
                             'override_epsg':override, 'extra_debug':debug_config})
    return ret

def get_metadata(tile_info, conn_info, use_bruty_cached="", use_caches={}):
    metadata_fields = []
    metadata_records = []
    # conn_info.database = "metadata"
    for dtype in (PREREVIEW, REVIEWED, ENC, GMRT, SENSITIVE):
        query_database = False
        tablename = tile_info.metadata_table_name(dtype)
        if use_bruty_cached and use_caches[dtype]:
            root_dir = pathlib.Path(use_bruty_cached)
            db_name = root_dir.joinpath(tile_info.bruty_db_name(dtype, True))
            cache_fname = pathlib.Path(db_name).joinpath(f"last_used_{conn_info.database}_{tablename}.pickle")
            try:
                cache_file = open(cache_fname, "rb")
                mfields = pickle.load(cache_file)
                mrecords = pickle.load(cache_file)
            except:
                query_database = True
        else:
            query_database = True
        if query_database:
            try:
                mfields, mrecords = get_nbs_records(tablename, conn_info)
            except psycopg2.errors.UndefinedTable:
                print(f"{tablename} doesn't exist.")
                continue
        metadata_records.append(mrecords)
        metadata_fields.append(mfields)
    # @todo move this record stuff into a function in nbs_postgres.py
    all_meta_records = {}
    all_simple_records = {}
    for n, meta_table_recs in enumerate(metadata_records):
        # id_col = metadata_fields[n].index('nbs_id')
        for record_dict in meta_table_recs:
            # record_dict = dict(zip(metadata_fields[n], record))
            simple_record = meta_review.MetadataDatabase._simplify_record(record_dict)
            # re-casts the db values into other/desired types but then we just want a plain dictionary for pickle
            simple_fuse_record = dict(meta_review.records_to_fusemetadata(simple_record))
            all_simple_records[record_dict['nbs_id']] = simple_fuse_record
        all_meta_records.update({rec['nbs_id']: rec for rec in meta_table_recs})

    sorted_recs, names_list, sort_dict = id_to_scoring(metadata_records, for_navigation_flag=(False, False),
                                                       never_post_flag=(False, False))
    return all_simple_records, sort_dict


def remove_finished_processes(tile_processes, remaining_tiles, max_tries):
    for key in list(tile_processes.keys()):
        if not tile_processes[key].console_process.is_running():
            try:
                retry = False
                if tile_processes[key].succeeded():
                    reason = "finished"
                # 4 is the code for LOCKED
                elif tile_processes[key].finish_code() == TILE_LOCKED:
                    reason = "LOCKED"
                    tile_processes[key].clear_finish_code()  # remove record so we don't just pile up a lot of needless records
                    retry = True
                    # reduce the counter by one so we don't stop trying just because the tile was locked
                    remaining_tiles[key][1] -= 1
                elif remaining_tiles[key][1] + 1 >= max_tries:
                    reason = "failed too many times"
                else:
                    reason = "failed and will retry"
                    retry = True
            except sqlite3.OperationalError as e:
                # this is happening when the network drive loses connection, not sure if it is recoverable.
                # treat it like an unhandled exception in the subprocess - retry and increment the count
                # also wait a moment to let the network recover in case that helps
                msg = traceback.format_exc()
                LOGGER.warning(f"Exception accessing bruty db for {remaining_tiles[key][0]} {key.dtype} res={key.resolution}:\n{msg}")
                reason = "failed with a sqlite Operational error"
                retry = True
                time.sleep(5)
            LOGGER.info(f"Export {reason} for {remaining_tiles[key][0]} {key.dtype} res={key.resolution}")
            if retry:
                remaining_tiles[key][1] += 1  # set the tile to try again later and note that it is retrying
            else:
                del remaining_tiles[key]  # remove the tile from the list of tiles to process in the future
            del tile_processes[key]  # remove the instance from our list of active processes


def do_keyboard_actions(remaining_tiles, tile_processes):
    action = user_action()
    if action == QUIT:
        raise UserCancelled("User pressed keyboard quit")
    elif action == HELP:
        descr_of_running_processes = '\n'.join([f'{prc.tile_info} {k.resolution} {k.dtype}' for k, prc in tile_processes.items()])
        print(f"Remaining tiles: {len(remaining_tiles)}\nCurrently running:\n{descr_of_running_processes}")


def main(config):
    """
    Returns
    -------

    """
    debug_config = config.getboolean('DEBUG', False)
    port = config.get('lock_server_port', None)
    use_locks(port)
    ignore_pids = psutil.pids()
    env_path = config.get('environment_path', r'D:\languages\miniconda3\Scripts\activate')
    env_name = config.get('environment_name', r'NBS')
    decimals = config.getint('decimals', None)
    minimized = config.getboolean('MINIMIZED', False)
    use_cached_meta = config.getboolean('USE_CACHED_METADATA', False)
    use_cached_enc_meta = config.getboolean('USE_CACHED_ENC_METADATA', False)
    conn_info = connect_params_from_config(config)
    # conn_info.database = 'metadata'
    raise "Not finished updating export for new table structure - like tile_info having .res, .datatype, .for_nav"
    tile_list = list(iterate_tiles_table(config))
    tile_list.sort(key=lambda t: t.pb)
    max_tries = config.getint('max_tries', 3)
    max_processes = config.getint('processes', 5)
    try:
        if debug_config:
            user_res = [float(dt.strip()) for dt in parse_multiple_values(config['res'])]
        else:
            user_res = None
    except KeyError:
        user_res = None
    # make a dictionary to track which tiles are still remaining.
    # Start with all the possible export combinations of tile+res (all dtypes are used for export unlike combine)
    # make sure to delete the entries when they finish
    remaining_tiles = {}
    for tile_info in tile_list:
        res1, closing1 = tile_info.resolution, tile_info.closing_dist
        if interactive_debug and debug_config and max_processes < 2:
            if user_res and res1 not in user_res:
                continue
        remaining_tiles[TileToProcess(tile_info.hash_id(res1), res1, closing=closing1)] = [tile_info, 0]

    cached_metadata_pb = None
    files_to_delete = []
    time_format = "%Y%m%d_%H%M%S"
    export_time = datetime.datetime.now().strftime(time_format)
    quitter = False
    tile_processes = {}
    try:
        while remaining_tiles:
            for current_tile in list(remaining_tiles):
                if current_tile in tile_processes:  # already running this one.  Wait til finished or stopped to retry if needed.
                    continue

                do_keyboard_actions(remaining_tiles, tile_processes)
                remove_finished_processes(tile_processes, remaining_tiles, max_tries)
                # waiting in here allows the tiles to be reliably processed in order (user request)
                while len(tile_processes) >= max_processes:  # wait for at least one process to finish
                    time.sleep(10)
                    do_keyboard_actions(remaining_tiles, tile_processes)
                    remove_finished_processes(tile_processes, remaining_tiles, max_tries)

                try:
                    tile_info = remaining_tiles[current_tile][0]
                except KeyError:  # the tile was running and in the list but got removed while we were looping on the cached list of tiles
                    continue

                # Lock all the possible combine databases so we can export safely -- this is only effective on one OS
                # so if linux is combining and Windows tries to export they won't see each others locks.
                # We're adding a postgres lock which is cross platform inside the combine.py and tile_export.py
                # to help this problem (but network errors may lose the postgres connection)
                locks = []
                try:
                    for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                        for for_nav in (True, False):
                            combine_path = pathlib.Path(config['data_dir']).joinpath(tile_info.bruty_db_name(dtype, for_nav))
                            lock_path = WorldDatabase.metadata_filename_in_dir(combine_path).with_suffix(".lock")
                            os.makedirs(lock_path.parent, exist_ok=True)
                            lock = Lock(lock_path, fail_when_locked=True, flags=LockFlags.SHARED | LockFlags.NON_BLOCKING)
                            lock.acquire()
                            locks.append(lock)
                except AlreadyLocked:  # release locks but don't remove from the list so we come back to it later
                    LOGGER.debug(f"Tile currently locked - {tile_info.full_name}")
                    for lock in locks:
                        lock.release()
                else:  # try to export since all the combines locked
                    # if the product branch (pb) is new then read and cache the data in a pickle file to pass to the new console
                    # Also, if cached metadata from the combine is desired then read it every time
                    # since we don't know if all combines were done with the same metadata tables.
                    if tile_info.pb != cached_metadata_pb or tile_info.locality != cached_locality or \
                            tile_info != cached_utm or use_cached_meta or use_cached_enc_meta:
                        # determine the location of the cached metadata if any is to be used
                        cache_dir = config['data_dir'] if use_cached_meta or use_cached_enc_meta else ""
                        # determine which datatypes will use the cache
                        # (enc will normally use cached data, it's being auto-updated regularly which would create a need to constantly reprocess it)
                        cache_flags = {dtype: use_cached_meta for dtype in (REVIEWED, PREREVIEW, GMRT, SENSITIVE)}
                        cache_flags[ENC] = use_cached_enc_meta or use_cached_meta
                        # read the metadata, either from postgres server or disk cache based on the caches flags
                        all_simple_records, sort_dict = get_metadata(tile_info, conn_info, use_bruty_cached=cache_dir, use_caches=cache_flags)
                        fobj, cache_file = tempfile.mkstemp(".cache.pickle")
                        os.close(fobj)
                        if not use_cached_meta or interactive_debug:
                            files_to_delete.append(cache_file)  # add the filename to a list to delete later
                        # put the records into a file for the export to use
                        meta_cache = open(cache_file, "wb")
                        pickle.dump(all_simple_records, meta_cache)
                        pickle.dump(sort_dict, meta_cache)
                        meta_cache.close()
                        cached_metadata_pb = tile_info.pb
                        cached_locality = tile_info.locality
                        cached_utm = tile_info.utm

                    # closing_dist = tile_record[closing_index]

                    # to make a full utm zone database, take the tile_info and set geometry and tile to None.
                    # need to make a copy first
                    # tile_info.geometry, tile_info.tile = None, None
                    # full_db = create_world_db(config['data_dir'], tile_info, dtype, nav_flag_value)
                    if interactive_debug and debug_config and max_processes < 2:
                        comp = partial(nbs_survey_sort, sort_dict)
                        combine_and_export(config, tile_info, all_simple_records, comp, export_time, decimals)
                        del remaining_tiles[current_tile]
                    else:
                        try:
                            # store+check the success code in the 'qualified for navigation' database
                            db = create_world_db(config['data_dir'], tile_info, REVIEWED, True)
                        except sqlite3.OperationalError as e:
                            # this happened as a network issue - we will skip it for now and come back and give the network some time to recover
                            # but we will count it as a retry in case there is a corrupted file or something
                            msg = traceback.format_exc()
                            LOGGER.warning(f"Exception accessing bruty db for {tile_info} {current_tile.resolution}m {current_tile.dtype}:\n{msg}")
                            time.sleep(5)
                            remaining_tiles[current_tile][1] += 1
                            if remaining_tiles[current_tile][1] > max_tries:
                                del remaining_tiles[current_tile]
                        else:
                            fobj, tile_cache = tempfile.mkstemp(".tile.pickle")
                            os.close(fobj)
                            tile_cache_file = open(tile_cache, "wb")
                            pickle.dump(tile_info, tile_cache_file)
                            tile_cache_file.close()
                            files_to_delete.append(tile_cache)  # mark this to delete later
                            LOGGER.info(f"exporting {tile_info.full_name}")
                            fingerprint = str(current_tile.hash_id) + "_" + datetime.datetime.now().isoformat()

                            pid, script_path = launch(config._source_filename, cache_file, tile_cache, export_time, env_path=env_path,
                                         env_name=env_name, tile_id=(tile_info.tile, tile_info.resolution), decimals=decimals, minimized=minimized,
                                         remove_cache=use_cached_meta, fingerprint=fingerprint)
                            running_process = ConsoleProcessTracker(["python", fingerprint, script_path])
                            if running_process.console.last_pid != pid:
                                LOGGER.warning(f"Process ID mismatch {pid} did not match the found {running_process.console.last_pid}")
                            # print(running_process.console.is_running(), running_process.app.is_running(), running_process.app.last_pid)
                            tile_processes[current_tile] = TileProcess(running_process, tile_info, db, fingerprint, locks)
                    del locks  # releases if not stored in the tile_processes

            # remove finished processes from the list or this becomes an infinite loop
            remove_finished_processes(tile_processes, remaining_tiles, max_tries)
            if len(remaining_tiles) > 0:  # not all files have finished, give time for processing or locks to finish
                time.sleep(10)
    except UserCancelled:
        pass
    for cache_file in files_to_delete:
        # @TODO store the cache_file with the running process so it gets deleted sooner -
        #   have to watch out in case more than one process use the same metadata cache
        remove_file(cache_file)


if __name__ == '__main__':

    # default_config_name = "default.config"
    # Runs the main function for each config specified in sys.argv
    run_command_line_configs(main, "Export", section="EXPORT")
