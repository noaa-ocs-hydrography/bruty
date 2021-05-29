import os
import sys
import time
import traceback
from datetime import datetime
import subprocess
import pathlib
import shutil
import logging
import io

import numpy

from nbs.bruty.raster_data import TiffStorage, LayersEnum
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
from nbs.bruty.utils import onerr, user_quit
from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import get_records, get_sorting_info, get_transform_metadata, connect_params_from_config
from nbs.scripts.convert_csar import convert_csar_python

_debug = False

LOGGER = get_logger('nbs.bruty.insert')
CONFIG_SECTION = 'insert'


def find_surveys_to_clean(db, sort_dict, names_list):
    """
    Parameters
    ----------
    db
    sort_dict
    names_list

    Returns
    -------

    """
    invalid_surveys = []
    unfinished_surveys = []
    out_of_sync_surveys = []
    id_to_path = {itm[1]: itm[2] for itm in names_list}
    for sid in db.started_ids.keys():
        # find any surveys in the WorldDatabase that are no longer in the Postgres list
        if sid not in sort_dict:
            db.db.LOGGER.info(f"found {sid} in the bruty data but not valid in postgres database - will remove")
            invalid_surveys.append(sid)
        else:
            # this call checks for a conversion of csar that is newer than the csar itself
            raw_pth = cached_conversion_name(id_to_path[sid])
            # if the filename changed or the original data (like csar) was updated but the converted file wasn't (like npz output from csar)
            # then mark for removal
            if raw_pth != db.started_ids[sid][0]:  # raw_path comes from metadata tables while started_ids are from the processed bruty sqlite
                db.db.LOGGER.info(f"found {sid} in the bruty data but path changed in database - will remove")
                db.db.LOGGER.info(f"\t{raw_pth} vs {db.started_ids[sid][0]}")
                # Track the partial inserts by listing that it is started - and when it crashes it will not be in included_ids
                invalid_surveys.append(sid)
        if sid not in db.included_ids:
            db.db.LOGGER.info(f"found {sid} in the bruty data but was not finished - consider removal (is another process adding it now?)")
            unfinished_surveys.append(sid)
    # if the file modification time on disk doesn't match the recorded time then mark for removal
    for sid, (path, tiles, orig_mtime, trans_id) in db.started_ids.items():
        path = pathlib.Path(path)
        try:
            mtime = path.stat().st_mtime
            if mtime != orig_mtime:
                out_of_sync_surveys.append(sid)
        except FileNotFoundError:
            out_of_sync_surveys.append(sid)

    return invalid_surveys, unfinished_surveys, out_of_sync_surveys


def cached_conversion_name(path):
    """  Find a cached conversion on disk that matches the supplied original path.
    For a CSAR file this would be a .bruty.npz or .csar.tif file.
    If the conversion file exists but is older than the supplied path then it is ignored and the original path will be returned.

    Parameters
    ----------
    path
        file path to check for pre-existing conversion

    Returns
    -------
    The original filename if no cached conversion exists, otherwise the path to the converted file

    """
    # convert csar names to exported data, 1 of 3 types
    if str(path).lower().endswith(".csar"):
        csar_path = pathlib.Path(path)
        for ext in (".csar.tif", ".bruty.npz"):
            export_path = csar_path.with_suffix(ext)
            if export_path.exists():
                # if the npz or tif file is newer than the geopackage or csar then use the exported file
                # - otherwise pretend it's not there and force a reconvert
                if (export_path.stat().st_mtime - csar_path.stat().st_mtime) >= 0:
                    path = str(export_path)
                    break

    return path


def process_nbs_database(world_db_path, table_names, database, username, password, hostname='OCS-VS-NBS01', port='5434',
                         for_navigation_flag=(True, True), extra_debug=False, override_epsg=NO_OVERRIDE):
    """
    Parameters
    ----------
    world_db_path
    table_names
    database
    username
    password
    hostname
    port
    for_navigation_flag
    extra_debug
    override_epsg

    Returns
    -------

    """

    all_fields, all_records = get_records(table_names, database, username, password, hostname, port, cache_dir=world_db_path)
    sorted_recs, names_list, sort_dict, comp = get_sorting_info(all_fields, all_records, for_navigation_flag)
    transform_metadata = get_transform_metadata(all_fields, all_records)
    if names_list:
        unconverted_csars = []
        files_not_found = []
        failed_to_insert = []
        db = WorldDatabase.open(world_db_path)
        trans_id = db.transaction_groups.add_oid_record(("INSERT", datetime.now(), os.getpid(), 0, 0))
        # fixme
        #   @todo Remove and recompute any surveys that no longer have valid Postgres records (and maybe the modified times too)
        invalid_surveys, unfinished_surveys, out_of_sync_surveys = find_surveys_to_clean(db, sort_dict, names_list)
        if unfinished_surveys:
            LOGGER.warning(f"Unfinished surveys detected, consider stopping and running clean_db unless there are"
                           " other active bruty combines\n{unfinished_surveys}")
        if out_of_sync_surveys:
            LOGGER.warning(f"Out of sync surveys detected, consider stopping and running clean_db\n{out_of_sync_surveys}")
        if invalid_surveys:
            msg = f"There are surveys without valid scores in the bruty database, shut down bruty instances and run clean_db\n{invalid_surveys}"
            LOGGER.error(msg)
            raise IndexError(msg)

        while names_list:
            num_names = len(names_list)
            for i in range(num_names-1, -1, -1):
                if user_quit():
                    names_list = []
                    db.transaction_groups.set_quit(trans_id)
                    break
                (filename, sid, path) = names_list[i]
                # if extra_debug and sid not in (721744, 720991, 720301):  # , 764261, 764263
                if extra_debug and i > 3:  # , 764261, 764263
                    names_list.pop(i)
                    continue
                LOGGER.debug(f'starting {sid} {path}')
                # # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
                msg = f"{datetime.now().isoformat()}  processing {num_names - i} of {num_names}"
                LOGGER.debug(msg)
                print(msg)  # we want this on screen and the debug file but not the info file (info is the default screen level)

                # convert csar names to exported data, 1 of 3 types
                orig_path = pathlib.Path(path)
                if path.lower().endswith(".csar"):
                    path = cached_conversion_name(path)
                    if path.lower().endswith(".csar"):
                        try:
                            with NameLock(path) as lck:
                                meta = transform_metadata[sid]
                                LOGGER.info(f"Trying to perform csar conversion for nbs_id={sid}, {path}")
                                path = convert_csar_python(path, meta)
                                if not path:
                                    LOGGER.error(f"no csar conversion file found for nbs_id={sid}, {path}")
                                    unconverted_csars.append(names_list.pop(i))
                                    continue
                        except LockNotAcquired:
                            LOGGER.info(f"{path} is locked and is probably being converted by another process")
                            continue
                if not os.path.exists(path):
                    LOGGER.info(f"{path} didn't exist")
                    files_not_found.append(names_list.pop(i))
                    continue

                # FIXME there is the possibility that we load metadata looking for SID=xx while it is being processed.
                #    Then it gets written to disk as we figure out what tiles to lock.
                #    We could check in the insert function again (once locks are obtained)
                #    to make sure survey=xx is not in the already processed list.
                sid_in_db = True
                if sid not in db.included_ids:
                    sid_in_db = False

                # @todo fixme - make a being processed list and check that the survey is not already being processed.
                #   This is an issue with the zip files overwriting a file being read
                #   but longer term in not starting to process/read the same file - especially for point data where we read the whole
                #   dataset to figure out the tiles to lock

                if not sid_in_db:
                    try:
                        lock = FileLock(path)  # this doesn't work with the file lock - just the multiprocessing locks
                        if lock.acquire():
                            try:
                                if path.endswith(".csv"):
                                    if os.path.exists(path):
                                        # points are in opposite convention as BAGs and exported CSAR tiffs, so reverse the z component
                                        db.insert_txt_survey(path, dformat=[('x', 'f8'), ('y', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')],
                                                             override_epsg=override_epsg, contrib_id=sid, compare_callback=comp, reverse_z=True,
                                                             transaction_id=trans_id)

                                elif path.endswith(".npy") or path.endswith(".npz"):
                                    db.insert_txt_survey(path, dformat=[('x', 'f8'), ('y', 'f8'), ('depth', 'f8'), ('uncertainty', 'f8')],
                                                         override_epsg=override_epsg, contrib_id=sid, compare_callback=comp,
                                                         transaction_id=trans_id)
                                else:
                                    db.insert_survey(path, override_epsg=override_epsg, contrib_id=sid, compare_callback=comp,
                                                     transaction_id=trans_id)
                            except BrutyError as e:
                                failed_to_insert.append((str(e), sid, path))
                            else:
                                LOGGER.info(f'inserted {path}')
                            names_list.pop(i)
                        else:
                            # print(f"{path} was locked - probably another process is working on it")
                            raise LockNotAcquired()
                    except LockNotAcquired:
                        LOGGER.debug(f'files in use for {sid} {path}')
                        LOGGER.debug('skipping to next survey')
                else:
                    LOGGER.debug(f"{sid} already in database")
                    names_list.pop(i)
        if unconverted_csars or files_not_found or failed_to_insert:
            LOGGER.debug("\n\n")  # extra space on screen
            LOGGER.warning("\nMissing, failed or skipped files:")
            for (filename, sid, path) in unconverted_csars:
                LOGGER.warning(f"skipped csar - no tif or npy found {sid}  {path}")
            for (filename, sid, path) in files_not_found:
                LOGGER.warning(f"file not found: {sid}  {path}")
            for (err_str, sid, path) in failed_to_insert:
                LOGGER.warning(f"failed to process: {sid}  {path}\n    {err_str}")
        else:
            LOGGER.info("\n\nAll files processed")
            db.transaction_groups.set_finished(trans_id)

    else:
        LOGGER.warning(f"No matching records found in tables {table_names}")
        LOGGER.warning(f"  for_navigation_flag used:{for_navigation_flag[0]}")
        if for_navigation_flag[0]:
            LOGGER.warning(f"  and for_navigation value must equal: {for_navigation_flag[1]}")


def main():
    """
    Returns
    -------

    """
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))

    warnings = ""

    for config_filename, config_file in iter_configs(use_configs):
        make_family_of_logs("nbs", config_filename.parent.joinpath("logs", config_filename.name + "_" + str(os.getpid())),
                            remove_other_file_loggers=False)
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Run  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
        if 'lock_server_port' in config:
            port = int(config['lock_server_port'])
            use_locks(port)
        db_path = pathlib.Path(config['combined_datapath'])
        try:  # see if there is an exising Bruty database
            db = WorldDatabase.open(db_path)
        except FileNotFoundError:  # create an empty bruty database
            with NameLock(db_path, "wb", EXCLUSIVE) as creation_lock:  # this will wait for the file to be available
                # on the slim chance that another process was making the database, let's check again since we now own the lock
                try:  # see if there is an exising Bruty database
                    WorldDatabase.open(db_path)
                except FileNotFoundError:  # really create an empty bruty database
                    try:
                        resx, resy = map(float, config['resolution'].split(','))
                    except:
                        resx = resy = float(config['resolution'])

                    if resx > 4:
                        zoom = 10
                    elif resx > 2:
                        zoom = 11
                    elif resx > 1:
                        zoom = 12
                    else:
                        zoom = 13
                    LOGGER.debug(f'zoom = {zoom}')
                    # NAD823 zone 19 = 26919.  WGS84 would be 32619
                    epsg = int(config['epsg'])
                    # use this to align the database to something else (like caris for testing)
                    offset_x = float(config['offset_x']) if 'offset_x' in config else 0
                    offset_y = float(config['offset_y']) if 'offset_y' in config else 0
                    db = WorldDatabase(
                        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y, zoom_level=zoom))
                    del db
        if False and _debug:
            hostname, port, username, password = None, None, None, None
            tablenames_raw, database = config['tablenames'], config['database']
            tablenames = parse_multiple_values(tablenames_raw)
        else:
            tablenames, database, hostname, port, username, password = connect_params_from_config(config)
        if 'DEBUG' in config:
            debug_config = (config['DEBUG'].lower().strip() == "true")
        else:
            debug_config = False
        override = db.db.epsg if (config['override'].lower().strip() == "true") else NO_OVERRIDE
        if 'use_for_navigation_flag' in config:
            if config['use_for_navigation_flag'].lower().strip() == "true":
                use_nav_flag = True
            elif config['use_for_navigation_flag'].lower().strip() == "false":
                use_nav_flag = False
            else:
                raise ValueError("use_for_navigation_flag must be true or false")
        else:
            use_nav_flag = True

        if 'for_navigation_equals' in config:
            if config['for_navigation_equals'].lower().strip() == "true":
                nav_flag_value = True
            elif config['for_navigation_equals'].lower().strip() == "false":
                nav_flag_value = False
            else:
                raise ValueError("for_navigation_equals must be true or false")
        else:
            nav_flag_value = True

        process_nbs_database(db_path, tablenames, database, username, password, hostname, port, for_navigation_flag=(use_nav_flag, nav_flag_value),
                             override_epsg=override, extra_debug=debug_config)

    # data_dir = pathlib.Path("c:\\data\\nbs\\test_data_output")  # avoid putting in the project directory as pycharm then tries to cache everything I think
    # def make_clean_dir(name):
    #     use_dir = data_dir.joinpath(name)
    #     if os.path.exists(use_dir):
    #         shutil.rmtree(use_dir, onerror=onerr)
    #     os.makedirs(use_dir)
    #     return use_dir
    # subdir = r"test_pbc_19_exact_multi_locks"
    # db_path = data_dir.joinpath(subdir)
    # make_clean_dir(subdir)

    # # create logger with 'spam_application'
    # logger = logging.getLogger('process_nbs')
    # logger.setLevel(logging.DEBUG)
    # # create file handler which logs even debug messages
    # fh = logging.FileHandler(db_path.joinpath('process_nbs.log'))
    # fh.setLevel(logging.DEBUG)
    # # create console handler with a higher log level
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)
    # # add the handlers to the logger
    # logger.addHandler(fh)
    # logger.addHandler(ch)
    #

    # db_path = make_clean_dir(r"test_pbc_19_db")  # reset the database


if __name__ == '__main__':

    # default_config_name = "default.config"

    # turn prints into logger messages
    save_print = False
    if save_print:
        orig_print = print
        def print(*args, **kywds):
            f = io.StringIO()
            ky = kywds.copy()
            ky['file'] = f
            orig_print(*args, **ky)  # build the string
            LOGGER.debug(f.getvalue()[:-1])  # strip the newline at the end
    main()


# "V:\NBS_Data\PBA_Alaska_UTM03N_Modeling"
# UTMN 03 through 07 folders exist
# /metadata/pba_alaska_utm03n_modeling
# same each utm has a table
# \\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBA_Alaska_UTM03N_Modeling
# v drive literal