import os
import sys
import argparse
import time
import traceback
from datetime import datetime
import subprocess
import pathlib
import shutil
import logging
import io

import numpy
from osgeo import gdal, osr, ogr

from nbs.bruty import world_raster_database
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
# import locks from world_raster_database in case we are in debug mode and have turned locks off
from nbs.bruty.world_raster_database import LockNotAcquired, AreaLock, FileLock, BaseLockException, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, Lock
# from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
from nbs.bruty.utils import onerr, user_action, remove_file, QUIT, HELP
from nbs.configs import get_logger, run_command_line_configs, make_family_of_logs, show_logger_handlers
from nbs.bruty.nbs_postgres import get_records, get_sorting_info, get_transform_metadata, ConnectionInfo, connect_params_from_config
from nbs.scripts.convert_csar import convert_csar_python
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db
from nbs_utils.points_utils import to_npz

interactive_debug = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False


LOGGER = get_logger('nbs.bruty.insert')
CONFIG_SECTION = 'insert'
VERSION = (1, 0, 0)
__version__ = '.'.join(map(str, VERSION))

def find_surveys_to_update(db, sort_dict, names_list):
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
    metadata_mismatch = []
    new_surveys = []
    id_to_path = {itm.sid: itm.data_path for itm in names_list}
    # Find any surveys that are listed in the included table but not in the started table.
    # This may have happened if the clean function crashed (?)
    for sid in set(db.included_ids).difference(db.started_ids):
        invalid_surveys.append(sid)
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
            # raw_path comes from metadata tables while started_ids are from the processed bruty sqlite
            if raw_pth != db.started_ids[sid].survey_path:
                db.db.LOGGER.info(f"found {sid} in the bruty data but path changed in database - will remove")
                db.db.LOGGER.info(f"\t{raw_pth} vs {db.started_ids[sid].survey_path}")
                # Track the partial inserts by listing that it is started - and when it crashes it will not be in included_ids
                invalid_surveys.append(sid)
        if sid not in db.included_ids:
            db.db.LOGGER.info(f"found {sid} in the bruty data but was not finished - consider removal (is another process adding it now?)")
            unfinished_surveys.append(sid)

    # if the file modification time on disk doesn't match the recorded time then mark for removal
    for sid, rec in db.started_ids.items():
        path = pathlib.Path(rec.survey_path)
        try:
            mtime = path.stat().st_mtime
            if mtime != rec.mtime:
                out_of_sync_surveys.append(sid)
        except FileNotFoundError:
            out_of_sync_surveys.append(sid)

    # Find surveys not included yet and
    # Find surveys whose decay score or resolution changed even though the survey data (modified time checked above) hasn't changed
    # bad original metadata would cause this (should have been CATZOC B instead of CATZOC A for example)
    # FIXME -- when a decay process is done all the scores will move down but may not change sort order.
    #    Allow for a metadata change to occur as long as it doesn't change sort order with the other surveys included.
    for survey in names_list:
        try:
            rec = db.included_ids[survey.sid]
            if not rec.sorting_metadata == (survey.decay, survey.resolution):
                metadata_mismatch.append(survey.sid)
        except KeyError:
            new_surveys.append(survey.sid)  # survey not added yet
    return invalid_surveys, unfinished_surveys, out_of_sync_surveys, metadata_mismatch, new_surveys


def cached_conversion_name(path):
    """  Find a cached conversion on disk that matches the supplied original path.
    For a CSAR file this would be a .bruty.npz or .csar.tif file.
    For a GeoPackage file this would be a .bruty.npz file.
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
    if str(path).lower().endswith(".csar") or str(path).lower().endswith(".gpkg"):
        orig_path = pathlib.Path(path)
        for ext in (".csar.tif", ".bruty.npz"):
            export_path = orig_path.with_suffix(ext)
            if export_path.exists():
                # if the npz or tif file is newer than the geopackage or csar then use the exported file
                # - otherwise pretend it's not there and force a reconvert
                if (export_path.stat().st_mtime - orig_path.stat().st_mtime) >= 0:
                    path = str(export_path)
                    break

    return path


def get_postgres_processing_info(world_db_path, conn_info, for_navigation_flag=(True, True), exclude=None):
    all_fields, all_records = get_records(conn_info, cache_dir=world_db_path)
    sorted_recs, names_list, sort_dict, comp = get_sorting_info(all_fields, all_records, for_navigation_flag, exclude=exclude)
    transform_metadata = get_transform_metadata(all_fields, all_records)
    return sorted_recs, names_list, sort_dict, comp, transform_metadata


def clean_nbs_database(world_db_path, names_list, sort_dict, comp, subprocesses=5):
    try:
        db = WorldDatabase.open(world_db_path)
    except FileNotFoundError:
        print(world_db_path, "not found")
    else:

        invalid_surveys, unfinished_surveys, out_of_sync_surveys, metadata_mismatch, new_surveys = find_surveys_to_update(db, sort_dict, names_list)
        if unfinished_surveys:
            LOGGER.info(f"Unfinished surveys detected\n{unfinished_surveys}")
        if out_of_sync_surveys:
            LOGGER.info(f"Out of sync surveys detected\n{out_of_sync_surveys}")
        if invalid_surveys:
            msg = f"There are surveys without valid scores in the bruty database\n{invalid_surveys}"
            LOGGER.info(msg)
        if metadata_mismatch:
            msg = f"There are surveys whose sorting metadata has changed\n{metadata_mismatch}"
            LOGGER.info(msg)

        # surveys are only removed so this trans_id will not show up on any records in the metadata
        # also this record will serve as a check point for validation so make this record even if there is no action to take.
        data = db.transaction_groups.data_class()
        data.ttype = "CLEAN"
        data.ttime = datetime.now()
        data.process_id = os.getpid()
        data.finished = 0
        data.user_quit = 0
        trans_id = db.transaction_groups.add_oid_record(data)  # ("CLEAN", datetime.now(), os.getpid(), 0, 0))
        removals = set()
        removals.update(invalid_surveys)
        removals.update(unfinished_surveys)
        removals.update(out_of_sync_surveys)
        removals.update(metadata_mismatch)
        if removals or len(db.reinserts.unfinished_records()) > 0:
            db.clean(removals, compare_callback=comp, transaction_id=trans_id, subprocesses=subprocesses)
        db.transaction_groups.set_finished(trans_id)


def process_nbs_database(world_db, conn_info, for_navigation_flag=(True, True), extra_debug=False, override_epsg=NO_OVERRIDE, exclude=None, crop=False):
    """ Reads the NBS postgres metadata table to find all surveys in a region and insert them into a Bruty combined database.

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
    crop

    Returns
    -------

    """
    world_db_path = world_db if isinstance(world_db, (str, pathlib.Path)) else world_db.db.data_path
    sorted_recs, names_list, sort_dict, comp, transform_metadata = get_postgres_processing_info(world_db_path, conn_info, for_navigation_flag, exclude=exclude)
    clean_nbs_database(world_db_path, names_list, sort_dict, comp, subprocesses=1)
    if names_list:
        ret = process_nbs_records(world_db, names_list, sort_dict, comp, transform_metadata, extra_debug, override_epsg, crop=crop)
    else:
        LOGGER.warning(f"No matching records found in tables {conn_info.tablenames}")
        LOGGER.warning(f"  for_navigation_flag used:{for_navigation_flag[0]}")
        if for_navigation_flag[0]:
            LOGGER.warning(f"  and for_navigation value must equal: {for_navigation_flag[1]}")
        ret = 0  # this will leave a lot of windows open when a full production branch is done, so make it zero exit code
    return ret


def get_converted_csar(path, transform_metadata, sid):
    path = cached_conversion_name(path)
    if path.lower().endswith(".csar") and os.path.exists(path):
        # use the file lock on CSAR, we need this lock whether or not we are running a lock server and
        # there shouldn't be many of them or changing fast (which is the problem with Windows locks)
        lock_name = path + ".conversion.lock"
        with Lock(lock_name, 'w', fail_when_locked=True) as _lck:
            meta = transform_metadata[sid]
            LOGGER.info(f"Trying to perform csar conversion for nbs_id={sid}, {path}")
            path = convert_csar_python(path, meta)
        remove_file(lock_name)
    return path


def process_nbs_records(world_db, names_list, sort_dict, comp, transform_metadata, extra_debug=False, override_epsg=NO_OVERRIDE, crop=False):
    unconverted_csars = []
    files_not_found = []
    failed_to_insert = []
    db = WorldDatabase.open(world_db) if isinstance(world_db, str) else world_db
    try:
        if extra_debug:
            print("Logs for the opened Bruty DB")
            show_logger_handlers(db.db.LOGGER)
            print("\n\nGeneral loggers")
            show_logger_handlers(LOGGER)
        data = db.transaction_groups.data_class()
        data.ttype = "INSERT"
        data.ttime = datetime.now()
        data.process_id = os.getpid()
        data.finished = 0
        data.user_quit = 0
        trans_id = db.transaction_groups.add_oid_record(data)  # ("INSERT", datetime.now(), os.getpid(), 0, 0))
        # # fixme
        # #   @todo Remove and recompute any surveys that no longer have valid Postgres records (and maybe the modified times too)
        # invalid_surveys, unfinished_surveys, out_of_sync_surveys, metadata_mismatch, new_surveys = find_surveys_to_update(db, sort_dict, names_list)
        # if unfinished_surveys:
        #     LOGGER.warning(f"Unfinished surveys detected, consider stopping and running clean_db unless there are"
        #                    " other active bruty combines\n{unfinished_surveys}")
        # if out_of_sync_surveys:
        #     LOGGER.warning(f"Out of sync surveys detected, consider stopping and running clean_db\n{out_of_sync_surveys}")
        # if invalid_surveys:
        #     msg = f"There are surveys without valid scores in the bruty database, shut down bruty instances and run clean_db\n{invalid_surveys}"
        #     LOGGER.error(msg)
        #     raise IndexError(msg)

        while names_list:
            time.sleep(1)  # when waiting for one locked file this stops it from being a cpu hog
            num_names = len(names_list)
            for i in range(num_names-1, -1, -1):
                action = user_action()
                if action == QUIT:
                    names_list = []
                    db.transaction_groups.set_quit(trans_id)
                    break
                elif action == HELP:
                    print(f"Number of surveys remaining: {len(names_list)}")
                survey = names_list[i]
                sid = survey.sid
                path = survey.data_path
                sort_info = (survey.decay, survey.resolution)
                if extra_debug:
                    pth = path.upper()
                    # if not ('H12010' in pth or 'H06443' in pth or 'H12023' in pth):  # a VR, gdal raster and points survey (points is in not__for_nav)
                    # if extra_debug and sid not in (721744, 720991, 720301):  # , 764261, 764263
                    # if extra_debug and i > 3:  # , 764261, 764263
                    #     names_list.pop(i)
                    #     continue
                    print(path)
                LOGGER.debug(f'starting {sid} {path}')
                # # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
                msg = f"{datetime.now().isoformat()}  processing {num_names - i} of {num_names}"
                LOGGER.debug(msg)
                print(msg)  # we want this on screen and the debug file but not the info file (info is the default screen level)

                # convert csar names to exported data, 1 of 3 types
                _orig_path = pathlib.Path(path)
                if path.lower().endswith(".csar"):
                    try:
                        path = get_converted_csar(path, transform_metadata, sid)
                        if not path:
                            LOGGER.error(f"no csar conversion file found for nbs_id={sid}, {path}")
                            unconverted_csars.append(names_list.pop(i))
                            continue
                    except (LockNotAcquired, BaseLockException):
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
                                                             transaction_id=trans_id, sorting_metadata=sort_info)

                                elif path.endswith(".npy") or path.endswith(".npz"):
                                    db.insert_txt_survey(path, dformat=[('x', 'f8'), ('y', 'f8'), ('depth', 'f8'), ('uncertainty', 'f8')],
                                                         override_epsg=override_epsg, contrib_id=sid, compare_callback=comp,
                                                         transaction_id=trans_id, sorting_metadata=sort_info, crop=crop)
                                else:
                                    db.insert_survey(path, override_epsg=override_epsg, contrib_id=sid, compare_callback=comp,
                                                     transaction_id=trans_id, sorting_metadata=sort_info)
                            except BrutyError as e:
                                failed_to_insert.append((str(e), survey))
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
            for survey in unconverted_csars:
                LOGGER.warning(f"skipped csar - no tif or npy found {survey.sid}  {survey.data_path}")
            for survey in files_not_found:
                LOGGER.warning(f"file not found: {survey.sid}  {survey.data_path}")
            for (err_str, survey) in failed_to_insert:
                LOGGER.warning(f"failed to process: {survey.sid}  {survey.data_path}\n    {err_str}")
            ret = 3
        else:
            LOGGER.info("\n\nAll files processed")
            db.transaction_groups.set_finished(trans_id)
            ret = 0
        return ret
    except Exception as e:
        db.db.LOGGER.error(traceback.format_exc())
        raise e

def make_parser():
    parser = argparse.ArgumentParser(description='Combine a NBS postgres table(s) into a Bruty dataset')
    parser.add_argument("-?", "--show_help", action="store_true",
                        help="show this help message and exit")

    parser.add_argument("-d", "--database", type=str, metavar='database', default="metadata",
                        help="postgres database table holding the records to process")
    parser.add_argument("-r", "--port", type=str, metavar='port', default='5434',  # nargs="+"
                        help="postgres database connection port")
    parser.add_argument("-o", "--host", type=str, metavar='host', default='OCS-VS-NBS05',
                        help="postgres host")
    parser.add_argument("-u", "--user", type=str, metavar='user', default="",
                        help="username to connect with to database")
    parser.add_argument("-p", "--password", type=str, metavar='password', default="",
                        help="password to connect to postgres database")
    parser.add_argument("-t", "--table", action='append', type=str, dest="tables", metavar='table', default=[],
                        help="table to read from postgres database, can specify more than once")
    parser.add_argument("-x", "--exclude", action='append', type=int, dest="exclude", default=[],
                        help="nbs_ids to exclude from the combine process")
    parser.add_argument("-b", "--bruty_path", type=str, metavar='bruty_path', default="",
                        help="location to store Bruty data")
    parser.add_argument('--debug', action='store_true', dest='debug',
                        default=False, help="turn on debugging code")
    parser.add_argument("-e", "--override_epsg", type=int, metavar='override_epsg', default=NO_OVERRIDE,
                        help="override incoming data epsg with this value")
    parser.add_argument("-l", "--lock_server", type=int, metavar='lock_server', default=None,
                        help="override incoming data epsg with this value")
    parser.add_argument('-i', '--ignore_for_navigation', action='store_true', dest='ignore_for_nav',
                        default=False, help="ignore the for_navigation flag")
    parser.add_argument('-n', '--not_for_nav', action='store_true', dest='not_for_nav',
                        default=False, help="require the for_navigation flag to be False")
    parser.add_argument('-c', action='store_true', dest='crop',
                        default=False, help="crop data to tile extents to avoid error (for ENC data covering many zones)")
    parser.add_argument("-g", "--logger_path", type=str, metavar='logger_path', default="",
                        help="location to store logger messages")
    parser.add_argument("-f", "--fingerprint", type=str, metavar='fingerprint', default="",
                        help="fingerprint to store success/fail code with in sqlite db")


    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.show_help or not args.bruty_path or not args.tables:
        parser.print_help()
        ret = 1

    if args.bruty_path:
        if args.logger_path:
            make_family_of_logs("nbs", args.logger_path, remove_other_file_loggers=False)
            make_family_of_logs("nbs", args.logger_path + "_" + str(os.getpid()), remove_other_file_loggers=False)
        conn_info = ConnectionInfo(args.database, args.user, args.password, args.host, args.port, args.tables)
        use_locks(args.lock_server)
        try:
            print(f"Processing {args.bruty_path} for_navigation_flag={(not args.ignore_for_nav, not args.not_for_nav)}")
            ret = process_nbs_database(args.bruty_path, conn_info, for_navigation_flag=(not args.ignore_for_nav, not args.not_for_nav),
                                       extra_debug=args.debug, override_epsg=args.override_epsg, exclude=args.exclude, crop=args.crop)
        except Exception as e:
            traceback.print_exc()
            msg = f"{args.bruty_path} for_navigation_flag={(not args.ignore_for_nav, not args.not_for_nav)} had an unhandled exception - see message above"
            print(msg)
            c = LOGGER
            while c:
                for hdlr in c.handlers:
                    print(hdlr)
                c = c.parent
            LOGGER.error(traceback.format_exc())
            LOGGER.error(msg)
            ret = 99
        if args.fingerprint:
            try:
                db = WorldDatabase.open(args.bruty_path)
                d = db.completion_codes.data_class()
                d.ttime = datetime.now()
                d.code = ret
                d.fingerprint = args.fingerprint
                db.completion_codes[args.fingerprint] = d
            except:
                pass

    sys.exit(ret)