""" This script checks if the last operation was a combine that succeeded.
The intent is to check if there were errors (network failures) that caused the combines to crash.

"""
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
import subprocess
import pathlib
import shutil
import logging
import io
import platform

import numpy
import psutil

import fuse_dev
import nbs.bruty

from nbs.bruty import world_raster_database
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.exceptions import UserCancelled, BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
# import locks from world_raster_database in case we are in debug mode and have turned locks off
from nbs.bruty.world_raster_database import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
# from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
from nbs.bruty.nbs_locks import Lock, AlreadyLocked, LockFlags
from nbs.bruty.utils import onerr, user_action, popen_kwargs, ConsoleProcessTracker, QUIT, HELP
from nbs.scripts.tile_specs import TileInfo
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values, show_logger_handlers, make_family_of_logs
# , iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config, connection_with_retries
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db, TileToProcess, TileProcess
from nbs.scripts.combine import process_nbs_database

interactive_debug = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

LOGGER = get_logger('nbs.bruty.check')
CONFIG_SECTION = 'check'

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")

def main(config):
    """
    Returns
    -------

    """
    quitter = False
    debug_config = config.getboolean('DEBUG', False)
    minimized = config.getboolean('MINIMIZED', False)
    env_path = config.get('environment_path')
    env_name = config.get('environment_name')
    port = config.get('lock_server_port', None)
    use_locks(port)
    ignore_pids = psutil.pids()
    max_processes = config.getint('processes', 5)
    max_tries = config.getint('processes', 2)
    conn_info = connect_params_from_config(config)
    exclude = parse_multiple_values(config.get('exclude_ids', ''))
    # only keep X decimals - to help compression and storage size
    decimals = config.getint('decimals', None)
    root, cfg_name = os.path.split(config._source_filename)
    log_path=os.path.join(root, "logs", cfg_name)
    # make a logger just for this process - the main log will have all the subprocesses in the file as well
    make_family_of_logs("nbs", log_path + "_manager_" + str(os.getpid()), remove_other_file_loggers=False)
    if debug_config:
        show_logger_handlers(LOGGER)
    remaining_tiles = {}
    use_nav_flag = True  # config.getboolean('use_for_navigation_flag', True)
    try:
        user_dtypes = [dt.strip() for dt in parse_multiple_values(config['dtypes'])]
    except KeyError:
        user_dtypes = None
    try:
        if debug_config:
            user_res = [float(dt.strip()) for dt in parse_multiple_values(config['res'])]
        else:
            user_res = None
    except KeyError:
        user_res = None
    for tile_info in iterate_tiles_table(config):
        for res in tile_info.resolutions:
            if user_res and res not in user_res:
                continue
            for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                if user_dtypes and dtype not in user_dtypes:
                    continue
                for nav_flag_value in (True, False):
                    remaining_tiles[TileToProcess(tile_info.hash_id(res), res, dtype, nav_flag_value)] = [tile_info, 0]
    debug_launch = interactive_debug and debug_config and max_processes < 2
    tile_processes = {}
    # careful not to iterate the dictionary and delete from it at the same time, so make a copy of the list of keys first
    for current_tile in list(remaining_tiles.keys()):
        try:
            tile_info = remaining_tiles[current_tile][0]
        except KeyError:  # the tile was running and in the list but got removed while we were looping on the cached list of tiles
            continue
        tile_info.res = current_tile.res  # set this each time for each resolution listed in the data object

        if not current_tile.nav_flag and current_tile.dtype == ENC:
            del remaining_tiles[current_tile]
            continue
        # to make a full utm zone database, take the tile_info and set geometry and tile to None.
        # need to make a copy first
        # tile_info.geometry, tile_info.tile = None, None
        # full_db = create_world_db(config['data_dir'], tile_info, dtype, current_tile.nav_flag_value)
        db = create_world_db(config['data_dir'], tile_info, current_tile.dtype, current_tile.nav_flag)
        max_t = datetime(1, 1, 1); last_action=None
        for k, v in db.transaction_groups.items():
            if v.ttime > max_t: 
                max_t = v.ttime
                last_action = v
        max_t = datetime(1, 1, 1); last_code=None
        for k, v in db.completion_codes.items():
            if v.ttime > max_t and v.ttype == "INSERT":
                max_t = v.ttime
                last_code = v
        if last_action is not None:
            if last_action.ttype == "INSERT" and last_action.finished == 1:
                if last_code is not None:
                    if last_code.code == 0:
                        status="      done"
                    else:
                        status = "UNFINISHED"
                else:
                    status = "UNSURE - Combine was last run before updates were made to Bruty"
            elif last_action.ttype == "CLEAN":
                status = "    UNSURE - Either there was no data or a crash occurred"
            else:
                status="UNFINISHED"
            LOGGER.info(f"{status} {db.db.data_path} Last operations: Action {last_action.ttype} at {last_action.ttime}, Code {last_code.ttime} {last_code.code}")


if __name__ == '__main__':

    # default_config_name = "default.config"

    # # turn prints into logger messages
    # save_print = False
    # if save_print:
    #     orig_print = print
    #     def print(*args, **kywds):
    #         f = io.StringIO()
    #         ky = kywds.copy()
    #         ky['file'] = f
    #         orig_print(*args, **ky)  # build the string
    #         LOGGER.debug(f.getvalue()[:-1])  # strip the newline at the end

    # Runs the main function for each config specified in sys.argv
    run_command_line_configs(main, "Insert", section="COMBINE")


# "V:\NBS_Data\PBA_Alaska_UTM03N_Modeling"
# UTMN 03 through 07 folders exist
# /metadata/pba_alaska_utm03n_modeling
# same each utm has a table
# \\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBA_Alaska_UTM03N_Modeling
# v drive literal
