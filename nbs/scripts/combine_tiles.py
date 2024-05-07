""" Run combine.py on each tile listed in the combine_spec table as defined by the build flag and the config parameters specified
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
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values, show_logger_handlers
# , iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config, connection_with_retries
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db, TileToProcess, TileProcess
from nbs.scripts.combine import process_nbs_database

interactive_debug = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

LOGGER = get_logger('nbs.bruty.insert')
CONFIG_SECTION = 'insert'

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")


def launch(world_db, conn_info, for_navigation_flag=(True, True), override_epsg=NO_OVERRIDE, extra_debug=False, new_console=True,
           lock=None, exclude=None, crop=False, log_path=None, env_path=r'', env_name='', minimized=False,
           fingerprint=""):
    """ for_navigation_flag = (use_nav_flag, nav_flag_value)

    fingerprint is being used to find the processes that are running.
    Because of launching cmd.exe there is no direct communication to the python process.
    Supplying fingerprint will make both the cmd console and the python process easier to find using psutil.
    """
    # either launch in a new console with subprocess.popen or use multiprocessing.Process.  Could also consider dask.
    world_db_path = world_db if isinstance(world_db, str) else world_db.db.data_path
    if new_console:
        # spawn a new console, activate a python environment and run the combine.py script with appropriate arguments
        restore_dir = os.getcwd()
        os.chdir(pathlib.Path(__file__).parent)
        # FIXME -- figure out and remove this hack
        bruty_code = nbs.bruty.__path__._path[0]
        bruty_code = str(pathlib.Path(bruty_code).parent.parent)
        nbs_code = fuse_dev.__path__._path[0]
        nbs_code = str(pathlib.Path(nbs_code).parent)
        if platform.system() == 'Windows':
            separator = ";"
            env_var_cmd = "set"
        else:
            separator = ":"
            env_var_cmd = "export"
        # looks like activate is overwriting the pythonpath, so specify it after the activate command
        cmds = [f'{env_var_cmd} TCL_LIBRARY=what', f'{env_var_cmd} TIX_LIBRARY=', f'{env_var_cmd} TK_LIBRARY=']
        if env_path:
            cmds.append(env_path + " " + env_name)  # activate the environment
        cmds.append(f'{env_var_cmd} PYTHONPATH={nbs_code}{separator}{bruty_code}')  # add the NBS and Bruty code to the python path
        combine_args = ['python combine.py']
        for table in conn_info.tablenames:
            combine_args.extend(["-t", table])
        for exclusion in exclude:
            combine_args.extend(['-x', exclusion])
        combine_args.extend(["-b", str(world_db_path)])
        if not for_navigation_flag[0]:  # not using the navigation flag
            combine_args.append("-i")
        else:  # using the navigation flag, so  see if it should be True (default) or False (needs arg)
            if not for_navigation_flag[1]:
                combine_args.append("-n")
        if crop:
            combine_args.append('-c')
        if override_epsg != NO_OVERRIDE:
            combine_args.extend(["-e", str(override_epsg)])
        if extra_debug:
            combine_args.append("--debug")
        if lock:
            combine_args.extend(["-l", lock])
        if log_path:
            combine_args.extend(["-g", log_path])
        if fingerprint:
            combine_args.extend(['-f', fingerprint])
        combine_args.extend(["-d", conn_info.database, "-r", str(conn_info.port), "-o", conn_info.hostname, "-u", conn_info.username,
                             "-p", '"'+conn_info.password+'"'])
        cmds.append(" ".join(combine_args))
        if platform.system() == 'Windows':
            cmds.append("exiter.bat 0")  # exiter.bat closes the console if there was no error code, keeps it open if there was an error
            args = 'cmd.exe /K ' + "&&".join(cmds)
            kwargs = popen_kwargs(activate=False, minimize=minimized)  # windows specific flags start flags
        else:
            cmds.extend(["exit", "0"])
            args = ['sh', '-c', ';'.join(cmds)]
            kwargs = {}

        proc = subprocess.Popen(args, **kwargs)
        os.chdir(restore_dir)
        ret = proc.pid
    else:
        raise Exception("same console multiprocessing is untested")
        # could actually do the same subprocess.Popen in the same console, or multiprocessing which would remove need for psutil, or use dask(?)
        multiprocessing.Process(target=process_nbs_database, args = (db, conn_info), kwargs={'for_navigation_flag':(use_nav_flag, nav_flag_value),
                             'override_epsg':override, 'extra_debug':debug_config})
    return ret


def remove_finished_processes(tile_processes, remaining_tiles, max_tries):
    for key in list(tile_processes.keys()):
        if not tile_processes[key].console_process.is_running():
            retry = False
            if tile_processes[key].succeeded():
                reason = "finished"
            # 3 is the combine code for bad/missing data
            elif tile_processes[key].finish_code() == 3:
                reason = "ended with data errors"
            elif remaining_tiles[key][1] + 1 >= max_tries:
                reason = "failed too many times"
            else:
                reason = "failed and will retry"
                retry = True
            LOGGER.info(f"Combine {reason} for {remaining_tiles[key][0]} {key.dtype} res={key.res}")
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
        descr_of_running_processes = '\n'.join([f'{prc.tile_info} {k.res} {k.dtype}' for k, prc in tile_processes.items()])
        print(f"Remaining tiles: {len(remaining_tiles)}\nCurrently running:{len(tile_processes)}\n{descr_of_running_processes}")


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
    try:
        while remaining_tiles:
            # careful not to iterate the dictionary and delete from it at the same time, so make a copy of the list of keys first
            for current_tile in list(remaining_tiles.keys()):
                do_keyboard_actions(remaining_tiles, tile_processes)
                if current_tile in tile_processes:  # already running this one.  Wait til finished or stopped to retry if needed.
                    continue
                try:
                    tile_info = remaining_tiles[current_tile][0]
                except KeyError:  # the tile was running and in the list but got removed while we were looping on the cached list of tiles
                    continue
                tile_info.res = current_tile.res  # set this each time for each resolution listed in the data object

                LOGGER.info(f"start combine for {tile_info} {current_tile.res}m {current_tile.dtype}, for_navigation:{current_tile.nav_flag}")
                if not current_tile.nav_flag and current_tile.dtype == ENC:
                    LOGGER.info(f"  Skipping ENC with for_navigation=False since all ENC data must be for navigation")
                    del remaining_tiles[current_tile]
                    continue
                # to make a full utm zone database, take the tile_info and set geometry and tile to None.
                # need to make a copy first
                # tile_info.geometry, tile_info.tile = None, None
                # full_db = create_world_db(config['data_dir'], tile_info, dtype, current_tile.nav_flag_value)
                db = create_world_db(config['data_dir'], tile_info, current_tile.dtype, current_tile.nav_flag)
                try:
                    lock = Lock(db.metadata_filename().with_suffix(".lock"), fail_when_locked=True, flags=LockFlags.EXCLUSIVE|LockFlags.NON_BLOCKING)
                    lock.acquire()
                    override = db.db.epsg if config.getboolean('override', False) else NO_OVERRIDE
                    conn_info.tablenames = [tile_info.metadata_table_name(current_tile.dtype)]
                    fingerprint = str(current_tile.hash_id) + "_" + datetime.now().isoformat()
                    if debug_launch:
                        use_locks(port)
                        ret = process_nbs_database(db, conn_info, for_navigation_flag=(use_nav_flag, current_tile.nav_flag),
                                                   extra_debug=debug_config, override_epsg=override, exclude=exclude, crop=(current_tile.dtype==ENC))
                        del remaining_tiles[current_tile]
                    else:
                        remove_finished_processes(tile_processes, remaining_tiles, max_tries)
                        while len(tile_processes) >= max_processes:  # wait for at least one process to finish
                            time.sleep(10)
                            do_keyboard_actions(remaining_tiles, tile_processes)
                            remove_finished_processes(tile_processes, remaining_tiles, max_tries)

                        root, cfg_name = os.path.split(config._source_filename)
                        pid = launch(db, conn_info, for_navigation_flag=(use_nav_flag, current_tile.nav_flag),
                                     override_epsg=override, extra_debug=debug_config, lock=port, exclude=exclude, crop=(current_tile.dtype==ENC),
                                     log_path=os.path.join(root, "logs", cfg_name), env_path=env_path, env_name=env_name, minimized=minimized,
                                     fingerprint=fingerprint)
                        running_process = ConsoleProcessTracker(["python", fingerprint, "combine.py"])
                        if running_process.console.last_pid != pid:
                            LOGGER.warning(f"Process ID mismatch {pid} did not match the found {running_process.console.last_pid}")
                        # print(running_process.console.is_running(), running_process.app.is_running(), running_process.app.last_pid)
                        tile_processes[current_tile] = TileProcess(running_process, tile_info, db, fingerprint, lock)
                    del lock  # unlocks if the lock wasn't stored in the tile_process
                except AlreadyLocked:
                    LOGGER.info(f"delay combine due to data lock for {tile_info} {current_tile.res}m {current_tile.dtype}, for_navigation:{current_tile.nav_flag}")
            # remove finished processes from the list or this becomes an infinite loop
            remove_finished_processes(tile_processes, remaining_tiles, max_tries)

            if len(remaining_tiles) > 0:  # not all files have finished, give time for processing or locks to finish
                time.sleep(10)
    except UserCancelled:
        pass
    except Exception as e:
        traceback.print_exc()
        msg = f"combine_tiles.py had an unhandled exception - see message above"
        print(msg)
        LOGGER.error(traceback.format_exc())
        LOGGER.error(msg)
        ret = 99

    old = """
    db_path = pathlib.Path(config['combined_datapath'])
    try:  # see if there is an exising Bruty database
        db = WorldDatabase.open(db_path)
    except FileNotFoundError:  # create an empty bruty database
        with NameLock(db_path, "wb", EXCLUSIVE) as _creation_lock:  # this will wait for the file to be available
            # on the slim chance that another process was making the database, let's check again since we now own the lock
            try:  # see if there is an exising Bruty database (in case someone else was making it at the same time)
                db = WorldDatabase.open(db_path)
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
                offset_x = config.getfloat('offset_x', 0)
                offset_y = config.getfloat('offset_y', 0)
                db = WorldDatabase(
                    UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path,
                                           offset_x=offset_x, offset_y=offset_y, zoom_level=zoom))

    conn_info = connect_params_from_config(config)
    override = db.db.epsg if config.getboolean('override', False) else NO_OVERRIDE
    use_nav_flag = config.getboolean('use_for_navigation_flag', True)
    nav_flag_value = config.getboolean('for_navigation_equals', True)

    process_nbs_database(db_path, conn_info, for_navigation_flag=(use_nav_flag, nav_flag_value),
                         override_epsg=override, extra_debug=debug_config)
    """
    # avoid putting in the project directory as pycharm then tries to cache everything I think
    # data_dir = pathlib.Path("c:\\data\\nbs\\test_data_output")
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
