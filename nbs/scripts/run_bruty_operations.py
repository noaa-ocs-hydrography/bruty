""" Run combine.py on each tile listed in the combine_spec tables (views) as defined by the build flag and the config parameters specified
"""
import multiprocessing
from dataclasses import dataclass
import os
import sqlite3
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
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values, show_logger_handlers, get_log_level
# , iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import ENC, connect_params_from_config
from nbs.scripts.tile_specs import create_world_db, TileToProcess, TileProcess, TileManager, ResolutionTileInfo, CombineTileInfo, TileInfo
from nbs.scripts.combine import process_nbs_database, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS, perform_qc_checks
from nbs.bruty.tile_export import combine_and_export
from nbs.debugging import get_call_logger, setup_call_logger, log_calls

interactive_debug = True
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False
debug_launch = False

LOGGER = get_logger('nbs.bruty.insert')
CONFIG_SECTION = 'insert'
profiling = False
use_nav_flag = True

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")


def launch_export(config_path, tile_info, use_caches=False,
           env_path=r'D:\languages\miniconda3\Scripts\activate', env_name='NBS', profile_tile_id=(0, 0),
           decimals=None, minimized=False, fingerprint=""):

    """ for_navigation_flag = (use_nav_flag, nav_flag_value)
    """
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
    # if profiling:
    #     args.extend(["-m", "cProfile", "-o", str(pathlib.Path(config_path).parent) + f"\\timing{tile_info.tile}_{tile_info.resolution}.profile"])
    script_path = os.path.join(bruty_code, "tile_export.py")
    args.extend([script_path, "-c", config_path, "-k", str(tile_info.pk)])
    if decimals is not None:
        args.extend(["-d", str(decimals)])
    if fingerprint:
        args.extend(['-f', fingerprint])
    if use_caches:
        args.append("-u")

    # exiter closes the console if there was no error code, keeps it open if there was an error
    args.extend(["&", r"..\scripts\exiter.bat", f"{SUCCEEDED}", f"{TILE_LOCKED}"])  # note && only joins commands if they succeed, so just use one ampersand
    # because we are launching separate windows we can't use the subprocess.poll and returncode.
    # maybe we should switch to just logs and not leave a window on the screen for the user to see
    # that would make it easier to check the returncode
    proc = subprocess.Popen(args, **popen_kwargs(activate=False, minimize=minimized))
    # os.chdir(restore_dir)
    ret = proc.pid, script_path
    return ret


def launch_combine(world_db, view_pk_id, config_pth, tablenames, use_navigation_flag=True, override_epsg=False, extra_debug=False,
           exclude=None, crop=False, log_path=None, env_path=r'', env_name='', minimized=False,
           fingerprint="", delete_existing=False, log_level=logging.INFO):
    """

    fingerprint is being used to find the processes that are running.
    Because of launching cmd.exe there is no direct communication to the python process.
    Supplying fingerprint will make both the cmd console and the python process easier to find using psutil.
    """
    # either launch in a new console with subprocess.popen or use multiprocessing.Process.  Could also consider dask.
    world_db_path = world_db if isinstance(world_db, str) else world_db.db.data_path
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
    for table in tablenames:
        combine_args.extend(["-t", table])
    for exclusion in exclude:
        combine_args.extend(['-x', exclusion])
    combine_args.extend(["-k", str(view_pk_id)])
    combine_args.extend(["-b", str(world_db_path)])
    combine_args.extend(["-c", str(config_pth)])
    if not use_navigation_flag:  # not using the navigation flag
        combine_args.append("-i")
    combine_args.extend(['--log_level', str(log_level)])
    if crop:
        combine_args.append('-r')
    if override_epsg:
        combine_args.extend(["-e"])
    if extra_debug:
        combine_args.append("--debug")
    if delete_existing:
        combine_args.append("--delete")
    if log_path:
        combine_args.extend(["-g", log_path])
    if fingerprint:
        combine_args.extend(['-f', fingerprint])
    # combine_args.extend(["-d", conn_info.database, "-r", str(conn_info.port), "-o", conn_info.hostname, "-u", conn_info.username,
    #                      "-p", '"'+conn_info.password+'"'])
    cmds.append(" ".join(combine_args))
    if platform.system() == 'Windows':
        cmds.append(f"exiter.bat {SUCCEEDED} {TILE_LOCKED}")  # exiter.bat closes the console if there was no error code, keeps it open if there was an error
        args = 'cmd.exe /K ' + "&".join(cmds)  # note && only joins commands if they succeed = "0", so just use one ampersand so we can use different return codes
        kwargs = popen_kwargs(activate=False, minimize=minimized)  # windows specific flags start flags
    else:
        cmds.extend(["exit", f"{SUCCEEDED}", f"{TILE_LOCKED}"])
        args = ['sh', '-c', ';'.join(cmds)]
        kwargs = {}

    proc = subprocess.Popen(args, **kwargs)
    os.chdir(restore_dir)
    ret = proc.pid
    return ret

# attribute(@feature, 'combine_start_time') is not NULL and ((attribute(@feature, 'combine_start_time') > attribute(@feature, 'combine_end_time')) or ((attribute(@feature, 'combine_start_time') is not NULL and attribute(@feature, 'combine_end_time') is NULL)))
def remove_finished_processes(tile_processes, tile_manager):
    for key in list(tile_processes.keys()):
        if not tile_processes[key].console_process.is_running():
            old_tile = tile_processes[key].tile_info
            LOGGER.info(f"Combine for {old_tile} exited")
            try:
                tile_manager.remove(old_tile)  # remove the tile from the list of tiles to process in the future
            except KeyError:
                pass  # the records will disappear from the manager when refreshed from postgres and the code sees it was running
            del tile_processes[key]  # remove the instance from our list of active processes


def do_keyboard_actions(remaining_tiles, tile_processes):
    action = user_action()
    if action == QUIT:
        raise UserCancelled("User pressed keyboard quit")
    elif action == HELP:
        descr_of_running_processes = '\n'.join([f'{prc.tile_info}' for k, prc in tile_processes.items()])
        print(f"Remaining tiles: {len(remaining_tiles)}\nCurrently running:{len(tile_processes)}\n{descr_of_running_processes}")

@dataclass
class TileRuns:
    info: TileInfo
    count: int

def export_tile(tile_info, config, sql_info):
    decimals = config.getint('decimals', None)
    use_cached_meta = config.getboolean('USE_CACHED_METADATA', False)
    env_path = config.get('environment_path')
    env_name = config.get('environment_name')
    minimized = config.getboolean('MINIMIZED', False)
    # log_level = get_log_level(config)

    return_process = None

    if interactive_debug and debug_launch:
        combine_and_export(config, tile_info, decimals, use_cached_meta)
        return_process = None
    else:
        LOGGER.info(f"exporting {tile_info.full_name}")
        fingerprint = str(tile_info.hash_id) + "_" + datetime.datetime.now().isoformat()

        pid, script_path = launch_export(config._source_filename, tile_info, use_caches=use_cached_meta,
                                         env_path=env_path, env_name=env_name,
                                         decimals=decimals, minimized=minimized, fingerprint=fingerprint)
        running_process = ConsoleProcessTracker(["python", fingerprint, script_path])
        if running_process.console.last_pid != pid:
            LOGGER.warning(f"Process ID mismatch {pid} did not match the found {running_process.console.last_pid}")
        # print(running_process.console.is_running(), running_process.app.is_running(), running_process.app.last_pid)
        return_process = TileProcess(running_process, tile_info, fingerprint)

    return return_process


def combine_tile(tile_info, config, conn_info, debug_config=False, log_path=""):
    env_path = config.get('environment_path')
    env_name = config.get('environment_name')
    override = config.getboolean('override', False)
    log_level = get_log_level(config)
    exclude = parse_multiple_values(config.get('exclude_ids', ''))
    delete_existing = config.getboolean('delete_existing_if_cleanup_needed', False)
    minimized = config.getboolean('MINIMIZED', False)

    # to make a full utm zone database, take the tile_info and set geometry and tile to None.
    # need to make a copy first
    # tile_info.geometry, tile_info.tile = None, None
    # full_db = create_world_db(config['data_dir'], tile_info, dtype, current_tile.nav_flag_value)

    # We're adding a postgres lock which is cross platform inside the combine.py and tile_export.py
    # to help this problem (but network errors may lose the postgres connection)
    root_path = pathlib.Path(config['data_dir'])
    db_path = pathlib.Path(config['data_dir']).joinpath(tile_info.bruty_db_name())
    conn_info.tablenames = [tile_info.metadata_table_name()]
    fingerprint = str(tile_info.hash_id()) + "_" + datetime.now().isoformat()
    return_process = None
    if debug_launch:
        setup_call_logger(db_path)
        # NOTICE -- this function will not write to the combine_spec_view table with the status codes etc.
        ret = process_nbs_database(root_path, conn_info, tile_info, use_navigation_flag=use_nav_flag,
                                   extra_debug=debug_config, override_epsg=override, exclude=exclude, crop=(tile_info.datatype == ENC),
                                   delete_existing=delete_existing, log_level=log_level)
        errors = perform_qc_checks(db_path, conn_info, (use_nav_flag, tile_info.for_nav), repair=True, check_last_insert=False)
    else:
        pid = launch_combine(root_path, tile_info.pk, config._source_filename, conn_info.tablenames, use_navigation_flag=use_nav_flag,
                             override_epsg=override, extra_debug=debug_config, exclude=exclude, crop=(tile_info.datatype == ENC),
                             log_path=log_path, env_path=env_path, env_name=env_name, minimized=minimized,
                             fingerprint=fingerprint, delete_existing=delete_existing, log_level=log_level)
        running_process = ConsoleProcessTracker(["python", fingerprint, "combine.py"])
        if running_process.console.last_pid != pid:
            LOGGER.warning(f"Process ID mismatch {pid} did not match the found {running_process.console.last_pid}")
        else:
            LOGGER.debug(f"Started PID {pid} for {tile_info}")

        # print(running_process.console.is_running(), running_process.app.is_running(), running_process.app.last_pid)
        return_process = TileProcess(running_process, tile_info, fingerprint)
    return return_process

def reached_max_load(tile_processes, max_processes):
    # #TODO This could also look at free memory and processors to determine if we should start a new process
    #   but currently we are just looking at the number of processes
    cnt = 0
    for process in tile_processes.values():
        if isinstance(process.tile_info, CombineTileInfo):
            cnt+=1
        elif isinstance(process.tile_info, ResolutionTileInfo):
            cnt+=2
    return cnt >= max_processes

def main(config):
    """
    Returns
    -------

    """

    quitter = False
    debug_config = config.getboolean('DEBUG', False)
    is_service = config.getboolean('RUN_AS_SERVICE', False)
    # port = config.get('lock_server_port', None)
    use_locks(None)  # @ TODO change this to only be using Postgres locks - either row locks on a table or advisory locks
    ignore_pids = psutil.pids()
    max_processes = config.getint('processes', 5)
    max_tries = config.getint('max_retries', 3)
    conn_info = connect_params_from_config(config)
    root, cfg_name = os.path.split(config._source_filename)
    log_path = os.path.join(root, "logs", cfg_name)
    if debug_config:
        show_logger_handlers(LOGGER)
    # use_nav_flag = True  # config.getboolean('use_for_navigation_flag', True)

    global debug_launch
    debug_launch = interactive_debug and debug_config and max_processes < 2
    # While user doesn't quit and have a setting for if stop when finished user config (server runs forever while user ends when no more tiles to combine)
    #   while running processes >= max_processes: wait
    #   Read the combine_spec_view
    #   Order by priority then balance the production branches (or other logic like user)?
    #   Run the highest priority tile
    tile_manager = TileManager(config, max_tries, allow_res=debug_config)
    tile_processes = {}
    try:
        tile_manager.refresh_tiles_list(needs_combining=True, needs_exporting=True)
        while is_service or tile_manager.remaining_tiles or tile_processes:  # run forever if a service, otherwise run until all tiles are combined
            # @TODO we need to change the log file occasionally to prevent it from getting too large
            # @TODO print("Move to unit test")
            # tile_manager.refresh_tiles_list(needs_combining=False, needs_exporting=False)
            # for x in range(15):
            #     next_tile = tile_manager.pick_next_tile(tile_processes)
            #     print(next_tile)
            #     tile_processes[next_tile.hash_id()] = TileProcess(None, next_tile, 1)
            # print(tile_processes)
            # raise
            all_priorities = list(tile_manager.priorities.keys())
            # we will try to restart tiles that say they are running in case they crashed without filling the end_time field (meaning they are not actually running)
            if all_priorities == [tile_manager.RUNNING_PRIORITY] or (is_service and not all_priorities):
                if all_priorities == [tile_manager.RUNNING_PRIORITY]:
                    LOGGER.info("Waiting to give running tiles time to finish")
                else:
                    LOGGER.info("Waiting to before checking for more tiles to process")
                time.sleep(60)
            while tile_manager.remaining_tiles:
                tile_info = tile_manager.pick_next_tile(tile_processes)

                do_keyboard_actions(tile_manager.remaining_tiles, tile_processes)

                remove_finished_processes(tile_processes, tile_manager)
                get_refresh = False
                while reached_max_load(tile_processes, max_processes):  # wait for at least one process to finish
                    for n in range(5):  # make the keyboard respond more often
                        time.sleep(2)
                        do_keyboard_actions(tile_manager.remaining_tiles, tile_processes)
                    remove_finished_processes(tile_processes, tile_manager)
                    if is_service:
                        get_refresh = True
                if get_refresh:  # restart the while loop with an updated list of tiles
                    break

                operation = "Combine" if isinstance(tile_info, CombineTileInfo) else "Export"
                LOGGER.info(f"starting {operation} for {tile_info}" +
                            f"\n  {len(tile_manager.remaining_tiles)} remain including the {len(tile_processes)} currently running")
                tile_info.refresh_lock_status(tile_manager.sql_obj)
                if not tile_info.is_locked:
                    # CombineTileInfo is a subclass of ResolutionTileInfo so we have to check for the subclass first
                    if isinstance(tile_info, CombineTileInfo):
                        returned_process = combine_tile(tile_info, config, conn_info, debug_config=debug_config, log_path=log_path)
                    elif isinstance(tile_info, ResolutionTileInfo):
                        combine_tiles = tile_info.get_related_combine_info(tile_manager.sql_obj)
                        is_ready = True
                        for tile in combine_tiles:
                            if tile.combine.needs_processing() or tile.is_locked:
                                is_ready = False
                                break
                        if is_ready:
                            returned_process = export_tile(tile_info, config, tile_manager.sql_obj)
                    if returned_process is not None:
                        tile_processes[tile_info.hash_id()] = returned_process
                # If the tile was locked then we will still remove it and let the next refresh get the tile again
                tile_manager.remove(tile_info)
            remove_finished_processes(tile_processes, tile_manager)
            tile_manager.refresh_tiles_list(needs_combining=True, needs_exporting=True)

    except UserCancelled:
        pass
    except Exception as e:
        traceback.print_exc()
        msg = f"combine_tiles.py had an unhandled exception - see message above"
        print(msg)
        LOGGER.error(traceback.format_exc())
        LOGGER.error(msg)



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
    run_command_line_configs(main, "Insert", section="COMBINE", log_suffix="_combine_tiles")

