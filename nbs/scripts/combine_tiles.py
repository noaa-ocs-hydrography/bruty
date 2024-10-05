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
from nbs.scripts.tile_specs import TileInfo
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values, show_logger_handlers, get_log_level
# , iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config, connection_with_retries
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db, TileToProcess, TileProcess
from nbs.scripts.combine import process_nbs_database, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS, perform_qc_checks
from nbs.debugging import get_call_logger, setup_call_logger, log_calls

interactive_debug = True
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

LOGGER = get_logger('nbs.bruty.insert')
CONFIG_SECTION = 'insert'

# FIXME
print("\nremove the hack setting the bruty and nbs directories into the python path\n")


def launch(world_db, view_pk_id, config_pth, tablenames, use_navigation_flag=True, override_epsg=NO_OVERRIDE, extra_debug=False, new_console=True,
           lock=None, exclude=None, crop=False, log_path=None, env_path=r'', env_name='', minimized=False,
           fingerprint="", delete_existing=False, log_level=logging.INFO):
    """

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
        if override_epsg != NO_OVERRIDE:
            combine_args.extend(["-e", str(override_epsg)])
        if extra_debug:
            combine_args.append("--debug")
        if delete_existing:
            combine_args.append("--delete")
        if lock:
            combine_args.extend(["-l", lock])
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
    else:
        raise Exception("same console multiprocessing is untested")
        # could actually do the same subprocess.Popen in the same console, or multiprocessing which would remove need for psutil, or use dask(?)
        multiprocessing.Process(target=process_nbs_database, args = (db, conn_info), kwargs={'use_navigation_flag':use_nav_flag,
                             'override_epsg':override, 'extra_debug':debug_config})
    return ret


def remove_finished_processes(tile_processes, tile_manager):
    for key in list(tile_processes.keys()):
        if not tile_processes[key].console_process.is_running():
            old_tile = tile_processes[key].tile_info
            try:
                retry = False
                if tile_processes[key].succeeded():
                    reason = "finished"
                # 3 is the combine code for bad/missing data
                elif tile_processes[key].finish_code() == DATA_ERRORS:
                    reason = "ended with data errors"
                # 4 is the code for LOCKED
                elif tile_processes[key].finish_code() == TILE_LOCKED:
                    reason = "LOCKED"
                    tile_processes[key].clear_finish_code()  # remove record so we don't just pile up a lot of needless records
                else:
                    reason = "failed and will retry if not over the maximum retry count"
            except (sqlite3.OperationalError, OSError) as e:
                # this is happening when the network drive loses connection, not sure if it is recoverable.
                # treat it like an unhandled exception in the subprocess - retry and increment the count
                # also wait a moment to let the network recover in case that helps
                msg = traceback.format_exc()
                LOGGER.warning(f"Exception accessing bruty db for {old_tile.tile_info}:\n{msg}")
                reason = "failed with a sqlite Operational error or OSError"
                retry = True
                time.sleep(5)
            LOGGER.info(f"Combine {reason} for {old_tile.tile_info}")
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

#
class TileManager:
    RUNNING_PRIORITY = -999

    def __init__(self, config, max_tries, allow_res=False):
        self.config = config
        self.max_tries = max_tries
        self.allow_res = allow_res
        self.user_dtypes, self.user_res = None, None
        self.read_user_settings(self.allow_res)
        self.remaining_tiles = {}

    def read_user_settings(self, allow_res=False):
        try:
            self.user_dtypes = [dt.strip() for dt in parse_multiple_values(self.config['dtypes'])]
        except KeyError:
            self.user_dtypes = None
        try:
            if allow_res:
                self.user_res = [float(dt.strip()) for dt in parse_multiple_values(self.config['res'])]
            else:
                self.user_res = None
        except KeyError:
            self.user_res = None

    def refresh_tiles_list(self, needs_combining=False):
        self.remaining_tiles = {}
        for tile_info in iterate_tiles_table(self.config, only_needs_to_combine=needs_combining, ignore_running=False, max_retries=self.max_tries):
            res = tile_info.resolution
            if self.user_res and res not in self.user_res:
                continue
            for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                if self.user_dtypes and dtype not in self.user_dtypes:
                    continue
                for nav_flag_value in (True, False):
                    self.remaining_tiles[tile_info.hash_id()] = tile_info

        # sort remaining_tiles by priority and then by balance of production branches
        # The remaining_tiles list is already sorted by tile number, so we can just sort by priority and then by production branch
        self.priorities = {}
        for hash, tile in self.remaining_tiles.items():
            use_priority = self._revised_priority(tile)
            try:
                self.priorities[use_priority][tile.pb][hash] = tile
            except KeyError:
                try:
                    self.priorities[use_priority][tile.pb] = {hash: tile}
                except KeyError:
                    self.priorities[use_priority] = {tile.pb: {hash: tile}}
    @staticmethod
    def _revised_priority(tile):
        use_priority = tile.priority if not tile.is_running else TileManager.RUNNING_PRIORITY
        if use_priority is None:
            use_priority = 0
        return use_priority

    def pick_next_tile(self, currently_running):
        # currently_running: dict(TileProcess)

        # pick the highest priority tile
        for priority in sorted(self.priorities.keys(), reverse=True):
            # balance the production branches
            # find the number of each production branch that is currently running
            # default the count to 0 for each production branch in case none are running
            pb_counts = {pb: 0 for pb in self.priorities[priority].keys()}
            for tile_process in currently_running.values():
                try:
                    pb_counts[tile_process.tile_info.pb] += 1
                except KeyError:  # something is running and no more of its production branch are in the list
                    pb_counts[tile_process.tile_info.pb] = 1

            for pb in sorted(pb_counts.keys(), key=lambda x: pb_counts[x]):
                for hash, tile in self.priorities[priority][pb].items():
                    if hash not in currently_running:
                        return tile
        return None
    def remove(self, tile):
        try:
            hash = tile.hash_id()
            del self.remaining_tiles[hash]
            del self.priorities[self._revised_priority(tile)][tile.pb][hash]
        except KeyError:
            pass

def main(config):
    """
    Returns
    -------

    """
    quitter = False
    debug_config = config.getboolean('DEBUG', False)
    minimized = config.getboolean('MINIMIZED', False)
    delete_existing = config.getboolean('delete_existing_if_cleanup_needed', False)
    is_service = config.getboolean('RUN_AS_SERVICE', False)
    env_path = config.get('environment_path')
    env_name = config.get('environment_name')
    port = config.get('lock_server_port', None)
    use_locks(port)
    ignore_pids = psutil.pids()
    max_processes = config.getint('processes', 5)
    max_tries = config.getint('max_retries', 3)
    conn_info = connect_params_from_config(config)
    log_level = get_log_level(config)
    exclude = parse_multiple_values(config.get('exclude_ids', ''))
    root, cfg_name = os.path.split(config._source_filename)
    log_path = os.path.join(root, "logs", cfg_name)
    if debug_config:
        show_logger_handlers(LOGGER)
    use_nav_flag = True  # config.getboolean('use_for_navigation_flag', True)

    debug_launch = interactive_debug and debug_config and max_processes < 2
    # While user doesn't quit and have a setting for if stop when finished user config (server runs forever while user ends when no more tiles to combine)
    #   while running processes >= max_processes: wait
    #   Read the combine_spec_view
    #   Order by priority then balance the production branches (or other logic like user)?
    #   Run the highest priority tile
    tile_manager = TileManager(config, max_tries, allow_res=debug_config)
    tile_processes = {}
    try:
        tile_manager.refresh_tiles_list(needs_combining=True)
        while is_service or tile_manager.remaining_tiles or tile_processes:  # run forever if a service, otherwise run until all tiles are combined
            # @TODO we need to change the log file occasionally to prevent it from getting too large
            # @TODO print("Move to unit test")
            # tile_manager.refresh_tiles_list(needs_combining=False)
            # for x in range(15):
            #     next_tile = tile_manager.pick_next_tile(tile_processes)
            #     print(next_tile)
            #     tile_processes[next_tile.hash_id()] = TileProcess(None, next_tile, None, 1, None)
            # print(tile_processes)
            # raise
            all_priorities = list(tile_manager.priorities.keys())
            # we will try to restart tiles that say they are running in case they crashed without filling the end_time field (meaning they are not actually running)
            if all_priorities == [tile_manager.RUNNING_PRIORITY] or (is_service and not all_priorities):
                if all_priorities == [tile_manager.RUNNING_PRIORITY]:
                    LOGGER.info("Waiting to give running tiles time to finish")
                else:
                    LOGGER.info("Waiting to before checking for more tiles to process")
                sleep(60)
            while tile_manager.remaining_tiles:
                next_tile = tile_manager.pick_next_tile(tile_processes)

                do_keyboard_actions(tile_manager.remaining_tiles, tile_processes)

                LOGGER.info(f"starting combine for {next_tile}" +
                            f"\n  {len(remaining_tiles)} remain including the {len(tile_processes) + 1} currently running")
                if not next_tile.for_nav and next_tile.datatype == ENC:
                    LOGGER.debug(f"  Skipping ENC with for_navigation=False since all ENC data must be for navigation")
                    tile_manager.remove(next_tile)
                    continue
                # to make a full utm zone database, take the tile_info and set geometry and tile to None.
                # need to make a copy first
                # tile_info.geometry, tile_info.tile = None, None
                # full_db = create_world_db(config['data_dir'], tile_info, dtype, current_tile.nav_flag_value)
                try:
                    db = create_world_db(config['data_dir'], tile_info, log_level=log_level)
                except (sqlite3.OperationalError, OSError) as e:
                    # this happened as a network issue - we will skip it for now and come back and give the network some time to recover
                    # but we will count it as a retry in case there is a corrupted file or something
                    msg = traceback.format_exc()
                    LOGGER.warning(f"Exception accessing bruty db for {tile_info}:\n{msg}")
                    time.sleep(5)
                else:
                    try:
                        # Lock all the database so we can write safely
                        # -- this is only effective on one OS
                        # so if linux is combining and Windows tries to export they won't see each others locks.
                        # We're adding a postgres lock which is cross platform inside the combine.py and tile_export.py
                        # to help this problem (but network errors may lose the postgres connection)
                        lock = Lock(db.metadata_filename().with_suffix(".lock"), fail_when_locked=True, flags=LockFlags.EXCLUSIVE|LockFlags.NON_BLOCKING)
                        lock.acquire()
                        override = db.db.epsg if config.getboolean('override', False) else NO_OVERRIDE
                        tablenames = [tile_info.metadata_table_name(tile_info.datatype)]
                        fingerprint = str(next_tile.hash_id()) + "_" + datetime.now().isoformat()
                        if debug_launch:
                            use_locks(port)
                            setup_call_logger(db.db.data_path)
                            # NOTICE -- this function will not write to the combine_spec_view table with the status codes etc.
                            ret = process_nbs_database(db, conn_info, next_tile, use_navigation_flag=use_nav_flag,
                                                       extra_debug=debug_config, override_epsg=override, exclude=exclude, crop=(next_tile.datatype==ENC),
                                                       delete_existing=delete_existing, log_level=log_level, view_pk_id=tile_info.view_id)
                            errors = perform_qc_checks(db.db.data_path, conn_info, (use_nav_flag, next_tile.for_nav), repair=True, check_last_insert=False)
                            tile_manager.remove(next_tile)
                        else:
                            remove_finished_processes(tile_processes, tile_manager)
                            get_refresh = False
                            while len(tile_processes) >= max_processes:  # wait for at least one process to finish
                                time.sleep(10)
                                do_keyboard_actions(remaining_tiles, tile_processes)
                                remove_finished_processes(tile_processes, tile_manager)
                                if is_service:
                                    get_refresh = True
                            if get_refresh:  # restart the while loop with an updated list of tiles
                                break
                            pid = launch(db, tile_info.view_id, config._source_filename, tablenames, use_navigation_flag=use_nav_flag,
                                         override_epsg=override, extra_debug=debug_config, lock=port, exclude=exclude, crop=(next_tile.datatype==ENC),
                                         log_path=log_path, env_path=env_path, env_name=env_name, minimized=minimized,
                                         fingerprint=fingerprint, delete_existing=delete_existing, log_level=log_level)
                            running_process = ConsoleProcessTracker(["python", fingerprint, "combine.py"])
                            if running_process.console.last_pid != pid:
                                LOGGER.warning(f"Process ID mismatch {pid} did not match the found {running_process.console.last_pid}")
                            else:
                                LOGGER.debug(f"Started PID {pid} for {next_tile}")

                            # print(running_process.console.is_running(), running_process.app.is_running(), running_process.app.last_pid)
                            tile_processes[next_tile.hash_id()] = TileProcess(running_process, tile_info, db, fingerprint, lock)
                        raise "Change all the things (iter_tiles) using hash_id which now includes datatype, res, not_for_nav"
                        del lock  # unlocks if the lock wasn't stored in the tile_process
                    except AlreadyLocked:
                        LOGGER.debug(f"delay combine due to data lock for {tile_info}")
            remove_finished_processes(tile_processes, tile_manager)
            tile_manager.refresh_tiles_list(needs_combining=True)

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

