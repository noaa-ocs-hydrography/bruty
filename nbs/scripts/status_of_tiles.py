""" this script checks if tiles would need to re-combined
It looks to see if there are new/changed surveys or ones that have been removed.
"""

import multiprocessing
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
import psutil

import fuse_dev
import nbs.bruty

from nbs.bruty import world_raster_database
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
# import locks from world_raster_database in case we are in debug mode and have turned locks off
from nbs.bruty.world_raster_database import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
# from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock
from nbs.configs import get_logger, run_command_line_configs
# , iter_configs, set_stream_logging, log_config, parse_multiple_values, make_family_of_logs
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config, connection_with_retries
from nbs.scripts.tile_specs import iterate_tiles_table, create_world_db
from nbs.scripts.combine import find_surveys_to_update, get_postgres_processing_info

interactive_debug = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

LOGGER = get_logger('nbs.bruty.status')

def main(config):
    """
    Returns
    -------

    """
    debug_config = config.getboolean('DEBUG', False)
    port = config.get('lock_server_port', None)
    use_locks(port)
    ignore_pids = psutil.pids()
    max_processes = config.getint('processes', 5)
    conn_info = connect_params_from_config(config)

    for tile_info in iterate_tiles_table(config):
        conn_info = connect_params_from_config(config)
        # FIXME make a copy of conn_info rather than switching .database back and forth
        invalid_surveys, unfinished_surveys, out_of_sync_surveys, metadata_mismatch, new_surveys = [], [], [], [], []
        LOGGER.info(f"Checking status of {tile_info.tile_name}")
        tile_info.summary = ""
        tile_info.out_of_date = False
        msg = f"Resolution={tile_info.resolution}m"
        tile_info.summary += msg + "\n"
        LOGGER.info(msg)

        for nav_flag_value in (True, False):
            for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                db = create_world_db(config['data_dir'], tile_info, dtype, nav_flag_value)
                conn_info.database = 'metadata'
                conn_info.tablenames = [tile_info.metadata_table_name(dtype)]
                sorted_recs, names_list, sort_dict, comp, transform_metadata = get_postgres_processing_info(None, conn_info,
                                                                                                            (True, nav_flag_value))
                inv, unf, sync, meta, news = find_surveys_to_update(db, sort_dict, names_list)
                # get names from bruty sqlite in case not in postgres anymore
                invalid_surveys.extend([pathlib.Path(db.included_ids[n].survey_path).name for n in inv])
                # get names from started_ids since they wouldn't have made it to bruty included_ids
                unfinished_surveys.extend([pathlib.Path(db.started_ids[n].survey_path).name for n in unf])
                # get names from bruty sqlite in case not in postgres
                out_of_sync_surveys.extend([pathlib.Path(db.included_ids[n].survey_path).name for n in sync])
                # mismatches should be in both postgres and sqlite
                metadata_mismatch.extend([names_list[sort_dict[n][-1]].from_filename for n in meta])
                # new surveys should only be in postgres
                new_surveys.extend([names_list[sort_dict[n][-1]].from_filename for n in news])
        if unfinished_surveys:
            msg = f"  Unfinished surveys detected\n{unfinished_surveys}"
            LOGGER.info(msg)
            tile_info.summary += msg + "\n"
        if out_of_sync_surveys:
            msg = f"  Out of sync surveys detected\n{out_of_sync_surveys}"
            LOGGER.info(msg)
            tile_info.summary += msg + "\n"
        if invalid_surveys:
            msg = f"  There are surveys without valid scores in the bruty database\n{invalid_surveys}"
            tile_info.summary += msg + "\n"
            LOGGER.info(msg)
        if metadata_mismatch:
            msg = f"  There are surveys whose sorting metadata has changed\n{metadata_mismatch}"
            tile_info.summary += msg + "\n"
            LOGGER.info(msg)
        if new_surveys:
            msg = f"  There are new surveys\n{new_surveys}"
            tile_info.summary += msg + "\n"
            LOGGER.info(msg)
        if not any([unfinished_surveys, out_of_sync_surveys, invalid_surveys, metadata_mismatch, new_surveys]):
            msg = f"  Resolution={tile_info.resolution}m is up to date"
            tile_info.summary += msg + "\n"
            LOGGER.info(msg)
        else:
            tile_info.out_of_date = True
        conn_info.database = "tile_specifications"
        conn, cursor = connection_with_retries(conn_info)
        tile_info.update_table_status(cursor)
        conn.commit()


if __name__ == '__main__':
    # this script checks if tiles would need to re-combined
    # It looks to see if there are new/changed surveys or oncs that have been removed.

    # Runs the main function for each config specified in sys.argv
    run_command_line_configs(main, "Status")
