import os
from datetime import datetime
import sys
import pathlib
import logging
import io
import traceback

from nbs.bruty import world_raster_database
from nbs.bruty.world_raster_database import WorldDatabase, AdvisoryLock, EXCLUSIVE, NON_BLOCKING, BaseLockException
from nbs.bruty.nbs_postgres import NOT_NAV
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config
from nbs.scripts.tile_specs import iterate_tiles_table, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS
from nbs.scripts.combine import get_postgres_processing_info, find_surveys_to_update, clean_nbs_database, perform_qc_checks

LOGGER = get_logger('nbs.bruty.validate')
CONFIG_SECTION = 'insert'


def main(config):
    """
    Returns
    -------

    """
    errors = {}
    debug_config = config.getboolean('DEBUG', False)
    repair = config.getboolean('repair', False)
    conn_info = connect_params_from_config(config)

    if debug_config:
        user_res = [float(dt.strip()) for dt in parse_multiple_values(config['res'])]
    else:
        user_res = None
    try:
        for tile_info in iterate_tiles_table(config):
            for res in tile_info.resolutions:
                if not user_res or res in user_res:
                    tile_info.res = res
                    for nav_flag_value in (True, False):
                        for dtype in (REVIEWED, PREREVIEW, ENC, GMRT, SENSITIVE):
                            db_path = pathlib.Path(config['data_dir']).joinpath(tile_info.bruty_db_name(dtype, nav_flag_value))
                            conn_info.tablenames = [tile_info.metadata_table_name(dtype)]
                            errors = perform_qc_checks(db_path, conn_info, (True, nav_flag_value), repair=repair)
                            # a missing database (like ENC_not_for_nav) returns None
                            if errors is not None and any(errors):
                                LOGGER.warning(f"{db_path}:\n  last insert finished without errors: {not errors[4]}\n"
                                               "  reinserts_remain: {errors[0]}\n  tile_missing: {errors[1]}\n"
                                               f"  tile_extra: {errors[2]}\n  contributor_missing:{errors[3]}")
        ret = SUCCEEDED
    except Exception as e:
        traceback.print_exc()
        msg = f"Validate.py had an unhandled exception - see message above"
        print(msg)
        c = LOGGER
        while c:
            for hdlr in c.handlers:
                print(hdlr)
            c = c.parent
        LOGGER.error(traceback.format_exc())
        LOGGER.error(msg)
        ret = UNHANDLED_EXCEPTION

    if errors:
        sys.exit(ret)




if __name__ == '__main__':
    # Runs the main function for each config specified in sys.argv
    run_command_line_configs(main, "Validate", section="VALIDATE")
