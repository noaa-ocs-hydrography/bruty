import os
import sys
import pathlib
import logging
import io
import traceback

from nbs.bruty.world_raster_database import WorldDatabase
from nbs.configs import get_logger, run_command_line_configs, parse_multiple_values
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, connect_params_from_config
from nbs.scripts.tile_specs import iterate_tiles_table, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS

LOGGER = get_logger('nbs.bruty.validate')
CONFIG_SECTION = 'insert'


def main(config):
    """
    Returns
    -------

    """
    errors = {}
    debug_config = config.getboolean('DEBUG', False)
    repair = config.getboolean('REPAIR', False)

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
                            try:
                                db = WorldDatabase.open(db_path)
                            except FileNotFoundError:
                                LOGGER.info(f"{db_path} was not found")
                            else:
                                LOGGER.info(f"Validating {db_path}")
                                LOGGER.info("*** Checking for positioning errors...")
                                position_errors = db.search_for_bad_positioning()
                                if not position_errors:
                                    LOGGER.info("checks ok")
                                LOGGER.info("*** Checking for unfinished reinserts...")
                                reinserts_remain = len(db.reinserts.unfinished_records()) > 0
                                if not reinserts_remain:
                                    LOGGER.info("checks ok")
                                LOGGER.info("*** Checking for orphaned accumulation directories...")
                                vr_orphaned_accum_db = db.search_for_accum_db()
                                if not vr_orphaned_accum_db:
                                    LOGGER.info("checks ok")
                                LOGGER.info("*** Checking if combine (insert) completed...")
                                last_clean = None
                                for rec in db.transaction_groups.values():
                                    if rec.ttype == 'CLEAN':
                                        if last_clean is None or rec.ttime > last_clean.ttime:
                                            last_clean = rec
                                last_inserts = []
                                for rec in db.transaction_groups.values():
                                    if rec.ttype == 'INSERT':
                                        if last_clean is None or rec.ttime > last_clean.ttime:
                                            last_inserts.append(rec)
                                last_insert = None
                                for op in db.completion_codes.values():
                                    if op.ttype == 'INSERT' or op.ttype is None:
                                        if last_insert is None or op.ttime > last_insert.ttime:
                                            last_insert = op
                                last_insert_unfinished = last_insert is None or last_insert.code != 0
                                if not last_inserts:
                                    LOGGER.info("No INSERT operations were performed after the last CLEAN")
                                else:
                                    unfinished = False
                                    one_finished = False
                                    for rec in last_inserts:
                                        if not rec.finished:
                                            unfinished = True
                                        if rec.finished and not rec.user_quit:  # at least once the program needed to finish without the user quitting
                                            one_finished = True
                                    if unfinished or not one_finished:
                                        msg = "At least one combine (INSERT) didn't complete since the last cleanup operation or user quit in ALL cases"
                                        LOGGER.info(msg)
                                        for rec in last_inserts:
                                            LOGGER.info(f'{rec.ttime.strftime("%Y/%m/%d - %H:%M:%S")}   completed:{bool(rec.finished)}    user quit:{bool(rec.user_quit)}')
                                        LOGGER.info(msg)
                                    else:
                                        LOGGER.info("checks ok")
                                LOGGER.info("*** Checking DB consistency...")
                                tile_missing, tile_extra, contributor_missing = db.validate()
                                if not tile_missing and not tile_extra and not contributor_missing:
                                    LOGGER.info("consistency checks ok")
                                else:
                                    if contributor_missing:
                                        # create a reinsert list for the missing contributors
                                        affected_contributors = {}
                                        for tile, contribs in contributor_missing.items():
                                            for contrib in contribs:
                                                affected_contributors.setdefault(contrib, []).append(tile)
                                        db.add_reinserts(affected_contributors)
                                        LOGGER.info("Added reinsert instructions to metadata.sqlite")
                                    elif tile_missing or tile_extra:
                                        repairs = set(tile_missing).union(tile_extra)
                                        db.repair_subtiles(repairs)
                                LOGGER.info("*** Finished checks")

                                if reinserts_remain or tile_missing or tile_extra or contributor_missing or last_insert_unfinished:
                                    errors[db_path] = reinserts_remain, tile_missing, tile_extra, contributor_missing, last_insert_unfinished
        for db_path, error_list in errors.items():
            LOGGER.warning(f"{db_path}:\n  last insert finished without errors: {not error_list[4]}\n  reinserts_remain: {error_list[0]}\n  tile_missing: {error_list[1]}\n"
                           f"  tile_extra: {error_list[2]}\n  contributor_missing:{error_list[3]}")
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
    run_command_line_configs(main, "Validate")
