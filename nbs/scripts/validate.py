import os
import sys
import pathlib
import logging
import io

import numpy

from nbs.bruty.world_raster_database import WorldDatabase
from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config

LOGGER = get_logger('nbs.bruty.validate')
CONFIG_SECTION = 'insert'



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
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Validating  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
        db_path = pathlib.Path(config['combined_datapath'])
        if os.path.exists(db_path.joinpath("wdb_metadata.json")) or \
                os.path.exists(db_path.joinpath("wdb_metadata.pickle")) or \
                os.path.exists(db_path.joinpath("wdb_metadata.class")):

            db = WorldDatabase.open(db_path)
            print("*** Checking for positioning errors...")
            position_errors = db.search_for_bad_positioning()
            if not position_errors:
                print("checks ok")
            print("*** Checking for unfinished reinserts...")
            reinserts_remain = len(db.reinserts.unfinished_records()) > 0
            if not reinserts_remain:
                print("checks ok")
            print("*** Checking for orphaned accumulation directories...")
            vr_orphaned_accum_db = db.search_for_accum_db()
            if not vr_orphaned_accum_db:
                print("checks ok")
            print("*** Checking if combine (insert) completed...")
            last_clean = None
            for rec in db.transaction_groups.values():
                if rec[0] == 'CLEAN':
                    if last_clean is None or rec[1] > last_clean[1]:
                        last_clean = rec
            last_inserts = []
            for rec in db.transaction_groups.values():
                if rec[0] == 'INSERT':
                    if last_clean is None or rec[1] > last_clean[1]:
                        last_inserts.append(rec)
            if not last_inserts:
                print("No INSERT operations were performed after the last CLEAN")
            else:
                unfinished = False
                one_finished = False
                for rec in last_inserts:
                    if not rec[3]:
                        unfinished = True
                    if rec[3] and not rec[4]:  # at least once the program needed to finish without the user quitting
                        one_finished = True
                if unfinished or not one_finished:
                    msg = "At least one combine (INSERT) didn't complete since the last cleanup operation or user quit in ALL cases"
                    print(msg)
                    for rec in last_inserts:
                        print(f'{rec[1].strftime("%Y/%m/%d - %H:%M:%S")}   completed:{bool(rec[3])}    user quit:{bool(rec[4])}')
                    print(msg)
                else:
                    print("checks ok")
            print("*** Checking DB consistency...")
            tile_missing, tile_extra, contributor_missing = db.validate()
            if not tile_missing and not tile_extra and not contributor_missing:
                print("checks ok")
            if contributor_missing:
                # create a reinsert list for the missing contributors
                affected_contributors = {}
                for tile, contribs in contributor_missing.items():
                    for contrib in contribs:
                        affected_contributors.setdefault(contrib, []).append(tile)
                db.add_reinserts(affected_contributors)
                print("Added reinsert instructions to metadata.sqlite")
            print("*** Finished checks")

            if reinserts_remain or tile_missing or tile_extra or contributor_missing:
                sys.exit(3)




if __name__ == '__main__':

    # default_config_name = "default.config"

    # turn prints into logger messages
    orig_print = print
    def print(*args, **kywds):
        f = io.StringIO()
        ky = kywds.copy()
        ky['file'] = f
        orig_print(*args, **ky)  # build the string
        LOGGER.info(f.getvalue()[:-1])  # strip the newline at the end
    main()
