import os
import sys
import pathlib
import logging
import io
from datetime import datetime

from nbs.bruty.world_raster_database import WorldDatabase, use_locks
from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config
from nbs.bruty.nbs_postgres import connect_params_from_config
from nbs.bruty.nbs_postgres import get_sorting_records
from nbs.scripts.combine_surveys import find_surveys_to_clean

LOGGER = get_logger('nbs.bruty.clean')
CONFIG_SECTION = 'insert'


def clean_nbs_database(world_db_path, table_names, database, username, password, hostname='OCS-VS-NBS01', port='5434', for_navigation_flag=(True, True), subprocesses=5):
    try:
        db = WorldDatabase.open(world_db_path)
    except FileNotFoundError:
        print(world_db_path, "not found")
    else:
        sorted_recs, names_list, sort_dict, comp = get_sorting_records(table_names, database, username, password, hostname, port, for_navigation_flag)

        invalid_surveys, unfinished_surveys, out_of_sync_surveys = find_surveys_to_clean(db, sort_dict, names_list)
        if unfinished_surveys:
            LOGGER.info(f"Unfinished surveys detected\n{unfinished_surveys}")
        if out_of_sync_surveys:
            LOGGER.info(f"Out of sync surveys detected\n{out_of_sync_surveys}")
        if invalid_surveys:
            msg = f"There are surveys without valid scores in the bruty database\n{invalid_surveys}"
            LOGGER.info(msg)
        # surveys are only removed so this trans_id will not show up on any records in the metadata
        # also this record will serve as a check point for validation so make this record even if there is no action to take.
        trans_id = db.transaction_groups.add_oid_record(("CLEAN", datetime.now(), os.getpid(), 0, 0))
        if invalid_surveys or unfinished_surveys or out_of_sync_surveys or len(db.reinserts.unfinished_records()) > 0:
            db.clean(invalid_surveys, unfinished_surveys, out_of_sync_surveys, compare_callback=comp, transaction_id=trans_id, subprocesses=subprocesses)
        db.transaction_groups.set_finished(trans_id)


def main():
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))

    warnings = ""
    for config_filename, config_file in iter_configs(use_configs):
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Cleaning  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
        db_path = pathlib.Path(config['combined_datapath'])
        if os.path.exists(db_path.joinpath("wdb_metadata.json")) or \
                os.path.exists(db_path.joinpath("wdb_metadata.pickle")) or \
                os.path.exists(db_path.joinpath("wdb_metadata.class")):
            if 'lock_server_port' in config:
                port = int(config['lock_server_port'])
                use_locks(port)
            tablenames, database, hostname, port, username, password = connect_params_from_config(config)
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
            try:
                clean_nbs_database(db_path, tablenames, database, username, password, hostname, port, for_navigation_flag=(use_nav_flag, nav_flag_value))
            except RuntimeError:
                sys.exit(2)


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
