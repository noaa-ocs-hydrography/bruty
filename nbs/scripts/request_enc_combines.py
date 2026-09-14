from nbs.bruty.nbs_postgres import pg_update
from tile_specs import SQLConnectionAndCursor, TileInfo, basic_conditions
from nbs.bruty.nbs_postgres import connect_params_from_config
from nbs.configs import get_logger, run_command_line_configs
import traceback
LOGGER = get_logger('nbs.bruty.insert')

def main(config):
    """
    Returns
    -------

    """
    try:
        conn_info_exports = connect_params_from_config(config)
        conn_info_exports.database = config.get('export_database', None)
        conn_info_exports.tablenames = [config.get('export_table', "")]
        sql_obj = SQLConnectionAndCursor(conn_info_exports)
        pg_update(sql_obj.cursor, TileInfo.SOURCE_TABLE, where=basic_conditions(config), request_enc=True)
        LOGGER.info("request_enc field set to true")
    except Exception as e:
        traceback.print_exc()
        msg = f"request_enc_combines.py had an unhandled exception - see message above"
        print(msg)
        LOGGER.error(traceback.format_exc())
        LOGGER.error(msg)


if __name__ == '__main__':
    # Runs the main function for each config specified in sys.argv
    run_command_line_configs(main, "ENCS", section="COMBINE", log_suffix="_combine_tiles", rotating=True)

