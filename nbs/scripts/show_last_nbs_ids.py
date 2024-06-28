from nbs.bruty.nbs_postgres import get_tablenames, show_last_ids, connection_with_retries
from nbs.configs import run_command_line_configs
from nbs.bruty.nbs_postgres import ConnectionInfo, connect_params_from_config


def main(config):
    conn_info = connect_params_from_config(config)
    connection, cursor = connection_with_retries(conn_info)
    tablenames = get_tablenames(cursor)
    show_last_ids(cursor, tablenames)


if __name__ == "__main__":
    run_command_line_configs(main, "Export", section="EXPORT")
