from nbs.bruty import nbs_postgres
from nbs.configs import parse_multiple_values, iter_configs, run_command_line_configs
from nbs.bruty.nbs_postgres import ConnectionInfo, connect_params_from_config

def main(config):
    conn_info = connect_params_from_config(config)
    revise = False
    if not revise:
        nbs_postgres.make_all_serial_columns(conn_info, True)
    else:  # or to revise a table:
        for update_table in ('pbd_california_utm11n_mllw_qualified', 'pbd_california_utm11n_mllw_unqualified'):
            start = nbs_postgres.start_integer(update_table)
            nbs_postgres.create_identity_column(update_table, start, conn_info, force_restart=True)

if __name__ == "__main__":
    run_command_line_configs(main, "Export", section="EXPORT")
