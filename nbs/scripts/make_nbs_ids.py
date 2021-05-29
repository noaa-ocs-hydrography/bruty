from nbs.bruty import nbs_postgres
from nbs.configs import parse_multiple_values, iter_configs

if __name__ == "__main__":
    revise = False
    if not revise:
        nbs_postgres.make_all_serial_columns(True)
    else:  # or to revise a table:
        for config_filename, config_file in iter_configs([r'C:\git_repos\bruty_dev_debugging\nbs\scripts\barry.gallagher.la\bg_dbg.config']):
            config = config_file['DEFAULT']
        _t, database, hostname, port, username, password = nbs_postgres.connect_params_from_config(config)
        for update_table in ('pbd_california_utm11n_mllw_qualified', 'pbd_california_utm11n_mllw_unqualified'):
            start = nbs_postgres.start_integer(update_table)
            nbs_postgres.create_identity_column(update_table, start, database, username, password, hostname, port, force_restart=True)
