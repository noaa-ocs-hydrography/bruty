import pathlib
from nbs.bruty import nbs_postgres
from nbs.configs import parse_multiple_values, iter_configs
from nbs.bruty.nbs_postgres import get_nbs_records, start_integer, get_transform_metadata, ConnectionInfo, connect_params_from_config, connection_with_retries
from nbs.scripts import combine
from nbs.configs import get_logger
from nbs.bruty.world_raster_database import LockNotAcquired, AreaLock, FileLock, BaseLockException, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, Lock

LOGGER = get_logger('bruty.scripts.convert')

if __name__ == "__main__":
    for config_filename, config_file in iter_configs([r'C:\git_repos\bruty\nbs\scripts\barry.gallagher.la\debug.config']):
        config = config_file['DEFAULT']
        conn_info = connect_params_from_config(config)
        connection, cursor = connection_with_retries(conn_info)
        cursor.execute("""SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';""")
        tablenames = list(cursor.fetchall())
        for (tablename,) in tablenames:
            if True:
            # if ("PBB" in tablename.upper() and "16" in tablename):
            # if ("15" in tablename and "PBG" in tablename.upper()):
                try:
                    start = start_integer(tablename)
                except ValueError as e:
                    print(f'{tablename} was not recognized as an nbs_id capable table\n    {str(e)}')
                else:
                    fields, records = get_nbs_records(tablename, conn_info)
                    transform_metadata = get_transform_metadata([fields], [records])
                    for cnt, record in enumerate(records):
                        fname = record['manual_to_filename']
                        sid = record['nbs_id']
                        # convert csar names to exported data, 1 of 3 types
                        if fname:
                            path = fname  # pathlib.Path(fname)
                            if path.lower().endswith(".csar"):
                                try:
                                    path = combine.get_converted_csar(path, transform_metadata, sid)
                                    if not path:
                                        LOGGER.error(f"no csar conversion file found for nbs_id={sid}, {path}")
                                        # unconverted_csars.append(names_list.pop(i))
                                        continue
                                except (LockNotAcquired, BaseLockException):
                                    LOGGER.info(f"{path} is locked and is probably being converted by another process")
                                    continue
            else:
                LOGGER.info(f"skipping {tablename}")