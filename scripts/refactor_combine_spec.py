import datetime
import pathlib

from nbs import configs
from nbs.bruty import nbs_postgres
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE

config = configs.read_config(r"C:\Git_Repos\Bruty\nbs\scripts\barry.gallagher\debug.config", log_files=False,
                             base_config_path = r"C:\Git_Repos\Bruty\nbs\scripts\base_configs")['DEFAULT']
conn_info = nbs_postgres.connect_params_from_config(config)
conn_info.database = "tile_specifications"
conn_info.tablenames = ['combine_spec_tiles']
con, cursor = nbs_postgres.connection_with_retries(conn_info)

if False:  # split the old table into two
    raise Exception("This code is not needed anymore - and won't work as written")
    cursor.execute(f"""select t_id, resolution, closing_distance from combine_spec_tiles""")
    tiles = cursor.fetchall()
    done = {}
    for (tile_id, res, closing) in tiles:
        cursor.execute(f"""INSERT INTO combine_spec_resolutions (tile_id, closing_distance, resolution) VALUES (%s,%s,%s) RETURNING r_id""", (tile_id, closing, res))
        id_of_new_res = cursor.fetchone()[0]
        for not_for_nav in (True, False):
            for dtype in ("qualified", "unqualified", "enc", "sensitive", "gmrt"):
                cursor.execute(f"""INSERT INTO combine_spec_bruty (res_id, not_for_nav, datatype) VALUES (%s,%s,%s) RETURNING id""", (id_of_new_res, not_for_nav, dtype))
            id_of_new_row = cursor.fetchone()[0]
    con.commit()
    con.close()

def fill_spec_tables_from_disk():
    # fill in initial values
    #  WHERE production_branch='PBC' and tile=1 and resolution=4
    cursor.execute(f"""select production_branch, locality, utm, hemisphere, datum, tile, resolution, datatype, for_navigation, c_id, res_id, tile_id from view_individual_combines""")
    tiles = cursor.fetchall()
    for (production_branch, locality, utm, hemisphere, datum, tile, resolution, datatype, for_nav, c_id, r_id, t_id) in tiles:
        world_db_path = fr"X:\bruty_databases\{production_branch}_{locality}_utm{utm}{hemisphere}_{datum}_Tile{tile}_res{int(resolution)}_{datatype}{'_not_for_navigation'*(not for_nav)}"
        try:
            db = WorldDatabase.open(world_db_path)
        except FileNotFoundError:
            # datatypes that did not have surveys - mark as no data (-1 code)
            if pathlib.Path(world_db_path).exists():
                now = datetime.datetime.now()
                cursor.execute(f"""UPDATE view_individual_combines SET combine_start_time=%s, combine_end_time=%s, combine_exit_code=%s WHERE c_id=%s""",
                               (now, now, -1, c_id))
            # datatypes that had never run in any way - skip
            else:
                print(f"Could not open {world_db_path}")
        else:
            max_t = datetime.datetime(1, 1, 1)
            last_code = None
            for k, v in db.completion_codes.items():
                if v.ttime > max_t and v.ttype == "INSERT":
                    max_t = v.ttime
                    last_code = v
            if last_code:
                start_time = last_code.fingerprint.split('_')[1].replace('T', " ")
                if datatype.lower() == 'qualified':
                    cursor.execute(f"""SELECT export_time FROM xbox WHERE production_branch=%s AND tile=%s AND resolution=%s AND locality=%s AND hemisphere=%s AND datum=%s AND resolution=%s ORDER BY export_time DESC LIMIT 1""",
                                   (production_branch, tile, resolution, locality, hemisphere, datum, resolution))
                    rec = cursor.fetchone()
                    if rec is not None:
                        export_time = rec[0]
                        export_code = 0
                    else:
                        export_time = None
                        export_code = None
                    cursor.execute(f"""UPDATE spec_tiles SET combine_request_time=%s, export_request_time=%s WHERE t_id=%s""",
                                   (datetime.datetime(1970, 1, 1), export_time, t_id))
                    cursor.execute(f"""UPDATE spec_resolutions SET export_start_time=%s, export_end_time=%s, export_code=%s WHERE r_id=%s""",
                                   (export_time, export_time, export_code, r_id))
                cursor.execute(f"""UPDATE view_individual_combines SET combine_start_time=%s, combine_end_time=%s, combine_code=%s, combine_data_location=%s,
                    combine_warnings_log=NULL, combine_info_log=NULL, combine_tries=1, combine_running=False 
                    WHERE c_id=%s""",
                               (start_time, last_code.ttime, last_code.code, world_db_path, c_id))

if __name__ == "__main__":
    fill_spec_tables_from_disk()
    con.commit()
    con.close()
