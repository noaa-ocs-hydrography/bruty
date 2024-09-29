from nbs import configs
from nbs.bruty import nbs_postgres

config = configs.read_config(r"C:\Git_Repos\Bruty\nbs\scripts\barry.gallagher\debug.config", log_files=False,
                             base_config_path = r"C:\Git_Repos\Bruty\nbs\scripts\base_configs")['DEFAULT']
conn_info = nbs_postgres.connect_params_from_config(config)
conn_info.database = "tile_specifications"
conn_info.tablenames = ['combine_spec_tiles']
con, cursor = nbs_postgres.connection_with_retries(conn_info)
cursor.execute(f"""select id, resolution, closing_distance from combine_spec_tiles""")
tiles = cursor.fetchall()
for tile_id, resolutions, closings in tiles:
    for res, closing in zip(resolutions, closings):
        for for_nav in (True, False):
            for dtype in ("qualified", "unqualified", "gmrt", "enc", "sensitive"):
                cursor.execute(f"""INSERT INTO combine_spec_resolutions (tile_id, closing_distance, resolution, not_for_nav, datatype) VALUES (%s,%s,%s,%s,%s) RETURNING id""", (tile_id, closing, res, for_nav, dtype))
                id_of_new_row = cursor.fetchone()[0]
con.commit()
con.close()
