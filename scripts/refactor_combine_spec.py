from nbs import configs
from nbs.bruty import nbs_postgres

# The command to make the view for combine spec
""" 
CREATE or REPLACE VIEW combine_spec_view as
SELECT * FROM combine_spec_bruty B JOIN combine_spec_resolutions R ON (B.res_id = R.r_id) JOIN combine_spec_tiles T ON (R.tile_id = T.t_id);
"""

""" 
CREATE or REPLACE VIEW combine_spec_overview as
SELECT CONCAT('Tile ',tile,' ', resolution,'m ', datum, ' ', production_branch, '_', utm, hemisphere, ' ', locality) tile_name,
	bool_and(end_time>combine_time and start_time>=combine_time) is_finished, bool_or(start_time>=combine_time) has_started, bool_and(start_time<combine_time) is_waiting, 
	sum((end_time>=start_time and start_time>=combine_time)::int) ran, sum((end_time<start_time or start_time<combine_time)::int) remaining, sum((end_time<start_time)::int) running,
	MIN(end_time) age, geometry
FROM combine_spec_view
	-- WHERE t_id=3
GROUP BY tile, utm, datum, resolution, production_branch, hemisphere, locality, geometry
ORDER BY production_branch, utm, tile desc;
"""

config = configs.read_config(r"C:\Git_Repos\Bruty\nbs\scripts\barry.gallagher\debug.config", log_files=False,
                             base_config_path = r"C:\Git_Repos\Bruty\nbs\scripts\base_configs")['DEFAULT']
conn_info = nbs_postgres.connect_params_from_config(config)
conn_info.database = "tile_specifications"
conn_info.tablenames = ['combine_spec_tiles']
con, cursor = nbs_postgres.connection_with_retries(conn_info)
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
