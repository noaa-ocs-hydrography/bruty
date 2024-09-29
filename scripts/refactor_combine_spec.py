import datetime

from nbs import configs
from nbs.bruty import nbs_postgres
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, UTMTileBackendExactRes, NO_OVERRIDE

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

"""
CREATE OR REPLACE FUNCTION edit_combine_spec_view()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
   BEGIN
      IF TG_OP = 'UPDATE' THEN
       UPDATE combine_spec_bruty SET start_time=NEW.start_time, end_time=NEW.end_time, exit_code=NEW.exit_code, process_log=NEW.process_log, 
		  data_location=NEW.data_location, out_of_date=NEW.out_of_date, change_summary=NEW.change_summary WHERE b_id=OLD.b_id;
       RETURN NEW;
      ELSIF TG_OP = 'DELETE' THEN
       DELETE FROM combine_spec_bruty WHERE b_id=OLD.b_id;
       RETURN NULL;
      END IF;
      RETURN NEW;
    END;
$function$;
"""

"""
CREATE OR REPLACE TRIGGER edit_combine_spec_view_trigger
    INSTEAD OF DELETE OR UPDATE 
    ON public.combine_spec_view
    FOR EACH ROW
    EXECUTE FUNCTION public.edit_combine_spec_view();
"""

config = configs.read_config(r"C:\Git_Repos\Bruty\nbs\scripts\barry.gallagher\debug.config", log_files=False,
                             base_config_path = r"C:\Git_Repos\Bruty\nbs\scripts\base_configs")['DEFAULT']
conn_info = nbs_postgres.connect_params_from_config(config)
conn_info.database = "tile_specifications"
conn_info.tablenames = ['combine_spec_tiles']
con, cursor = nbs_postgres.connection_with_retries(conn_info)
if False:
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
if True:
    #  WHERE production_branch='PBC' and tile=1 and resolution=4
    cursor.execute(f"""select production_branch, locality, utm, hemisphere, datum, tile, resolution, datatype, not_for_nav, b_id, t_id from combine_spec_view""")
    tiles = cursor.fetchall()
    for (production_branch, locality, utm, hemisphere, datum, tile, resolution, datatype, not_for_nav, b_id, t_id) in tiles:
        world_db_path = fr"X:\bruty_databases\{production_branch}_{locality}_utm{utm}{hemisphere}_{datum}_Tile{tile}_res{int(resolution)}_{datatype}{'_not_for_navigation'*not_for_nav}"
        try:
            db = WorldDatabase.open(world_db_path)
        except FileNotFoundError:
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
                    cursor.execute(f"""UPDATE combine_spec_tiles SET combine_time=%s WHERE t_id=%s""", (datetime.datetime(1970, 1, 1), t_id))
                cursor.execute(f"""UPDATE combine_spec_view SET start_time=%s, end_time=%s, exit_code=%s WHERE b_id=%s""",
                               (start_time, last_code.ttime, last_code.code, b_id))
