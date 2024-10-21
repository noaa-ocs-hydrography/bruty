DROP VIEW IF EXISTS view_individual_combines;
CREATE VIEW view_individual_combines AS
	-- QGIS likes a key as the first column unless you used "setKeyColumn" in api so add it first (as a different name to avoid conflict)
	-- Also I want to remove the buffered_geometry (makes QGIS really slow) so have to manually add the spec_resolutions columns
	SELECT B.c_id as view_id, TI.request_combine, TI.combine_request_time, B.combine_running,
		TI.request_export, TI.export_request_time,R.export_running,
		TI.production_branch, TI.utm, TI.hemisphere, TI.tile, TI.datum, TI.locality,
		R.resolution, R.closing_distance, B.datatype, B.for_navigation,
		TI.priority, B.combine_start_time, B.combine_end_time,
		B.combine_code, B.combine_tries, B.combine_data_location, B.combine_warnings_log, B.combine_info_log,
		R.export_start_time, R.export_end_time,
		R.export_code, R.export_tries, R.export_warnings_log, R.export_info_log, R.export_data_location,
		B.change_summary, B.summary_datetime, B.out_of_date,
		B.c_id, B.res_id, R.tile_id, TI.geometry
	FROM spec_combines B JOIN spec_resolutions R ON (B.res_id = R.r_id) JOIN spec_tiles TI ON (R.tile_id = TI.t_id);


CREATE OR REPLACE FUNCTION edit_combine_view()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
	IF TG_OP = 'UPDATE' THEN
		UPDATE spec_combines SET combine_start_time=NEW.combine_start_time, combine_end_time=NEW.combine_end_time, combine_code=NEW.combine_code,
			combine_info_log=NEW.combine_info_log, combine_warnings_log=NEW.combine_warnings_log,
			combine_tries=NEW.combine_tries, combine_data_location=NEW.combine_data_location,
			out_of_date=NEW.out_of_date, change_summary=NEW.change_summary, summary_datetime=NEW.summary_datetime
			WHERE c_id=OLD.c_id;
		UPDATE spec_resolutions SET export_start_time=NEW.export_start_time, export_end_time=NEW.export_end_time, export_code=NEW.export_code,
			export_info_log=NEW.export_info_log, export_warnings_log=NEW.export_warnings_log,
			export_tries=NEW.export_tries, export_data_location=NEW.export_data_location
			WHERE r_id=OLD.res_id;
		IF NEW.request_combine = True THEN
			UPDATE spec_tiles SET request_combine=TRUE WHERE t_id = (select tile_id from spec_resolutions WHERE r_id=NEW.res_id);
		END IF;
		IF NEW.request_export = True THEN
			UPDATE spec_tiles SET request_export=TRUE WHERE t_id = (select tile_id from spec_resolutions WHERE r_id=NEW.res_id);
		END IF;
		RETURN NEW;
	ELSIF TG_OP = 'DELETE' THEN
		DELETE FROM spec_combines WHERE c_id=OLD.c_id;
		RETURN NULL;
	END IF;
	RETURN NEW;
END;
$function$;


CREATE OR REPLACE TRIGGER edit_combine_view_trigger
    INSTEAD OF INSERT OR DELETE OR UPDATE
    ON public.view_individual_combines
    FOR EACH ROW
    EXECUTE FUNCTION public.edit_combine_view();



-- Obsolete view -- just using individual combines now
DROP VIEW IF EXISTS view_individual_resolutions;
CREATE or REPLACE VIEW view_individual_resolutions as
	-- QGIS likes a key as the first column unless you used "setKeyColumn" in api so add it first (as a different name to avoid conflict)
	-- Also I want to remove the buffered_geometry (makes QGIS really slow) so have to manually add the spec_resolutions columns
	SELECT r_id as res_id, TI.request_export, TI.export_request_time, R.export_running,
		TI.production_branch, TI.utm, TI.hemisphere, TI.tile, TI.datum, TI.locality,
		R.resolution, R.closing_distance, R.export_start_time, R.export_end_time,
		R.export_code, R.export_tries, R.export_warnings_log, R.export_info_log,
		R.tile_id, TI.geometry  -- exit_code, tries, end_time, start_time, resolution, closing_distance, r_id, TI.*
	FROM spec_resolutions R JOIN spec_tiles TI ON (R.tile_id = TI.t_id);


CREATE OR REPLACE FUNCTION edit_resolution_view()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
BEGIN
	IF TG_OP = 'UPDATE' THEN
		UPDATE spec_resolutions SET export_start_time=NEW.export_start_time, export_end_time=NEW.export_end_time, export_code=NEW.export_code,
			export_info_log=NEW.export_info_log, export_warnings_log=NEW.export_warnings_log,
			export_tries=NEW.export_tries, export_data_location=NEW.export_data_location
			WHERE r_id=NEW.res_id;
		IF NEW.request_export = True THEN
			UPDATE spec_tiles SET request_export=TRUE WHERE t_id = OLD.tile_id;  -- (select tile_id from spec_resolutions WHERE r_id=OLD.tile_id);
		END IF;
		RETURN NEW;
	ELSIF TG_OP = 'DELETE' THEN
		DELETE FROM spec_resolutions WHERE r_id=OLD.res_id;
		RETURN NULL;
	END IF;
	RETURN NEW;
END;
$function$;


CREATE OR REPLACE TRIGGER edit_resolutions_view_trigger
    INSTEAD OF INSERT OR DELETE OR UPDATE
    ON public.view_individual_resolutions
    FOR EACH ROW
    EXECUTE FUNCTION public.edit_resolution_view();


-- Also obsolete
DROP VIEW IF EXISTS view_grouped_combines;
CREATE or REPLACE VIEW view_grouped_combines as
SELECT CONCAT('Tile ',tile,' ', resolution,'m ', datum, ' ', production_branch, '_', utm, hemisphere, ' ', locality) tile_name,
	request_combine, priority, bool_or(combine_code>0) has_errors, bool_and(combine_end_time>combine_request_time and combine_start_time>=combine_request_time) is_finished, bool_or(combine_start_time>=combine_request_time) has_started, bool_and(combine_start_time<combine_request_time) is_waiting,
	sum((combine_end_time>=combine_start_time and combine_start_time>=combine_request_time)::int) ran, sum((combine_end_time<combine_start_time or combine_start_time<combine_request_time)::int) remaining, sum((combine_end_time<combine_start_time)::int) running,
	MIN(combine_end_time) age, geometry, tile, utm, datum, resolution, production_branch, hemisphere, locality, res_id, tile_id
FROM view_individual_combines
GROUP BY tile, utm, datum, resolution, production_branch, hemisphere, locality, geometry, request_combine, res_id, tile_id, priority
ORDER BY production_branch, utm, tile desc;


CREATE OR REPLACE FUNCTION edit_view_grouped_combines()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $function$
   BEGIN
      IF TG_OP = 'UPDATE' THEN
       UPDATE spec_tiles SET request_combine=NEW.request_combine, request_export=NEW.request_export WHERE t_id=OLD.t_id;
       RETURN NEW;
      -- ELSIF TG_OP = 'DELETE' THEN
      --  DELETE FROM spec_combines WHERE c_id=OLD.c_id;
      --  DELETE FROM spec_resolutions WHERE r_id=OLD.r_id
      --  RETURN NULL;
      END IF;
      RETURN NEW;
    END;
$function$;

CREATE OR REPLACE TRIGGER edit_view_grouped_combines_trigger
    INSTEAD OF DELETE OR UPDATE OR INSERT
    ON public.view_grouped_combines
    FOR EACH ROW
    EXECUTE FUNCTION public.edit_view_grouped_combines();
