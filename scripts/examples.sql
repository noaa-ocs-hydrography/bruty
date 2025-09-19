-- Gemini prompt and result (edited)
-- Given two tables, spec_tiles and spec_resolutions, where spec_resolutions has a value tile_id that relates to the t_id column in spec_tiles. Write sql that for every record in spec_resolutions with resolution = 16 and has a related spec_tiles record with production_branch='PBD' and utm =16, 17 or 18 then create a new record in spec_resolutions with resolution=4 and the same closing_distance as the original record

INSERT INTO spec_resolutions (
    -- List all columns of spec_resolutions here
    tile_id,
    resolution,
    closing_distance,
	msg
    /* Add any other columns that exist in spec_resolutions
       and whose values should be copied from the original record
       For example: r_id, other_column1, other_column2
       If r_id is an auto-incrementing primary key, do NOT include it in this list.
       If r_id is NOT auto-incrementing, you'll need to generate a unique one.
       For simplicity, assuming r_id is SERIAL/IDENTITY and will be generated automatically. */
    --creation_date, -- Example of copying other fields
    --status         -- Example of copying other fields
)
SELECT
    sr.tile_id,
	--sr.resolution,
    4 AS resolution, -- New resolution value
    sr.closing_distance,
	'adding IGLD 4m from 16m scripted' as msg
    /* Select values for any other columns copied from the original record
       sr.creation_date, -- Example: Copy original creation_date */
    -- sr.status         -- Example: Copy original status
FROM
    spec_resolutions sr
JOIN
    spec_tiles st ON sr.tile_id = st.t_id
WHERE
    sr.resolution = 16
    AND st.production_branch = 'PBD'
    AND st.utm IN (16, 17, 18);

-- update the D: paths to UNC (windows style)
UPDATE spec_resolutions
SET export_warnings_log = REPLACE(export_warnings_log,
'D:\git_repos\bruty\nbs\scripts\OCS.SVC.NBS\logs',
'\\OCS-S-HyperV17\NBS_Store1\configs\bruty\windows\logs')


UPDATE spec_combines
SET combine_warnings_log = REPLACE(combine_warnings_log,
'D:\git_repos\bruty\nbs\scripts\OCS.SVC.NBS\logs',
'\\OCS-S-HyperV17\NBS_Store1\configs\bruty\windows\logs')
WHERE combine_warnings_log LIKE '%config_combine_3686_40848%'
-- DELETE FROM spec_resolutions WHERE r_id > 3677;
