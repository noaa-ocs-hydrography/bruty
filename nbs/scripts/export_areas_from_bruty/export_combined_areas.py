import gc
import os
import sys
import time
from datetime import datetime
import pathlib
import logging
import io
from functools import partial

import rasterio
from osgeo import gdal, osr, ogr

from nbs.bruty.world_raster_database import WorldDatabase, LockNotAcquired, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING
from nbs.bruty.utils import get_crs_transformer, compute_delta_coord, transform_rect
from nbs.configs import get_logger, iter_configs, set_stream_logging, log_config, parse_multiple_values
from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, connect_params_from_config, make_contributor_csv

LOGGER = get_logger('nbs.bruty.export')
CONFIG_SECTION = 'export'

_debug = False


def export(db_paths, export_dir, export_areas_shape_filename, name_from_field, output_res, clip_to_shape_filename,
           table_names, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    epsgs = []
    # FIXME this is only working if the coordinate systems of the databases match -- x1,x2,y1,y2 are all computed from the first db crs
    for db_path in db_paths:
        db = WorldDatabase.open(db_path)
        epsgs.append(db.db.tile_scheme.epsg)
    epsgs_match = [epsg == epsgs[0] for epsg in epsgs]
    if not all(epsgs_match):
        raise ValueError(f"EPSGs must all match from world databases, found{epsgs}")
    # @todo - make a class to encapsulate NBS data tables rather than passing as lists.  Also to contain the logic about 'script vs manual' values etc
    all_fields = []
    all_records = []
    for table_name in table_names:
        fields, records = get_nbs_records(table_name, database, username, password, hostname=hostname, port=port)
        all_records.append(records)
        all_fields.append(fields)
    sorted_recs, names_list, sort_dict = id_to_scoring(all_fields, all_records)

    comp = partial(nbs_survey_sort, sort_dict)
    export_dir = pathlib.Path(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    # export_dir = make_clean_dir("pbc19_exports")
    # all_tiles_shape_fnames = ((r"G:\Data\NBS\Support_Files\MCD_Bands\Band3\Band3_V6.shp", (16, 16)),
    #                           (r"G:\Data\NBS\Support_Files\MCD_Bands\Band4\Band4_V6.shp", (8, 8)),
    #                           (r"G:\Data\NBS\Support_Files\MCD_Bands\Band5\Band5_V6.shp", (4, 4)),
    #                           )
    # export_area_shp_fname = r"G:\Data\NBS\Support_Files\MCD_Bands\PBC_Review\PBC_UTM19N_Review.shp"
    if clip_to_shape_filename is not None:
        ds_clip = gdal.OpenEx(clip_to_shape_filename)  # this is the product branch area rough extents, PBC, PBD, PBG etc
        clip_lyr = ds_clip.GetLayer(0)
        # pb_srs = clip_lyr.GetSpatialRef()
        # pb_export_epsg = rasterio.crs.CRS.from_string(pb_srs.ExportToWkt()).to_epsg()
        clip_minx, clip_maxx, clip_miny, clip_maxy = clip_lyr.GetExtent()  # this is the product branch area rough extents, PBC, PBD, PBG etc
    else:
        clip_minx, clip_maxx, clip_miny, clip_maxy = 0, 0, 0, 0

    # for export_areas_shape_filename, output_res in all_tiles_shape_fnames:
    ds = gdal.OpenEx(export_areas_shape_filename)
    # ds.GetLayerCount()
    lyr = ds.GetLayer(0)
    srs = lyr.GetSpatialRef()
    export_epsg = rasterio.crs.CRS.from_string(srs.ExportToWkt()).to_epsg()
    lyr.GetFeatureCount()
    lyrdef = lyr.GetLayerDefn()
    for i in range(lyrdef.GetFieldCount()):
        flddef = lyrdef.GetFieldDefn(i)
        if flddef.name == name_from_field:
            name_field = i
            break
    db = WorldDatabase.open(db_paths[0])
    crs_transform = get_crs_transformer(export_epsg, db.db.tile_scheme.epsg)
    inv_crs_transform = get_crs_transformer(db.db.tile_scheme.epsg, export_epsg)
    for feat in lyr:
        geom = feat.GetGeometryRef()
        # geom.GetGeometryCount()
        minx, maxx, miny, maxy = geom.GetEnvelope()  # (-164.7, -164.39999999999998, 67.725, 67.8)
        # output in WGS84
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        # crop to the area around the output area (so we aren't evaluating everyone on Earth)
        if clip_to_shape_filename is None or (clip_minx < cx < clip_maxx and clip_miny < cy < clip_maxy):
            cell_name = str(feat.GetField(name_field))
            if _debug:
                pass
                # if cell_name not in ('US4NY1CY', 'US5PVDBC'):
                #     continue

            print(f'processing area {name_from_field} = {cell_name}')
            # convert user res (4m in testing) size at center of cell for resolution purposes
            dx, dy = compute_delta_coord(cx, cy, *output_res, crs_transform, inv_crs_transform)

            bag_options_dict = {'VAR_INDIVIDUAL_NAME': 'Chief, Hydrographic Surveys Division',
                                'VAR_ORGANISATION_NAME': 'NOAA, NOS, Office of Coast Survey',
                                'VAR_POSITION_NAME': 'Chief, Hydrographic Surveys Division',
                                'VAR_DATE': datetime.now().strftime('%Y-%m-%d'),
                                'VAR_VERT_WKT': 'VERT_CS["unknown", VERT_DATUM["unknown", 2000]]',
                                'VAR_ABSTRACT': "This multi-layered file is part of NOAA Office of Coast Survey’s National Bathymetry. The National Bathymetric Source is created to serve chart production and support navigation. The bathymetry is compiled from multiple sources with varying quality and includes forms of interpolation. Soundings should not be extracted from this file as source data is not explicitly identified. The bathymetric vertical uncertainty is communicated through the associated layer. More generic quality and source metrics will be added with 2.0 version of the BAG format.",
                                'VAR_PROCESS_STEP_DESCRIPTION': f'Generated By GDAL {gdal.__version__} and NBS',
                                'VAR_DATETIME': datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                                'VAR_VERTICAL_UNCERT_CODE': 'productUncert',
                                # 'VAR_RESTRICTION_CODE=' + restriction_code,
                                # 'VAR_OTHER_CONSTRAINTS=' + other_constraints,
                                # 'VAR_CLASSIFICATION=' + classification,
                                # 'VAR_SECURITY_USER_NOTE=' + security_user_note
                                }
            tif_tags = {'EMAIL_ADDRESS': 'OCS.NBS@noaa.gov',
                        'ONLINE_RESOURCE': 'https://www.ngdc.noaa.gov',
                        'LICENSE': 'License cc0-1.0',
                        }

            export_utm = export_dir.joinpath(cell_name + "_utm.tif")
            x1, y1, x2, y2 = transform_rect(minx, miny, maxx, maxy, crs_transform.transform)
            dataset, dataset_score = db.make_export_rasters(export_utm, x1, y1, x2, y2, output_res)
            tile_count = 0

            for db_path in db_paths:
                db = WorldDatabase.open(db_path)
                tile_count += db.export_into_raster(dataset, dataset_score, compare_callback=comp)
            score_name = dataset_score.GetDescription()
            del dataset_score
            try:
                os.remove(score_name)
            except PermissionError:
                gc.collect()
                try:
                    os.remove(score_name)
                except PermissionError:
                    print(f"Failed to remove {score_name}, permission denied (in use?)")

            # cnt, utm_dataset = db.export_area(export_utm, x1, y1, x2, y2, output_res, compare_callback=comp)
            # export_path = export_dir.joinpath(cell_name + ".bag")
            # bag_options = [key + "=" + val for key, val in bag_options_dict.items()]
            # cnt2, ex_ds = db.export_area(export_path, minx, miny, maxx, maxy, (dx+dx*.1, dy+dy*.1), target_epsg=export_epsg,
            #                       driver='BAG', gdal_options=bag_options)

            if tile_count > 0:
                make_contributor_csv(export_utm, 3,
                                     table_names, database, username, password, hostname=hostname, port=port)
                # if not _debug:
                #     # output in native UTM -- Since the coordinates "twist" we need to check all four corners,
                #     # not just lower left and upper right
                # export_wgs = export_dir.joinpath(cell_name + "_wgs.tif")
                #     cnt, exported_dataset = db.export_area(export_wgs, minx, miny, maxx, maxy, (dx + dx * .1, dy + dy * .1),
                #                                            target_epsg=export_epsg, compare_callback=comp)
            else:
                del dataset  # close the gdal file
                gc.collect()  # make sure the gdal dataset released its lock
                # time.sleep(.3)
                os.remove(export_utm)
            print('done', cell_name)


def main():
    if len(sys.argv) > 1:
        use_configs = sys.argv[1:]
    else:
        use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))

    warnings = ""
    for config_filename, config_file in iter_configs(use_configs):
        stringio_warnings = set_stream_logging("bruty", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info(f'***************************** Start Run  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']

        try:
            resx, resy = map(float, config['resolution'].split(','))
        except:
            resx = resy = float(config['resolution'])
        clip = config['clip_to_shapefile'] if 'clip_to_shapefile' in config else None
        if _debug:
            hostname, port, username, password = None, None, None, None
            tablenames_raw, database = config['tablenames'], config['database']
            tablenames = parse_multiple_values(tablenames_raw)
        else:
            tablenames, database, hostname, port, username, password = connect_params_from_config(config)
        db_paths = parse_multiple_values(config['combined_datapaths'])
        export(db_paths, config['output_directory'], config['export_areas_shapefile'], config['name_from_field'],
               (resx, resy), clip, tablenames, database, username, password, hostname, port)


if __name__ == '__main__':

    # default_config_name = "default.config"

    # turn prints into logger messages
    orig_print = print
    def print(*args, **kywds):
        f = io.StringIO()
        ky = kywds.copy()
        ky['file'] = f
        orig_print(*args, **ky)  # build the string
        LOGGER.info(f.getvalue()[:-1])  # strip the newline at the end
    main()

