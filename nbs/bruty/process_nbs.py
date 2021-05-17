# import sys
# sys.path.append(r"C:\Git_Repos\nbs")
# sys.path.append(r"C:\Git_Repos\bruty")
import os
import sys
import traceback
from datetime import datetime
import subprocess
import pathlib
import shutil
from functools import partial
import logging
import io

import numpy
import rasterio
from osgeo import gdal, osr, ogr

from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review.meta_review import database_has_table, split_URL_port
from nbs.bruty.raster_data import TiffStorage, LayersEnum
from nbs.bruty.history import DiskHistory, RasterHistory, AccumulationHistory
from nbs.bruty.world_raster_database import WorldDatabase, UTMTileBackend
from nbs.bruty.utils import onerr
from nbs.bruty.utils import get_crs_transformer, compute_delta_coord, transform_rect

_debug = True


def make_serial_column(table_id, table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434', big=False):
    connection = connect_with_retries(database=database, user=username, password=password, host=hostname, port=port)
    cursor = connection.cursor()
    # used admin credentials for this
    # cursor.execute("create table serial_19n_mllw as (select * from pbc_utm19n_mllw)")
    # connection.commit()
    if big:
        serial_str = "bigserial"  # use bigserial if 64bit is needed
        bit_shift = 32  # allows 4 billion survey IDs and 4 billion table IDs
    else:
        serial_str = "serial"  # 32bit ints
        bit_shift = 20  # allows for one million survey IDs and 4096 table IDs
    start_val = table_id << bit_shift
    cursor.execute(f'ALTER TABLE {table_name} ADD column sid {serial_str};')
    cursor.execute(f'ALTER SEQUENCE {table_name}_sid_seq RESTART WITH 10000')
    cursor.execute(f"update {table_name} set sid=sid+{start_val}")
    connection.commit()


def get_nbs_records(table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    if _debug and hostname is None:
        import pickle
        f = open(r"C:\data\nbs\pbc19_mllw_metadata.pickle", 'rb')
        records = pickle.load(f)
        fields = pickle.load(f)
    else:
        connection = connect_with_retries(database=database, user=username, password=password, host=hostname, port=port)
        cursor = connection.cursor()
        cursor.execute(f'SELECT * FROM {table_name}')
        records = cursor.fetchall()
        fields = [desc[0] for desc in cursor.description]
    return fields, records


def nbs_survey_sort(id_to_score, pts, existing_arrays):
    # return arrays that merge_arrays will use for sorting.
    # basically the nbs sort is 4 keys: Decay Score, resolution, depth, alphabetical.
    # Decay and resolution get merged into one array since they are true for all points of the survey while depth varies with position.
    # alphabetical is a final tie breaker to make sure the same contributor is picked in the cases where the first three tie.

    # find all the contributors to look up
    all_contributors = existing_arrays[LayersEnum.CONTRIBUTOR]
    unique_contributors = numpy.unique(all_contributors[~numpy.isnan(all_contributors)])
    # make arrays to store the integer scores in
    existing_decay_and_res = all_contributors.copy()
    existing_alphabetical = all_contributors.copy()
    # for each unique contributor fill with the associated decay/resolution score and the alphabetical score
    for contrib in unique_contributors:
        existing_decay_and_res[all_contributors == contrib] = id_to_score[contrib][0]
        existing_alphabetical[all_contributors == contrib] = id_to_score[contrib][1]
    existing_elevation = existing_arrays[LayersEnum.ELEVATION]
    # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
    pts_contributors = pts[LayersEnum.CONTRIBUTOR + 2]
    unique_pts_contributors = numpy.unique(pts_contributors[~numpy.isnan(pts_contributors)])
    decay_and_res_score = pts_contributors.copy()
    alphabetical = pts_contributors.copy()
    for contrib in unique_pts_contributors:
        decay_and_res_score[pts_contributors == contrib] = id_to_score[contrib][0]
        alphabetical[pts_contributors == contrib] = id_to_score[contrib][1]
    elevation = pts[LayersEnum.ELEVATION + 2]

    return numpy.array((decay_and_res_score, elevation, alphabetical)), \
           numpy.array((existing_decay_and_res, existing_elevation, existing_alphabetical)), \
           (False, False, False)


def id_to_scoring(fields, records):
    # Create a dictionary that converts from the unique database ID to an ordering score
    # Basically the standings of the surveys,
    # First place is the highest decay score with a tie breaker of lowest resolution.  If both are the same they will have the same score
    # Alphabetical will have no duplicate standings (unless there is duplicate names) and first place is A (ascending alphabetical)
    # get the columns that have the important data
    decay_col = fields.index("decay_score")
    script_res_col = fields.index('script_resolution')
    manual_res_col = fields.index('manual_resolution')
    script_point_res_col = fields.index('script_point_spacing')
    manual_point_res_col = fields.index('manual_point_spacing')

    filename_col = fields.index('from_filename')
    path_col = fields.index('script_to_filename')
    manual_path_col = fields.index('manual_to_filename')
    id_col = fields.index('sid')
    rec_list = []
    names_list = []
    # make lists of the dacay/res with survey if and also one for name vs survey id
    for rec in records:
        decay = rec[decay_col]
        sid = rec[id_col]
        if decay is not None:
            res = rec[manual_res_col]
            if res is None:
                res = rec[script_res_col]
                if res is None:
                    res = rec[script_point_res_col]
                    if res is None:
                        res = rec[manual_point_res_col]
                        if res is None:
                            print("missing res on record:", sid, rec[filename_col])
                            continue
            path = rec[manual_path_col]
            # A manual string can be an empty string (not Null) and also protect against it looking empty (just a space " ")
            if path is None or not path.strip():
                path = rec[path_col]
                if path is None or not path.strip():
                    print("skipping missing to_path", sid, rec[filename_col])
                    continue
            rec_list.append((sid, res, decay))
            # Switch to lower case, these were from filenames that I'm not sure are case sensitive
            names_list.append((rec[filename_col].lower(), sid, path))  # sid would be the next thing sorted if the names match
    # sort the names so we can use an integer to use for sorting by name
    names_list.sort()
    # do an ordered 2 key sort on decay then res (lexsort likes them backwards)
    rec_array = numpy.array(rec_list)
    sorted_indices = numpy.lexsort([-rec_array[:, 1], rec_array[:, 2]])  # resolution, decay (flip the res so lowest score and largest res is first)
    sorted_recs = rec_array[sorted_indices]
    sort_val = 0
    prev_res, prev_decay = None, None
    sort_dict = {}
    # set up a dictionary that has the sorted value of the decay followed by resolution
    for n, (sid, res, decay) in enumerate(sorted_recs):
        # don't incremenet when there was a tie, this allows the next sort criteria to be checked
        if res != prev_res or decay != prev_decay:
            sort_val += 1
        sort_dict[sid] = [sort_val]
    # the NBS sort order then uses depth after decay but before alphabetical, so we can't merge the name sort with the decay+res
    # add a second value for the alphabetical naming which is the last resort to maintain constistency of selection
    for n, (filename, sid, path) in enumerate(names_list):
        sort_dict[sid].append(n)
    return sorted_recs, names_list, sort_dict


def process_nbs_database(world_db_path, table_name, database, username, password, hostname='OCS-VS-NBS01', port='5434'):
    fields, records = get_nbs_records(table_name, database, username, password, hostname=hostname, port=port)
    sorted_recs, names_list, sort_dict = id_to_scoring(fields, records)
    try:
        db = WorldDatabase.open(world_db_path)
    except FileNotFoundError:
        epsg
        db = WorldDatabase(UTMTileBackend(epsg, RasterHistory, DiskHistory, TiffStorage, world_db_path))  # NAD823 zone 19.  WGS84 would be 32619
    path_col = fields.index('script_to_filename')
    id_col = fields.index('sid')
    comp = partial(nbs_survey_sort, sort_dict)
    total = 0
    print('------------   changing paths !!!!!!!!!!')
    for i, (filename, sid, path) in enumerate(names_list):
        copy_data = False
        path_e = path.lower().replace('\\\\nos.noaa\\ocs\\hsd\\projects\\nbs\\nbs_data\\pbc_northeast_utm19n_mllw',
                                      r'E:\Data\nbs\PBC_Northeast_UTM19N_MLLW')
        path_c = path.lower().replace('\\\\nos.noaa\\ocs\\hsd\\projects\\nbs\\nbs_data\\pbc_northeast_utm19n_mllw',
                                      r'C:\Data\nbs\PBC_Northeast_UTM19N_MLLW')
        if copy_data:
            try:
                os.makedirs(os.path.dirname(path_c), exist_ok=True)
                if not path_e.endswith("csar"):
                    pass
                    # shutil.copy(path_e, path_c)
                else:
                    for mod_fname in (f"{path_e}.elev.tif", f"{path_e}.depth.tif", f"{path_e}.csv.zip"):
                        if os.path.exists(mod_fname):
                            shutil.copy(mod_fname, "c"+mod_fname[1:])
                            try:
                                os.remove(path_c)  # take the csar off disk
                                os.remove(path_c+"0")
                            except:
                                pass
            except FileNotFoundError:
                print("File missing", sid, path)
        # convert csar names to exported data, 1 of 3 types
        if path.endswith("csar"):
            for mod_fname in (f"{path_c}.elev.tif", f"{path_c}.depth.tif", f"{path_c}.csv.zip"):
                if os.path.exists(mod_fname):
                    path = mod_fname
        else:
            path = path_c
        if not os.path.exists(path):
            print(path, "didn't exist")
            continue
        # # @FIXME is contributor an int or float -- needs to be int 32 and maybe int 64 (or two int 32s)
        print('starting', path)
        print(datetime.now().isoformat(), i, "of", len(names_list))
        if sid not in db.included_ids:
            if path.endswith(".csv.zip"):
                csv_path = path[:-4]
                p = subprocess.Popen(f'python -m zipfile -e "{path}" "{os.path.dirname(path)}"')
                p.wait()
                if os.path.exists(csv_path):
                    try:
                        # points are in opposite convention as BAGs and exported CSAR tiffs, so reverse the z component
                        db.insert_txt_survey(csv_path, format=[('x', 'f8'), ('y', 'f8'), ('depth', 'f4'), ('uncertainty', 'f4')],
                                             override_epsg=db.db.epsg, contrib_id=sid, compare_callback=comp, reverse_z=True)
                    except ValueError:
                        print("Value Error")
                        print(traceback.format_exc())
                    os.remove(f'{csv_path}')
                else:
                    print("\n\nCSV was not extracted from zip\n\n\n")
            else:
                db.insert_survey(path, override_epsg=db.db.epsg, contrib_id=sid, compare_callback=comp)
            print('inserted', path)

    print('data MB:', total / 1000000)

def convert_csar():
    cnt = 0
    for record in rc:
        fname = record[4]  # manual_to_filename
        if fname is None or not fname.strip():
            fname = record[3]  # script_to_filename
        if fname is not None and fname.strip().lower().endswith("csar"):
            if record[67] or record[68]:  # has grid filled out
                local_fname = fname.lower().replace('\\\\nos.noaa\\OCS\\HSD\\Projects\\NBS\\NBS_Data'.lower(), r"E:\Data\nbs")
                if not os.path.exists(f"{local_fname}.csv.zip") and not os.path.exists(f"{local_fname}.depth.tif") and not os.path.exists(
                        f"{local_fname}.elev.tif"):
                    cnt += 1
                    print(local_fname)
                    cmd = f'{carisbatch} -r ExportRaster --output-format GEOTIFF --compression LZW --include-band Depth --include-band Uncertainty "{local_fname}" "{local_fname}.depth.tif"'
                    p = subprocess.Popen(cmd)
                    p.wait()
                    if not os.path.exists(f"{local_fname}.depth.tif"):
                        cmd = f'{carisbatch} -r ExportRaster --output-format GEOTIFF --compression LZW --include-band Elevation --include-band Uncertainty "{local_fname}" "{local_fname}.elev.tif"'
                        p = subprocess.Popen(cmd)
                        p.wait()
                        if not os.path.exists(f"{local_fname}.elev.tif"):
                            cmd = f'{carisbatch} -r exportcoveragetoascii --include-band Depth 3 --include-band Uncertainty 3 --output-crs EPSG:26919 --coordinate-format GROUND --coordinate-precision 2 --coordinate-unit m "{local_fname}" "{local_fname}.csv"'
                            p = subprocess.Popen(cmd)
                            p.wait()
                            if os.path.exists(f"{local_fname}.csv"):
                                p = subprocess.Popen(f'python -m zipfile -c "{local_fname}.csv.zip" "{local_fname}.csv"')
                                p.wait()
                                os.remove(f'{local_fname}.csv')
                                print("was points")
                            else:
                                print("failed as points and raster????????????????????")
                        else:
                            print("was raster")
                            break

if __name__ == '__main__':
    # data_dir = pathlib.Path(__file__).parent.parent.parent.joinpath('tests').joinpath("test_data_output")
    data_dir = pathlib.Path("c:\\data\\nbs\\test_data_output")  # avoid putting in the project directory as pycharm then tries to cache everything I think
    build = True
    export = False
    def make_clean_dir(name):
        use_dir = data_dir.joinpath(name)
        if os.path.exists(use_dir):
            shutil.rmtree(use_dir, onerror=onerr)
        os.makedirs(use_dir)
        return use_dir

    db_path = data_dir.joinpath(r"test_pbc_19_db_metacheck")
    make_clean_dir(r"test_pbc_19_db_metacheck")

    if not os.path.exists(db_path.joinpath("wdb_metadata.json")):
        build = True
        db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, db_path))  # NAD823 zone 19.  WGS84 would be 32619
        del db

    # create logger with 'spam_application'
    logger = logging.getLogger('process_nbs')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(db_path.joinpath('process_nbs.log'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    orig_print = print


    def print(*args, **kywds):
        f = io.StringIO()
        ky = kywds.copy()
        ky['file'] = f
        orig_print(*args, **ky)  # build the string
        # orig_print(f.getvalue())  # logger is printing to screen now
        logger.info(f.getvalue())


    print("Using database at", db_path)
    # db_path = make_clean_dir(r"test_pbc_19_db")  # reset the database

    if build:
        # logging.basicConfig(filename=db_path.joinpath("build.log"), format='%(asctime)s %(message)s', encoding='utf-8', level=logging.DEBUG)
        URL_FILENAME = r"c:\data\nbs\postgres_hostname.txt"
        CREDENTIALS_FILENAME = r"c:\data\nbs\postgres_scripting.txt"
        if _debug:
            hostname = None
            port = None
            username = None
            password = None
        else:
            with open(URL_FILENAME) as hostname_file:
                url = hostname_file.readline()
            hostname, port = split_URL_port(url)

            with open(CREDENTIALS_FILENAME) as database_credentials_file:
                username, password = [line.strip() for line in database_credentials_file][:2]

        table_name = 'serial_19n_mllw'
        database = 'metadata'
        process_nbs_database(db_path, table_name, database, username, password, hostname, port)

    if export:
        db = WorldDatabase.open(db_path)
        export_dir = make_clean_dir("pbc19_exports")
        all_tiles_shape_fnames = ((r"G:\Data\NBS\Support_Files\MCD_Bands\Band3\Band3_V6.shp", (16, 16)),
                                  (r"G:\Data\NBS\Support_Files\MCD_Bands\Band4\Band4_V6.shp", (8, 8)),
                                  (r"G:\Data\NBS\Support_Files\MCD_Bands\Band5\Band5_V6.shp", (4, 4)),
                             )
        export_area_shp_fname = r"G:\Data\NBS\Support_Files\MCD_Bands\PBC_Review\PBC_UTM19N_Review.shp"
        ds_pbc = gdal.OpenEx(export_area_shp_fname)  # this is the product branch area rough extents, PBC, PBD, PBG etc
        pb_lyr = ds_pbc.GetLayer(0)
        pb_srs = pb_lyr.GetSpatialRef()
        pb_export_epsg = rasterio.crs.CRS.from_string(pb_srs.ExportToWkt()).to_epsg()
        pb_minx, pb_maxx, pb_miny, pb_maxy = pb_lyr.GetExtent()  # this is the product branch area rough extents, PBC, PBD, PBG etc

        for tiles_shape_fname, output_res in all_tiles_shape_fnames:
            ds = gdal.OpenEx(tiles_shape_fname)
            # ds.GetLayerCount()
            lyr = ds.GetLayer(0)
            srs = lyr.GetSpatialRef()
            export_epsg = rasterio.crs.CRS.from_string(srs.ExportToWkt()).to_epsg()
            lyr.GetFeatureCount()
            lyrdef = lyr.GetLayerDefn()
            for i in range(lyrdef.GetFieldCount()):
                flddef = lyrdef.GetFieldDefn(i)
                if flddef.name == "CellName":
                    cell_field = i
                    break
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
                if pb_minx < cx < pb_maxx and pb_miny < cy < pb_maxy:
                    cell_name = feat.GetField(cell_field)
                    if _debug:
                        pass
                        # if cell_name not in ('US5BPGBD',):  # 'US5BPGCD'):
                        #     continue


                    print(cell_name)
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
                    export_wgs = export_dir.joinpath(cell_name + "_wgs.tif")
                    # FIXME this is using the wrong score, need to use the nbs_database scoring when exporting as well as when combining.
                    x1, y1, x2, y2 = transform_rect(minx, miny, maxx, maxy, crs_transform.transform)
                    cnt, utm_dataset = db.export_area(export_utm, x1, y1, x2, y2, output_res)
                    # export_path = export_dir.joinpath(cell_name + ".bag")
                    # bag_options = [key + "=" + val for key, val in bag_options_dict.items()]
                    # cnt2, ex_ds = db.export_area(export_path, minx, miny, maxx, maxy, (dx+dx*.1, dy+dy*.1), target_epsg=export_epsg,
                    #                       driver='BAG', gdal_options=bag_options)

                    if cnt > 0:
                        if not _debug:
                            # output in native UTM -- Since the coordinates "twist" we need to check all four corners,
                            # not just lower left and upper right
                            cnt, exported_dataset = db.export_area(export_wgs, minx, miny, maxx, maxy, (dx + dx * .1, dy + dy * .1),
                                                                   target_epsg=export_epsg)
                    else:
                        utm_dataset = None  # close the gdal file
                        os.remove(export_utm)
                    print('done', cell_name)
