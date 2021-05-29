import os
import time
import re
import sys
import msvcrt
import subprocess
# print('arguments were')
# for a in sys.argv:
#     print(a)
# raise Exception("Stop")
#
import pathlib
import shutil
from shapely import wkt, wkb
from osgeo import ogr, osr, gdal
import pickle
from functools import partial

import numpy
import psycopg2

from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review import meta_review
from nbs.bruty.utils import get_crs_transformer, make_mllw_height_wkt, user_quit, tqdm
from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, connect_params_from_config, make_contributor_csv
from nbs.bruty.world_raster_database import WorldDatabase, use_locks
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, start_server, current_address
from nbs.bruty.generalize import generalize
from nbs.bruty.raster_attribute_table import make_raster_attr_table
from xipe_dev.xipe.raster import where_not_nodata
from nbs.bruty.raster_data import LayersEnum


#  @todo  Make this script into a more robust modular program
#    - Allow for multiple output types (public+sensitive and public+prereview) at once in an efficient way (only export public once in this example)
#    - Only download the needed metadata (based on which databases are used, say PBC_UTM18 or PBG14)
#    - Make a class or functions for determining these
#      - maybe a class that holds the postgres metadata and the TIF output and can be copied then expanded with another metadata table
#    -


"""
1) Get tile geometries and attributes
2) Expand tile geometry based on closing distance
3) Extract data from all necessary Bruty DBs
4) Combine if more than one DB used
5) Binary closing on raster covering expanded area
6) Interpolation only on closed cells that had no bathymetry
7) Export original extents to raster
8) Create Raster Attribute Table (RAT)
"""

# 1) Get tile geometries and attributes
_debug = False
b_glen_interp = True

data_dir = pathlib.Path(r"E:\bruty_databases")
if _debug:
    data_dir = data_dir.joinpath('debug')


def remove_file(pth: (str, pathlib.Path), allow_permission_fail: bool = False, limit: int = 2, nth: int = 1, tdelay: float = 2, silent: bool = False):
    """ Try to remove a file and just print a warning if it doesn't work.
    Will retry 'nth' times every 'tdelay' seconds up to 'limit' times.
    Will not raise an error on FileNotFound but will for PermissionError unless allow_permission_fail is set to True.

    Parameters
    ----------
    pth
        path to the file to remove
    allow_permission_fail
        True = continue trying to remove the file if a permission error is encountered, False = raise exception
    limit
        number of attempts to make
    nth
        attempt number this is
    tdelay
        time in seconds to wait between attempts
    silent
        False = print a message when the file isn't removed due to not being found or,
        depending on allow_permission_fail, file being in use/not having permissions

    Returns
    -------

    """

    if allow_permission_fail:
        ok_except = (FileNotFoundError, PermissionError)
    else:
        ok_except = (FileNotFoundError, )
    try:
        os.remove(pth)
    except ok_except as ex:
        if nth > limit:
            if not silent:
                print(f"File not found or permission error {type(ex)}, {pth}")
        else:
            time.sleep(tdelay)
            remove_file(pth, allow_permission_fail, nth=nth+1, silent=silent)


try:
    search_exp = sys.argv[1]
    print(f"exporting tiles with {search_exp} in the name")
except IndexError:
    search_exp = ""  # this will always pass the regular expression

try:
    # allowing batch file date time to be supplied - when the launcher becomes a python program instead of batch it should be easier
    day_of_week = sys.argv[2]
    date_string = sys.argv[3]  # 02/24/2022
    time_string = sys.argv[4]  # 10:44:14.44 or 13:01:22.45 or " 1:01:22.45" -- there is a leading space and not zero padded
    h, m, s = time_string.split(":")
    if len(h) < 2:  # leading zero for hours
        h = "0" + h
    s = s.split(".")[0]  # get rid of fraction of seconds
    postfix = "_" + date_string[6:10] + date_string[0:2] + date_string[3:5] + "_" + h + m + s
except IndexError:
    postfix = ""

if True:  # not _debug
    user = ""
    password = ""
    host = ''
    port = ''
    connection = connect_with_retries(database="tile_specifications", user=user, password=password, host=host, port=port)
    cursor = connection.cursor()
    cursor.execute(f'SELECT *,ST_SRID(geometry) FROM bruty_bluetopo')
    records = cursor.fetchall()
    fields = [desc[0] for desc in cursor.description]
    pickle.dump([records, fields], open(data_dir.joinpath("bluetopo.pickle"), "wb"))
    table_names = []
    if "PBG14" in search_exp.upper():
        table_names.extend(["pbg_gulf_utm14n_mllw", "pbg_gulf_utm14n_mllw_sensitive",
                            "pbg_gulf_utm14n_mllw_prereview", "pbg_gulf_utm14n_mllw_enc"])
    if "PBG15" in search_exp.upper():
        table_names.extend(["pbg_gulf_utm15n_mllw", "pbg_gulf_utm15n_mllw_sensitive",
                            "pbg_gulf_utm15n_mllw_prereview", "pbg_gulf_utm15n_mllw_enc"])
    if "PBG16" in search_exp.upper():
        table_names.extend(["pbg_gulf_utm16n_mllw", "pbg_gulf_utm16n_mllw_sensitive",
                            "pbg_gulf_utm16n_mllw_prereview", "pbg_gulf_utm16n_mllw_enc", ])
    if "PBC19" in search_exp.upper():
        table_names.extend(["pbc_utm19n_mllw", "pbc_utm19n_mllw_sensitive",
                            "pbc_utm19n_mllw_prereview", "pbc_utm19n_mllw_enc", ])
    if "PBCHRD" in search_exp.upper():
        table_names.extend(["pbc_utm18n_hrd", "pbc_utm18n_hrd_sensitive",
                            "pbc_utm18n_hrd_prereview", "pbc_utm18n_hrd_enc", ])
    if "PBC18" in search_exp.upper():
        table_names.extend(["pbc_utm18n_mllw", "pbc_utm18n_mllw_sensitive",
                            "pbc_utm18n_mllw_prereview", "pbc_utm18n_mllw_enc", ])
    if "PBD11" in search_exp.upper():
        table_names.extend(["pbd_utm11n_mllw_lalb", "pbd_utm11n_mllw_lalb_prereview",
                            "pbd_utm11n_mllw_lalb_sensitive", "pbd_utm11n_mllw_lalb_enc", ])
    if "PBG18" in search_exp.upper():
        table_names.extend(["pbg_navassa_utm18n_mllw", "pbg_navassa_utm18n_mllw_sensitive",
                            "pbg_navassa_utm18n_mllw_prereview", "pbg_navassa_utm18n_mllw_enc", ])
    if "PBG19" in search_exp.upper():
        table_names.extend(["pbg_puertorico_utm19n_mllw", "pbg_puertorico_utm19n_mllw_sensitive",
                            "pbg_puertorico_utm19n_mllw_prereview", "pbg_puertorico_utm19n_mllw_enc", ])
    if "PBG20" in search_exp.upper():
        table_names.extend(["pbg_puertorico_utm20n_mllw", "pbg_puertorico_utm20n_mllw_sensitive",
                            "pbg_puertorico_utm20n_mllw_prereview", "pbg_puertorico_utm20n_mllw_enc", ])
    metadata_fields = []
    metadata_records = []
    for table_name in table_names:
        try:
            mfields, mrecords = get_nbs_records(table_name, "metadata", user, password, hostname=host, port=port)
            metadata_records.append(mrecords)
            metadata_fields.append(mfields)
        except psycopg2.errors.UndefinedTable:
            print(f"{table_name} doesn't exist.")
    pickle.dump([metadata_records, metadata_fields], open(data_dir.joinpath("metadata.pickle"), "wb"))
else:
    records, fields = pickle.load(open(data_dir.joinpath("bluetopo.pickle"), "rb"))
    metadata_records, metadata_fields = pickle.load(open(data_dir.joinpath("metadata.pickle"), "rb"))

# @todo move this record stuff into a function in nbs_postgres.py
all_meta_records = {}
all_simple_records = {}
for n, meta_table_recs in enumerate(metadata_records):
    id_col = metadata_fields[n].index('nbs_id')
    for record in meta_table_recs:
        record_dict = dict(zip(metadata_fields[n], record))
        simple_record = meta_review.MetadataDatabase._simplify_record(record_dict)
        simple_fuse_record = meta_review.records_to_fusemetadata(simple_record)  # re-casts the db values into other/desired types
        all_simple_records[record[id_col]] = simple_fuse_record
    all_meta_records.update({rec[id_col]: rec for rec in meta_table_recs})
name_index = fields.index("name")
resolution_index = fields.index("resolution")
closing_index = fields.index("closing_distance")
sensitive_index = fields.index('data_sensitive')
reviewed_index = fields.index('data_qualified')
prereview_index = fields.index('data_unqualified')
enc_index = fields.index('data_enc')
not_for_nav_index = fields.index('quality_filter')
geom_index = fields.index('geometry')
epsg_index = fields.index('st_srid')

sorted_recs, names_list, sort_dict = id_to_scoring(metadata_fields, metadata_records, for_navigation_flag=(False, False),
                                                   never_post_flag=(False, False))
comp = partial(nbs_survey_sort, sort_dict)

# print(fields)
# ['name', 'active', 'resolution', 'closing_distance', 'data_sensitive', 'data_qualified', 'data_unqualified', 'quality_filter',
# 'contributor_attributes', 'geometry', 'st_srid']

# print(r)
# ('Tile7_PBC18_4', None, 4.0, 100.0, False, True, False, False, None,
# '01030000000100000009000000CDCCCCCCCC<snip>',  26918)

# @todo keep multiple lock servers open on different ports, 5000 for the tile names then eventually different ones for each utm zone
use_locks(5000)

for tile_record in records:
    if user_quit():
        break
    try:
        with NameLock(tile_record[name_index]) as lck:  # this doesn't work with the file lock - just the multiprocessing locks
            # USING GDAL
            try:
                g = ogr.CreateGeometryFromWkb(bytes.fromhex(tile_record[geom_index]))
                minx, maxx, miny, maxy = g.GetEnvelope()
                # Out[44]: (-74.4, -73.725, 40.2, 40.5)

                # g.GetGeometryRef(0).GetPoints()
                # Out[48]:
                # [(-73.95, 40.2), (-74.1, 40.2), (-74.1, 40.425),  (-74.4, 40.425), (-74.4, 40.5),
                #  (-73.725, 40.5), (-73.725, 40.35), (-73.95, 40.35), (-73.95, 40.2)]
            except AttributeError:
                # *************************************************************************************************
                # USING SHAPELY
                g = wkb.loads(tile_record[geom_index], hex=True)

                minx, miny, maxx, maxy = g.bounds
                # Out[49]: (-74.4, 40.2, -73.725, 40.5)

                # g.boundary.xy
                # Out[28]:
                # (array('d', [-73.95, -74.1, -74.1, -74.4, -74.4, -73.725, -73.725, -73.95, -73.95]),
                #  array('d', [40.2, 40.2, 40.425, 40.425, 40.5, 40.5, 40.35, 40.35, 40.2]))

            # 2) Expand tile geometry based on closing distance

            closing_dist = tile_record[closing_index]
            if search_exp and not re.search(search_exp, tile_record[name_index], re.IGNORECASE):
                print(search_exp, "not in ", tile_record[name_index])
                continue
            print("exporting ", tile_record[name_index])
            if 'PBG14' in tile_record[name_index].upper():
                # use_locks(5014)
                target_epsg = 26914  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_gulf_utm14n_mllw"
                output_subdir = "PBG14"
            elif 'PBG15' in tile_record[name_index].upper():
                # use_locks(5015)
                target_epsg = 26915  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_gulf_utm15n_mllw"
                output_subdir = "PBG15"
            elif 'PBG16' in tile_record[name_index].upper():
                # use_locks(5016)
                target_epsg = 26916  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_gulf_utm16n_mllw"
                output_subdir = "PBG16"
            elif 'PBG18' in tile_record[name_index].upper():
                # use_locks(5016)
                target_epsg = 26918  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_navassa_utm18n_mllw"
                output_subdir = "PBG18"
            elif 'PBG19' in tile_record[name_index].upper():
                # use_locks(5016)
                target_epsg = 26919  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_puertorico_utm19n_mllw"
                output_subdir = "PBG19"
            elif 'PBG20' in tile_record[name_index].upper():
                # use_locks(5016)
                target_epsg = 26920  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbg_puertorico_utm20n_mllw"
                output_subdir = "PBG20"
            elif 'PBCHRD' in tile_record[name_index].upper():
                # use_locks(5018)
                target_epsg = 26918  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbc_utm18n_hrd"
                output_subdir = "PBC18_hrd"
            elif 'PBC18' in tile_record[name_index].upper():
                # use_locks(5018)
                target_epsg = 26918  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbc_utm18n_mllw"
                output_subdir = "PBC18"
            elif 'PBC19' in tile_record[name_index].upper():
                # use_locks(5019)
                target_epsg = 26919  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbc_utm19n_mllw"
                output_subdir = "PBC19"
            elif 'PBD11' in tile_record[name_index].upper():
                # use_locks(5011)
                target_epsg = 26911  # MCD WGS84
                db_epsg = tile_record[epsg_index]
                base_db = "pbd_utm11n_mllw"
                output_subdir = "PBD11"
            else:
                print(tile_record[name_index], "export instructions not found, skipping")
                continue

            crs_transform = get_crs_transformer(target_epsg, db_epsg)
            if crs_transform:
                xs, ys = [], []
                for x in (minx, maxx):
                    for y in (miny, maxy):
                        cx, cy = crs_transform.transform(x, y)
                        xs.append(cx)
                        ys.append(cy)
                minx = min(xs)
                maxx = max(xs)
                miny = min(ys)
                maxy = max(ys)

            # 3) Extract data from all necessary Bruty DBs
            db_base_path = data_dir
            REVIEWED = ""
            NOT_NAV = "_not_for_navigation"
            PREREVIEW = "_prereview"
            SENSITIVE = "_sensitive"
            ENC = "_enc"
            # @TODO - move all this to config ini files
            blue_topo = {reviewed_index: True, prereview_index: True, sensitive_index: False, enc_index: True, not_for_nav_index: True,
                         "name": "BlueTopo"}  # public use
            navigation = {reviewed_index: True, prereview_index: False, sensitive_index: True, enc_index: True, not_for_nav_index: False,
                          "name": "Navigation"}  # MCD charting use
            internal = {reviewed_index: True, prereview_index: True, sensitive_index: True, enc_index: True, not_for_nav_index: True,
                        "name": "Internal"}  # HSD planning use
            # @TODO make a cache of reviewed+enc since that gets used everytime
            for export_type in (blue_topo, navigation, internal):
                print(f"starting {export_type['name']}")
                databases = []
                tables = []
                for datatype, ext in [(reviewed_index, REVIEWED), (prereview_index, PREREVIEW), (sensitive_index, SENSITIVE), (enc_index, ENC)]:
                    if export_type[datatype]:  # should we add the main datatype?  (reviewed/qualified, prereview/unqualified, sensitive)
                        databases.append(base_db+ext)
                        # ENC is a navigation only product, so don't even look for enc_not_for_navigation
                        if export_type[not_for_nav_index] and ext != ENC:  # should we add the "not for navigation" data?
                            databases.append(base_db + ext + NOT_NAV)
                # make into
                all_times = []
                # @todo enc not_for_navigation won't exist - allow to fail gracefully?  What if public_not_for_navigation doesn't exist?
                remove_databases = []
                for db_name in databases:
                    try:
                        db = WorldDatabase.open(db_base_path.joinpath(db_name))
                    except FileNotFoundError as e:
                        if _debug:
                            remove_databases.append(db_name)
                            continue

                        if False:
                            k = b"T"
                            while k not in b"yYnN":
                                print(f"{db_name} not found, do you want to continue without it y/n?")
                                k = msvcrt.getch()
                            if k in b"yY":
                                remove_databases.append(db_name)
                                continue
                        raise e

                    all_times.extend([rec[1] for rec in db.transaction_groups.values()])
                    del db
                # remove any databases the user said was ok to skip
                for db_name in remove_databases:
                    databases.remove(db_name)

                if all_times:
                    last_time = max(all_times)
                    timestamp = last_time.strftime("%Y%m%d_%H%M%S")
                else:
                    timestamp = ""
                output_base_path = pathlib.Path(r"\\nos.noaa\OCS\HSD\Projects\NBS\bruty_tile_exports").joinpath(output_subdir).joinpath(export_type['name']).joinpath(timestamp)
                if _debug:
                    output_base_path = output_base_path.joinpath("debug")
                os.makedirs(output_base_path, exist_ok=True)
                fname_base = tile_record[name_index] + "_" + export_type['name'] + "_" + timestamp + "_" + str(int(closing_dist)) +"m" + postfix
                export_generalized_filename = output_base_path.joinpath(fname_base + ".generalized.tif")
                extracted_filename = output_base_path.joinpath(fname_base + " - original.tif")
                export_contributors_filename = output_base_path.joinpath(fname_base + ".revise_contrib.tif")
                export_cog_filename = output_base_path.joinpath(fname_base + ".tif")
                score_filename = extracted_filename.with_suffix(".score.tif")
                rat_filename = export_cog_filename.with_suffix(".tif.aux.xml")
                if os.path.exists(str(export_cog_filename) + ".aux.xml") and os.path.exists(rat_filename):
                    print("Already exported - skipping")
                    continue
                # re-export in case the previous export didn't finish
                if not _debug:
                    remove_file(extracted_filename, silent=True)
                    remove_file(score_filename, silent=True)
                if not _debug or not extracted_filename.exists():
                    dataset, dataset_score = None, None
                    cnt = 0
                    try:
                        for db_name in tqdm(databases, desc='bruty database', mininterval=.7):
                            db = WorldDatabase.open(db_base_path.joinpath(db_name))
                            if dataset is None:
                                wkt = make_mllw_height_wkt(db.db.tile_scheme.epsg)
                                res = tile_record[resolution_index]
                                # center the output cells at origin of UTM like Coast Survey standard -- this will align with Coast Survey Bruty tiles
                                # basically a cell center would fall at 0,0 of the coordinate system
                                use_minx = ((minx - closing_dist) // res) * res - res / 2.0
                                use_miny = ((miny - closing_dist) // res) * res - res / 2.0
                                dataset, dataset_score = db.make_export_rasters(extracted_filename, use_minx, use_miny,
                                                                                maxx + closing_dist, maxy + closing_dist, res)
                                dataset.SetProjection(wkt)
                            try:
                                cnt += db.export_into_raster(dataset, dataset_score, compare_callback=comp)
                            except KeyError as e:
                                print("KeyError", str(e))
                                open(output_base_path.joinpath(fname_base+f" - keyerror {str(e)}"), "w")
                                raise BrutyMissingScoreError(str(e))
                    except BrutyMissingScoreError:
                        continue
                    # if cnt == 0:
                    #     print("no data found in the databases for this area - skipping")
                    #     continue
                    del dataset, dataset_score  # release the file handle for the score file

                    # FIXME -- remove this temporary copy and exporting to "-original"
                    remove_file(export_generalized_filename, silent=True)
                    remove_file(export_cog_filename, silent=True)

                # FIXME - remove this change of nodata value once NBS is fixed to accept NaN in addition to 1000000
                #   also this give a problem in the HACK of storing large int32 in float32 tiffs
                #   the 1000000 becomes 1232348160 when translated back, so a contributor of 1232348160 would get stepped on by NaN
                #   numpy.frombuffer(numpy.array(1000000).astype(numpy.float32).tobytes(), numpy.int32)
                if not _debug or not export_generalized_filename.exists():
                    print("start generalize")
                    shutil.copyfile(extracted_filename, export_generalized_filename)
                    ds = gdal.Open(str(export_generalized_filename), gdal.GA_Update)
                    new_nodata = 1000000
                    for b in range(1, 4):
                        band = ds.GetRasterBand(b)
                        data = band.ReadAsArray()
                        data[numpy.isnan(data)] = new_nodata
                        band.SetNoDataValue(new_nodata)
                        band.WriteArray(data)
                    del band, ds

                    # if _debug:
                    #     shutil.copyfile(export_generalized_filename, export_generalized_filename.with_suffix(".convert_nodata.tif"))

                    if where_not_nodata(data, new_nodata).any():
                        # 4) Combine if more than one DB used
                        # @todo if we use export_into_raster the combine would already be done

                        # 5) Binary closing on raster covering expanded area
                        # @todo get this code from xipe
                        # ?? xipe_dev.xipe.raster.interpolate_within_closing()  # -- this looks to have a bunch of extra logic and such we don't want, Casiano suggested _close_interpolation_coverage

                        if b_glen_interp:
                            # @todo -- just call copy of process_csar instead
                            # srs = osr.SpatialReference()
                            # srs.ImportFromEPSG(target_epsg)

                            # FIXME - HACK  we should send a float version of the contributor number since we have a hack
                            #   of storing int32 inside the float32 tiff, but zero is the same in both so we can get away with it for now
                            # numpy.array(0, dtype=numpy.float32).tobytes()
                            # Out[7]: b'\x00\x00\x00\x00'
                            # numpy.array(0, dtype=numpy.int32).tobytes()
                            # Out[8]: b'\x00\x00\x00\x00'

                            generalize(str(export_generalized_filename), closing_dist, output_crs=target_epsg)  # call the generalize function which used to be process_csar script

                        else:
                            # interp_values = fuse.interpolator.bag_interpolator.process.RasterInterpolator()
                            interpolated_dataset = raster_interp.process.RasterInterpolator().interpolate(raster, 'linear', buffer = closing_distance)

                            # binary_mask = fuse.coverage.coverage._close_interpolation_coverage()
                            closed_mask = _close_interpolation_coverage(closing_mask, closing_iterations)
                            closed_mask = numpy.logical_and(closed_mask, raster_coverage)

                            ops = BufferedImageOps(interpolated_dataset)
                            for cnt, (ic, ir, cols, rows, col_buffer_lower, row_buffer_lower, nodata, data) in enumerate(
                                    ops.iterate_gdal(0, 0)):
                                altered_mask = numpy.zeros(data[0].shape, dtype=numpy.bool8)
                                tile = closed_mask[ir:ir + rows, ic:ic + cols]
                                if data[0].shape != tile.shape:
                                    altered_mask[ir:, ic:] = tile
                                else:
                                    altered_mask = tile
                                if not upsampled:
                                    masked_interpolation = numpy.where(altered_mask, data[0], nodata)
                                else:
                                    masked_interpolation = data[0]
                                ops.write_array(masked_interpolation, interpolated_dataset.GetRasterBand(1))

                            # 6) Interpolation only on closed cells that had no bathymetry
                            # @todo get this code from xipe

                            # 7) Export original extents to raster

                            # 8) Create Raster Attribute Table (RAT)
                            # @todo get this code from xipe
                            # @todo contributor_metadata needs to be revised to work on the global indices

                            xipe_v2.hack.make_raster_attributes_modified.make_enc_raster_attr_table_modified

                            xipe.raster.attribute_tables.create_RAT(raster_filename, contributor_metadata, fields=None, compute_counts=True)
                            data = RasterAttributeTableDataset.from_contributor_metadata(contributor_metadata)
                            data.write_to_raster(raster_filename, CONTRIBUTOR_BAND_NAME, fields, compute_counts)
                if not _debug or not(export_contributors_filename).exists():
                    shutil.copyfile(export_generalized_filename, export_contributors_filename)
                    generalized_ds = gdal.Open(str(export_contributors_filename), gdal.GA_Update)

                    print("Consolidate contributor IDs")
                    # print('convert the contributor integer to store inside a float buffer')
                    # a = numpy.array([1, 3, 5, 1234567890], numpy.int32)
                    # f = numpy.frombuffer(a.tobytes(), numpy.float32)
                    # b = numpy.frombuffer(f.tobytes(), numpy.int32)
                    # b
                    # array([1, 3, 5, 1234567890])
                    # Test that float 32 doesn't get truncated weirdly when taken to float64 and back
                    # i32 = numpy.arange(0, 100000000).astype(numpy.int32)
                    # f32 = numpy.frombuffer(i32.tobytes(), numpy.float32)
                    # f32.shape
                    # (100000000,)
                    # f64 = f32.astype(numpy.float64)
                    # i = numpy.frombuffer(f64.astype(numpy.float32).tobytes(), numpy.int32)
                    # numpy.all(i == i32)
                    # True
                    # FIXME - HACK -- encoding integer contributor number as float to fit in the tiff which is float32.
                    #  The raster_data classes should return recarrays (structured arrays) but
                    #  that will require some rework on the slicing and concatenation elsewhere.
                    #  Due to this, the export routine should convert the float back to int and also get fixed later.
                    #  Also have to modify the sorting routine to accommodate the difference
                    #  (translate the ints to floats there too)
                    # update all_simple_records based on the contributors being crazy floats (ints inside float buffers)
                    # and coming back to normal numbers (packed sequence starting from zero)
                    # make a mapping of the original ID number to the 'friendly' ID number
                    # add a column to the attribute table that has the full NBS ID integer as well as the friendly number
                    contrib_band = 3
                    generalized_contrib_band = generalized_ds.GetRasterBand(contrib_band)
                    float_contrib = generalized_contrib_band.ReadAsArray()
                    contribs = numpy.unique(float_contrib)
                    int_contribs = numpy.sort(numpy.frombuffer(contribs.astype(numpy.float32).tobytes(), numpy.int32)).tolist()
                    # leave the 1000000 value (Nan replacement) in the contributor space
                    try:
                        int_contribs.remove(numpy.frombuffer(numpy.array(1000000).astype(numpy.float32).tobytes(), numpy.int32)[0])
                    except ValueError:
                        pass
                    if not int_contribs:  # there is no data in this tile - make a fake contributor list so it continues on.
                        int_contribs = [0]
                    if int_contribs[0] != 0:  # make sure there is a zero entry for the generalization so enumerate below doesn't move a real contributor into the zero spot
                        int_contribs = [0] + int_contribs
                    for inew, iold in enumerate(int_contribs):
                        if inew == 0:  # the generalization has no change from int to float and would cause a keyerror in all_simple_records
                            continue
                        iold_as_float = numpy.frombuffer(numpy.array(iold, dtype=numpy.int32).tobytes(), dtype=numpy.float32)[0]
                        float_contrib[float_contrib==iold_as_float] = inew
                        all_simple_records[inew] = all_simple_records[iold]

                    generalized_contrib_band.WriteArray(float_contrib)
                    pickle.dump(all_simple_records, open(export_contributors_filename.with_suffix(".pickle"), "wb"))
                else:
                    all_simple_records = pickle.load(open(export_contributors_filename.with_suffix(".pickle"), "rb"))

                if not _debug or not(export_cog_filename).exists():
                    print("Make initial Cloud Optimized Geotiff")
                    cogdriver = gdal.GetDriverByName("COG")
                    cog_ds = cogdriver.CreateCopy(str(export_cog_filename), generalized_ds, 0, options=['OVERVIEW_RESAMPLING=CUBIC', 'COMPRESS=LZW', "BIGTIFF=YES", 'OVERVIEWS=IGNORE_EXISTING'])
                    # the overview of the contributor needs 'nearest' (otherwise it interpolates to non-contributor numbers like 15.65)
                    # which is what paletted data uses, while elevation and uncertainty needs "cubic" to display best (see fliers)
                    layers = [cog_ds.GetRasterBand(b + 1).GetDescription().lower() for b in range(cog_ds.RasterCount)]
                    if LayersEnum.CONTRIBUTOR.name.lower() in layers:
                        print("Replace contributor overview in Cloud Optimized Geotiff")
                        band_num = layers.index(LayersEnum.CONTRIBUTOR.name.lower()) + 1
                        band = cog_ds.GetRasterBand(band_num)

                        # Change the overviews on the contributor banc to use "nearest"
                        # bug in gdal <3.3 doesn't compute overviews consistently, work around by doing it twice and copying ourselves
                        major, minor = list(map(int, gdal.__version__.split(".")))[:2]
                        if major > 3 or major == 3 and minor > 2:
                            # in gdal 3.3+ just tell it to regenerate on the contributor band
                            gdal.RegenerateOverviews(band, [band.GetOverview(n) for n in range(band.GetOverviewCount())], 'NEAREST')
                        else:
                            # in gdal 3.2- make a second tif that uses "nearest" then copy those overviews into the original 'cubic' tif replacing the contributor overviews
                            nearest_filename = str(export_cog_filename)+".nearest.tif"
                            cog_ds_near = cogdriver.CreateCopy(nearest_filename, generalized_ds, 0,
                                                               options=['OVERVIEW_RESAMPLING=NEAREST', 'OVERVIEWS=IGNORE_EXISTING', 'COMPRESS=LZW', "BIGTIFF=YES"])
                            for i in range(cog_ds_near.GetRasterBand(band_num).GetOverviewCount()):
                                band.GetOverview(i).WriteArray(cog_ds_near.GetRasterBand(band_num).GetOverview(i).ReadAsArray())
                            del cog_ds_near
                            remove_file(nearest_filename)

                    del band, cog_ds, generalized_ds
                print("Make Raster Attribute Table")
                # Note that we are renumbering the contributors above so this is no longer the nbs_id but just revised indices stored in the contributor layer
                make_raster_attr_table(str(export_cog_filename), all_simple_records)  # make a raster attribute table for the generalized dataset
                # FIXME - remove when Caris is fixed
                #  -- change raster attributes for Caris which is failing on 'metre'
                rat_text = open(rat_filename, 'rb').read()
                new_rat_text = rat_text.replace(b"<UnitType>metre</UnitType>", b"<UnitType>m</UnitType>")
                if b"<UnitType>m</UnitType>" not in new_rat_text:
                    new_rat_text = rat_text.replace(b"</Description>", b"</Description><UnitType>m</UnitType>")
                open(rat_filename, 'wb').write(new_rat_text)

                # create_RAT(dataset)  # make a raster attribute table for the raw dataset

                # remove the score and extracted files
                if not _debug:
                    remove_file(score_filename, allow_permission_fail=True)
                    remove_file(extracted_filename, allow_permission_fail=True)
                    remove_file(export_generalized_filename, allow_permission_fail=True)
                    remove_file(export_contributors_filename, allow_permission_fail=True)
                    remove_file(export_contributors_filename.with_suffix(".pickle"), allow_permission_fail=True)


    except LockNotAcquired:
        print('files in use for ', tile_record[name_index])
        print('skipping to next review tile')
        continue




r"""
In a console -- based on https://www.mail-archive.com/gdal-dev@lists.osgeo.org/msg36586.html 
C:\Git_Repos\Bruty>gdalsrsinfo EPSG:26918+5866 -o WKT1 --single-line
COMPD_CS["NAD83 / UTM zone 18N + MLLW depth",PROJCS["NAD83 / UTM zone 18N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26918"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]
C:\Git_Repos\Bruty>gdalsrsinfo EPSG:26919+5866 -o WKT1 --single-line
COMPD_CS["NAD83 / UTM zone 19N + MLLW depth",PROJCS["NAD83 / UTM zone 19N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-69],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26919"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]

AXIS["Depth",DOWN]
AXIS["gravity-related height",UP]

In a python prompt -- (not working)

from vyperdatum.core import VyperCore, CRS
vc = VyperCore(vdatum_directory=r'C:\vdatum_all_20201203\vdatum')  # only have to set this once
vc.set_input_datum(26918, extents=(398310, 4128708, 398311, 4128709))  # have to provide an area of interest
vc.set_output_datum('mllw')
vc.out_crs.horiz_wkt = CRS.from_epsg(26918).to_wkt()  # this is done automatically, but only for raster for some reason
vc = VyperCore(vdatum_directory=r'C:\PydroTrunk\Miniconda36\NOAA\supplementals\VDatum')  # only have to set this once
vc.set_input_datum(26918, extents=(398310, 4128708, 398311, 4128709))  # have to provide an area of interest
vc.set_output_datum('mllw')
vc.out_crs.horiz_wkt = CRS.from_epsg(26918).to_wkt()  # this is done automatically, but only for raster for some reason
w19 = vc.out_crs.to_compound_wkt()
'COMPOUNDCRS["NAD83 / UTM zone 19N + mllw",PROJCRS["NAD83 / UTM zone 19N",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983",ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4269]],CONVERSION["UTM zone 19N",METHOD["Transverse Mercator",ID["EPSG",9807]],PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],PARAMETER["Longitude of natural origin",-69,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],ID["EPSG",8805]],PARAMETER["False easting",500000,LENGTHUNIT["metre",1],ID["EPSG",8806]],PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],USAGE[SCOPE["Engineering survey, topographic mapping."],AREA["North America - between 72°W and 66°W - onshore and offshore. Canada - Labrador; New Brunswick; Nova Scotia; Nunavut; Quebec. Puerto Rico. United States (USA) - Connecticut; Maine; Massachusetts; New Hampshire; New York (Long Island); Rhode Island; Vermont."],BBOX[14.92,-72,84,-66]],ID["EPSG",26919]],VERTCRS["mllw",VDATUM["mllw"],CS[vertical,1],AXIS["gravity-related height (H)",up],LENGTHUNIT["metre",1],REMARK["regions=[MENHMAgome13_8301,RICTbis22_8301],pipeline=proj=pipeline step proj=vgridshift grids=core\\geoid12b\\g2012bu0.gtx step +inv proj=vgridshift grids=REGION\\tss.gtx step proj=vgridshift grids=REGION\\mllw.gtx"]]]'


From file Glen had done, which seems to use the same EPSG but different axis, see OGC WKT on https://epsg.io/5866
'COMPD_CS["WGS 84 / UTM zone 18N + MLLW height",PROJCS["WGS 84 / UTM zone 18N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32618"]],VERT_CS["MLLW height",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Up",UP]]]'




Used this script to update existing files

w18 = r'COMPD_CS["NAD83 / UTM zone 18N + MLLW depth",PROJCS["NAD83 / UTM zone 18N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26918"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]'
w19 = r'COMPD_CS["NAD83 / UTM zone 19N + MLLW depth",PROJCS["NAD83 / UTM zone 19N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-69],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26919"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]'
sr19 = osr.SpatialReference(w19)
sr18 = osr.SpatialReference(w18)
for root, dirs, files in os.walk(r"D:\test_environments\Barry\data\PBC_exports"):
    for name in files:
        if name.lower().endswith(".tif") and "pbc19" in name.lower():
            ds = gdal.Open(os.path.join(root, name), gdal.GA_Update)
            # print(os.path.join(root, name))
            ds.SetSpatialRef(sr19)
            ds = None
            
for root, dirs, files in os.walk(r"D:\test_environments\Barry\data\PBC_exports"):
    for name in files:
        if name.lower().endswith(".tif") and "pbc18" in name.lower():
            ds = gdal.Open(os.path.join(root, name), gdal.GA_Update)
            # print(os.path.join(root, name))
            ds.SetSpatialRef(sr18)
            ds = None

from nbs.bruty import utils
wkt14 = utils.make_wkt(26914, down_to_up=True)
wkt15 = utils.make_wkt(26915, down_to_up=True)
wkt16 = utils.make_wkt(26916, down_to_up=True)
for root, dirs, files in os.walk(r"V:\bruty_tile_exports"):
    for name in files:
        if name.lower().endswith(".tif"):
            if "pbg14" in name.lower():
                wkt = wkt14
            elif "pbg15" in name.lower():
                wkt = wkt15
            elif "pbg16" in name.lower():
                wkt = wkt16
            else:
                raise Exception("stop, what pbg")
            try:
                ds = gdal.Open(os.path.join(root, name), gdal.GA_Update)
                # print(os.path.join(root, name))
                ds.SetProjection(wkt)
                ds = None
            except PermissionError:
                print("permission denied:", root, name)

'COMPD_CS["NAD83 / UTM zone 14N + MLLW",PROJCS["NAD83 / UTM zone 14N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-99],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26914"]],VERT_CS["MLLW",VERT_DATUM["unknown",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Up",UP]]]'

'COMPD_CS["NAD83 / UTM zone 18N + MLLW depth",PROJCS["NAD83 / UTM zone 18N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-75],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26914"]],VERT_CS["MLLW depth",VERT_DATUM["Mean Lower Low Water",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]'
'COMPD_CS["NAD83 / UTM zone 14N + MLLW depth",PROJCS["NAD83 / UTM zone 14N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-99],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26914"]],VERT_CS["MLLW depth",VERT_DATUM["unknown",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]]]'
'COMPD_CS["NAD83 / UTM zone 14N + MLLW",PROJCS["NAD83 / UTM zone 14N",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-99],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","26914"]],VERT_CS["MLLW",VERT_DATUM["unknown",2005,AUTHORITY["EPSG","1089"]],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Up",UP]]]'

"""

