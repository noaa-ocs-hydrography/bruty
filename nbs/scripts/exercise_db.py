from nbs.bruty.world_raster_database import *

def slow_process():
    # slow_tif = r"C:\Data\\H13222_MB_50cm_MLLW_1of1_interp_1.csar.tif"
    # slow_tif = r"\\nos.noaa\ocs\HSD\Projects\NBS\NBS_Data\PBG_Gulf_UTM14N_MLLW\NOAA_NCEI_OCS\BAGs\Manual\H13222_MB_50cm_MLLW_1of1_interp_1.csar.tif"
    # slow_bag = r"\\nos.noaa\ocs\HSD\Projects\NBS\NBS_Data\PBG_Gulf_UTM14N_MLLW\NOAA_NCEI_OCS\BAGs\Original\H13222\H13222_MB_50cm_MLLW_1of1.bag"
    # slow_points = r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data_Pre-Review\PBG_Gulf_UTM16N_MLLW\GMRT\TOPO-MASK\Manual\PBG_Gulf_UTM16N_MLLW_16m_0_clip.bruty.npy"
    npy = r"C:\Data\nbs\utm14\2016_NCMP_TX_14RPP7984_BareEarth_1mGrid_transformed.bruty.npy"
    gpkg = r"C:\Data\nbs\utm14\2016_NCMP_TX_14RPP7992_BareEarth_1mGrid_transformed.bruty.gpkg"
    bag_name = r"C:\Data\nbs\utm14\F00734_4m_MLLW_Xof4.bag"
    tif = r"C:\Data\nbs\utm14\VT_06_MTT_20150305_CS_B_12_MLT_PARTIAL_4m_interp_3.csar.tif"
    db_path = r"C:\Data\nbs\array_speed_testing"
    if os.path.exists(db_path):
        shutil.rmtree(db_path, onerror=onerr)
    db = WorldDatabase(
        UTMTileBackendExactRes(4, 4, 26914, RasterHistory, DiskHistory, TiffStorage, db_path))
    trans_id = db.add_transaction_group("INSERT", datetime.now())
    time.sleep(3)
    trans_id2 = db.add_transaction_group("INSERT", datetime.now())
    time.sleep(4)
    trans_id3 = db.add_transaction_group("NOT_REMOVE", datetime.now())
    db.insert_survey(gpkg, override_epsg=26914, contrib_id=1, survey_score=1, transaction_id=trans_id)
    db.insert_survey(tif, override_epsg=26914, contrib_id=2, survey_score=1, transaction_id=trans_id)
    db.insert_survey(npy, override_epsg=26914, contrib_id=3, survey_score=1, transaction_id=trans_id2)
    db.insert_survey(bag_name, override_epsg=26914, contrib_id=4, survey_score=1, transaction_id=trans_id3)

def try_sqlite():
    # update a metadata.pickle to sqlite
    for pickle_file in (pathlib.Path(r"E:\bruty_databases\pbg_gulf_utm14n_mllw\wdb_metadata.pickle"),
                        pathlib.Path(r"E:\bruty_databases\pbg_gulf_utm16n_mllw\wdb_metadata.pickle"),
                        ):
        # "C:\data\nbs\pbg14_metadata_not_json.pickle")
        meta = pickle.load(open(pickle_file, 'rb'))
        pickle.dump({'class': meta['class'], 'module': meta['module']}, open(pickle_file.with_suffix(".class"), "wb"))
        metadb = IncludedIds(pickle_file.with_suffix(".sqlite"))
        for pth, rec in list(meta['survey_paths'].items()):
            nbs_id = rec[0]
            record = list(rec[1:])
            metadb[nbs_id] = [pth] + record
        # print(list(metadb.keys()))
        # print(metadb[564364])
        # metasurv = IncludedSurveys(r"C:\data\nbs\pbg14_metadata_not_json.db")
        # print(metasurv[metadb[564364][0]])
        # del metadb[564364]

        metadb = StartedIds(pickle_file.with_suffix(".sqlite"))
        for pth, rec in list(meta['started_paths'].items()):
            nbs_id = rec[0]
            record = list(rec[1:])
            metadb[nbs_id] = [pth] + record[:2]
        # print(list(metadb.keys()))
        # print(metadb[564364])
        # metasurv = StartedSurveys(r"C:\data\nbs\pbg14_metadata_not_json.db")
        # print(metasurv[metadb[564364][0]])
        del metadb

def csar_conversions():
    # test reading the new geopackage, npy and tifs from fuse
    paths = [
        '\\\\nos.noaa\\OCS\\HSD\\Projects\\NBS\\NBS_Data\\PBG_Gulf_UTM14N_MLLW\\USACE\\eHydro_Galveston_CESWG\\Manual\\GI_24_BIL_20210514_CS_4m_interp.csar.tif',
        # r"C:\Data\nbs\geopackage_samples\H12425_MB_50cm_MLLW_6of17.tif",
        # r"C:\Data\nbs\geopackage_samples\H12425_MB_50cm_MLLW_7of17.tif",
        # r"C:\Data\nbs\geopackage_samples\H12425_MB_50cm_MLLW_12of17.tif",
        # r"C:\Data\nbs\geopackage_samples\2020_NCMP_PostSally_AL_16RDU3447_BareEarth_1mGrid_transformed.gpkg",
        # r"C:\Data\nbs\geopackage_samples\2020_NCMP_PostSally_AL_16RDU3447_BareEarth_1mGrid_transformed.npy",
        # r"C:\Data\nbs\geopackage_samples\2020_NCMP_PostSally_AL_16RDU3546_BareEarth_1mGrid_transformed.gpkg",
        # r"C:\Data\nbs\geopackage_samples\2020_NCMP_PostSally_AL_16RDU3546_BareEarth_1mGrid_transformed.npy",
        # r"C:\Data\nbs\geopackage_samples\H11835_MB_50cm_MLLW_1of5.tif",
        # r"C:\Data\nbs\geopackage_samples\H11835_VB_2m_MLLW_5of5.tif",
        # r"C:\Data\nbs\geopackage_samples\H13133_MB_1m_MLLW_3of3.tif",
        ]
    resx = resy = 32
    # NAD823 zone 19 = 26919.  WGS84 would be 32619
    epsg = 26914
    # use this to align the database to something else (like caris for testing)
    offset_x = 0
    offset_y = 0

    db_path = r"C:\Data\nbs\test_remove_reinsert\utm16_new_csar_exports"
    db = WorldDatabase(
        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                               zoom_level=10))
    for contrib_id, survey_path in enumerate(paths):
        db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db_path = r"C:\Data\nbs\test_remove_reinsert\utm16_npy_exports"
    db = WorldDatabase(
        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                               zoom_level=10))
    for contrib_id, survey_path in enumerate(paths):
        if ".npy" in survey_path.lower():
            db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db_path = r"C:\Data\nbs\test_remove_reinsert\utm16_gpkg_exports"
    db = WorldDatabase(
        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                               zoom_level=10))
    for contrib_id, survey_path in enumerate(paths):
        if ".gpkg" in survey_path.lower():
            db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db_path = r"C:\Data\nbs\test_remove_reinsert\utm16_tif_exports"
    db = WorldDatabase(
        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                               zoom_level=10))
    for contrib_id, survey_path in enumerate(paths):
        if ".tif" in survey_path.lower():
            db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)

def try_removal():
    # test removing real data
    paths = [
        r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BPS\Processed\H09170.csar",
        r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BPS\Processed\H10350.csar",
        r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BPS\Processed\H06443.csar",
        r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BAGs\Processed\H12137_MB_VR_MLLW.csar",
        r"\\nos.noaa\OCS\HSD\Projects\NBS\NBS_Data\PBC_Northeast_UTM19N_MLLW\NOAA_NCEI_OCS\BPS\Manual\H10795.csar",
    ]
    out_paths = [
        (r"C:\Data\nbs\test_remove_reinsert\H09170.csar.csv", 1),
        (r"C:\Data\nbs\test_remove_reinsert\H10350.csar.csv", 2),
        (r"C:\Data\nbs\test_remove_reinsert\H06443.csar.csv", 3),
        (r"C:\Data\nbs\test_remove_reinsert\H12137_MB_VR_MLLW.csar.csv.npy", 4),
        (r"C:\Data\nbs\test_remove_reinsert\H10795.csar.csv", 5),
    ]
    resx = resy = 64
    # NAD823 zone 19 = 26919.  WGS84 would be 32619
    epsg = 26919
    # use this to align the database to something else (like caris for testing)
    offset_x = 0
    offset_y = 0

    # db_path = r"C:\Data\nbs\test_remove_reinsert\utm19_removals_64m"
    # db = WorldDatabase(
    #     UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y, zoom_level=10))
    # for survey_path, contrib_id in out_paths:
    #     db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)

    db_path = r"C:\Data\nbs\test_remove_reinsert\utm19_removals_64m_remove_4"
    if os.path.exists(db_path):
        shutil.rmtree(db_path, onerror=onerr)
    try:
        shutil.copytree(db_path + " - Copy", db_path)
        db = WorldDatabase(
            UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                                   zoom_level=10))
    except FileNotFoundError:

        db = WorldDatabase(
            UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                                   zoom_level=10))
        for survey_path, contrib_id in out_paths:
            db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db.remove_and_recompute(4)

    out_paths.reverse()
    # db_path = r"C:\Data\nbs\test_remove_reinsert\utm19_removals_64m_reverse"
    # db = WorldDatabase(
    #     UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y, zoom_level=10))
    # for survey_path, contrib_id in out_paths:
    #     db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db_path = r"C:\Data\nbs\test_remove_reinsert\utm19_removals_64m_reverse_remove_4"
    db = WorldDatabase(
        UTMTileBackendExactRes(resx, resy, epsg, RasterHistory, DiskHistory, TiffStorage, db_path, offset_x=offset_x, offset_y=offset_y,
                               zoom_level=10))
    for survey_path, contrib_id in out_paths:
        db.insert_survey(survey_path, override_epsg=epsg, contrib_id=contrib_id, survey_score=contrib_id)
    db.remove_and_recompute(4)

def try_customarea():

    data_dir = pathlib.Path(r"G:\Data\NBS\H11305_for_Bruty")
    # orig_db = CustomArea(26916, 395813.2, 3350563.98, 406818.2, 3343878.98, 4, 4, data_dir.joinpath('bruty'))
    new_db = CustomArea(None, 395813.20000000007, 3350563.9800000004, 406818.20000000007, 3343878.9800000004, 4, 4,
                        data_dir.joinpath('bruty_debug_center'))
    # use depth band for uncertainty since it's not in upsample data
    new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\1of3.tif", 0, uncert_band=1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\2of3.tif", 0, uncert_band=1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\3of3.tif", 0, uncert_band=1, override_epsg=None)
    new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_1of3.bag", 1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_2of3.bag", 1, override_epsg=None)
    # new_db.insert_survey_gdal(r"G:\Data\NBS\H11305_for_Bruty\H11305_VB_5m_MLLW_3of3.bag", 1, override_epsg=None)
    new_db.export(r"G:\Data\NBS\H11305_for_Bruty\combine_new_centers.tif")

def mississippi():
    fname = r"G:\Data\NBS\Speed_test\H11045_VB_4m_MLLW_2of2.bag"
    ds = gdal.Open(fname)
    x1, resx, dxy, y1, dyx, resy = ds.GetGeoTransform()
    numx = ds.RasterXSize
    numy = ds.RasterYSize
    epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
    epsg = 26918
    ds = None
    # db = WorldDatabase(UTMTileBackendExactRes(4, 4, epsg, RasterHistory, DiskHistory, TiffStorage,
    #                                     r"G:\Data\NBS\Speed_test\test_db_world"))
    db = CustomArea(epsg, x1, y1, x1 + (numx + 1) * resx, y1 + (numy + 1) * resy, 4, 4, r"G:\Data\NBS\Speed_test\test_cust4")
    db.insert_survey_gdal(fname, override_epsg=epsg)
    db.export(r"G:\Data\NBS\Speed_test\test_cust4\export.tif")
    raise Exception("Done")

    # from nbs.bruty.history import MemoryHistory
    # from nbs.bruty.raster_data import MemoryStorage, RasterDelta, RasterData, LayersEnum, arrays_match
    from nbs.bruty.utils import save_soundings_from_image

    # from tests.test_data import master_data, data_dir

    # use_dir = data_dir.joinpath('tile4_vr_utm_db')
    # db = WorldDatabase(UTMTileBackend(26919, RasterHistory, DiskHistory, TiffStorage, use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    # db.export_area_old(use_dir.joinpath("export_tile_old.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)
    # db.export_area(use_dir.joinpath("export_tile_new.tif"), 255153.28, 4515411.86, 325721.04, 4591064.20, 8)

    build_mississippi = True
    export_mississippi = False
    process_utm_15 = True
    output_res = (4, 4)  # desired output size in meters
    data_dir = pathlib.Path(r'G:\Data\NBS\Mississipi')
    if process_utm_15:
        export_dir = data_dir.joinpath("UTM15")
        epsg = 26915
        max_lon = -90
        min_lon = -96
        max_lat = 35
        min_lat = 0
        use_dir = data_dir.joinpath('vrbag_utm15_debug_db')

        data_files = [(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13194_MB_VR_LWRP.bag", 92),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13193_MB_VR_LWRP.bag", 100),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", 94),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13188_MB_VR_LWRP.bag", 95),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13189_MB_VR_LWRP.bag", 96),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag", 97),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13191_MB_VR_LWRP.bag", 98),
                      (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag", 99),
                      # (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag.resampled_4m.uncert.tif", 77),
                      # (r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag.resampled_4m.uncert.tif", 79),
                      ]
        resamples = []
        # [r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag",
        #            r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13192_MB_VR_LWRP.bag",]
        for vr_path in resamples:
            resampled_path = vr_path + ".resampled_4m.tif"
            bag.VRBag_to_TIF(vr_path, resampled_path, 4, use_blocks=False)
            resampled_with_uncertainty = resampled_path = resampled_path[:-4] + ".uncert.tif"
            add_uncertainty_layer(resampled_path, resampled_with_uncertainty)
            data_files.append(resampled_with_uncertainty)
    else:
        export_dir = data_dir.joinpath("UTM16")
        epsg = 26916
        max_lon = -84
        min_lon = -90
        max_lat = 35
        min_lat = 0
        use_dir = data_dir.joinpath('vrbag_utm16_debug_db')
        data_files = [(r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13195_MB_VR_LWRP.bag", 93),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13196_MB_VR_LWRP.bag", 91),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13193_MB_VR_LWRP.bag", 100),
                      (r"G:\Data\NBS\Mississipi\UTM16\NCEI\H13194_MB_VR_LWRP.bag", 92),
                      ]

    if build_mississippi:
        if os.path.exists(use_dir):
            shutil.rmtree(use_dir, onerror=onerr)

    db = WorldDatabase(UTMTileBackendExactRes(*output_res, epsg, RasterHistory, DiskHistory, TiffStorage,
                                              use_dir))  # NAD823 zone 19.  WGS84 would be 32619
    if 0:  # find a specific point in the tiling database
        y, x = 30.120484, -91.030685
        px, py = crs_transform.transform(x, y)
        tile_index_x, tile_index_y = db.db.tile_scheme.xy_to_tile_index(px, py)

    if build_mississippi:

        for data_file, score in data_files:
            # bag_file = directory.joinpath(directory.name + "_MB_VR_LWRP.bag")
            if _debug:
                if 'H13190' not in data_file:
                    print("Skipped for debugging", data_file)
                    continue
            if 'H13194' in data_file:  # this file is encoded in UTM16 even in the UTM15 area
                override_epsg = 26916
            elif 'H13193' in data_file:  # this file is encoded in UTM15 even in the UTM16 area
                override_epsg = 26915
            else:
                override_epsg = epsg
            # db.insert_survey_gdal(bag_file, override_epsg=epsg)  # single res
            if str(data_file)[-4:] in (".bag",):
                db.insert_survey_vr(data_file, survey_score=score, override_epsg=override_epsg)
            elif str(data_file)[-4:] in ("tiff", ".tif"):
                db.insert_survey_gdal(data_file, survey_score=score)

    if export_mississippi:
        area_shape_fname = r"G:\Data\NBS\Support_Files\MCD_Bands\Band5\Band5_V6.shp"
        ds = gdal.OpenEx(area_shape_fname)
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
            # crop to the area around Mississippi
            if cx > min_lon and cx < max_lon and cy > min_lat and cy < max_lat:
                cell_name = feat.GetField(cell_field)
                if _debug:

                    ##
                    ## vertical stripes in lat/lon
                    ## "US5MSYAF" for example
                    # if cell_name not in ("US5MSYAF",):  # , 'US5MSYAD'
                    #     continue

                    ## @fixme  There is a resolution issue at ,
                    ## where the raw VR is at 4.2m which leaves stripes at 4m export so need to add
                    ## an upsampled dataset to fill the area (with lower score so it doesn't overwrite the VR itself)
                    if cell_name not in ('US5BPGBD',):  # 'US5BPGCD'):
                        continue

                    # @fixme  missing some data in US5PLQII, US5PLQMB  US5MSYAE -- more upsampling needed?

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

                export_path = export_dir.joinpath(cell_name + ".tif")
                cnt, exported_dataset = db.export_area(export_path, minx, miny, maxx, maxy, (dx + dx * .1, dy + dy * .1), target_epsg=export_epsg)

                # export_path = export_dir.joinpath(cell_name + ".bag")
                # bag_options = [key + "=" + val for key, val in bag_options_dict.items()]
                # cnt2, ex_ds = db.export_area(export_path, minx, miny, maxx, maxy, (dx+dx*.1, dy+dy*.1), target_epsg=export_epsg,
                #                       driver='BAG', gdal_options=bag_options)

                if cnt > 0:
                    # output in native UTM -- Since the coordinates "twist" we need to check all four corners,
                    # not just lower left and upper right
                    x1, y1, x2, y2 = transform_rect(minx, miny, maxx, maxy, crs_transform.transform)
                    cnt, utm_dataset = db.export_area(export_dir.joinpath(cell_name + "_utm.tif"), x1, y1, x2, y2, output_res)
                else:
                    exported_dataset = None  # close the gdal file
                    os.remove(export_path)
                os.remove(export_path.with_suffix(".score.tif"))

    test_soundings = False
    if test_soundings:
        soundings_files = [pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20210219_source.tiff"),
                           pathlib.Path(r"C:\Data\nbs\PBC19_Tile4_surveys\soundings\Tile4_4m_20201118_source.tiff"),
                           ]
        for soundings_file in soundings_files:
            ds = gdal.Open(str(soundings_file))
            # epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
            xform = ds.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
            d_val = ds.GetRasterBand(1)
            col_size = d_val.XSize
            row_size = d_val.YSize
            del d_val, ds
            x1, y1 = affine(0, 0, *xform)
            x2, y2 = affine(row_size, col_size, *xform)
            res = 50
            res_x = res
            res_y = res
            # move the minimum to an origin based on the resolution so future exports would match
            if x1 < x2:
                x1 -= x1 % res_x
            else:
                x2 -= x2 % res_x

            if y1 < y2:
                y1 -= y1 % res_y
            else:
                y2 -= y2 % res_y

            #  note: there is an issue where the database image and export image are written in reverse Y direction
            #  because of this the first position for one is top left and bottom left for the other.
            #  when converting the coordinate of the cell it basically ends up shifting by one
            #  image = (273250.0, 50.0, 0.0, 4586700.0, 0.0, -50.0)  db = (273250.0, 50, 0, 4552600.0, 0, 50)
            #  fixed by using cell centers rather than corners.
            #  Same problem could happen of course if the centers are the edges of the export tiff
            # db = CustomArea(26919, x1, y1, x2, y2, res_x, res_y, soundings_file.parent.joinpath('debug'))  # NAD823 zone 19.  WGS84 would be 32619
            # db.insert_survey_gdal(str(soundings_file))
            # db.export_area_new(str(soundings_file.parent.joinpath("output_soundings_debug5.tiff")), x1, y1, x2, y2, (res_x, res_y), )
            save_soundings_from_image(soundings_file, str(soundings_file) + "_3.gpkg", 50)

# test positions -- H13190, US5GPGBD, Mississipi\vrbag_utm15_full_db\4615\3227\_000001_.tif, Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP_resampled.tif
# same approx position
# 690134.03 (m), 3333177.81 (m)  is 41.7 in the H13190
# 690134.03 (m), 3333177.81 (m)  is 42.4 in the resampled
# 690133.98 (m), 3333178.01 (m)  is 42.3 in the \4615\3227\000001.tif
# 690133.60 (m), 3333177.74 (m)  is 42.3 in the US5GPGBD

# seems to be the same Z value of 41.7
# 690134.03 (m), 3333177.81 (m)  H13190
# 690138.14 (m), 3333177.79 (m)  resample  (right (east) one column)
# 690129.99 (m), 3333173.99 (m)  \4615\3227\000001.tif  (down+left (south west) one row+col)
# 690129.62 (m), 3333173.76 (m)  US5GPGBD  (down+left (south west) one row+col)

# from importlib import reload
# import HSTB.shared.gridded_coords
# bag.VRBag_to_TIF(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_MB_VR_LWRP.bag", r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_resample.tif", 4.105774879455566, bag.MEAN, nodata=1000000.)

# index2d = numpy.array([(655, 265)], dtype=numpy.int32)

# >>> print('x', refinement_llx + 9 * resolution_x + resolution_x / 2.0, 'y',refinement_lly + 8 * resolution_y + resolution_y / 2.0)
# x 690134.0489868548 y 3333177.797961975
# >>> print("x (cols)",xstarts[9],":", xends[9], "y (rows)",ystarts[8],":", yends[8])
# x (cols) 690131.9960994151 : 690136.1018732946 y (rows) 3333175.7450745353 : 3333179.850848415
# >>> print("rows",row_start_indices[8],":",row_end_indices[8], "cols",col_start_indices[9],":", col_end_indices[9])
# rows 4052 : 4053 cols 2926 : 2927
# >>> print('starts',HSTB.shared.gridded_coords.affine(row_start_indices[8], col_start_indices[9], *ds_val.GetGeoTransform()), ',  ends',HSTB.shared.gridded_coords.affine(row_end_indices[8], col_end_indices[9], *ds_val.GetGeoTransform()))
# starts (690131.995748028, 3333183.9557557716) ,  ends (690136.1015229075, 3333179.849980892)
# >>> ds_val.GetGeoTransform(), sr_grid.geotransform
# ((678118.498450741,  4.105774879455566,  0.0,  3349820.5555673256,  0.0,  -4.105774879455566),
#  (678118.498450741,  4.105774879455566,  0,  3303552.578450741,  0,  4.105774879455566))

# ds = gdal.Open(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13190_resample4.tif")
# b = ds.GetRasterBand(1)
# dep = b.ReadAsArray()
# b.GetNoDataValue()
# (dep!=0.0).any()
r"""  # script to add transaction records to original bruty databases
bruty_paths = [
r"E:\bruty_databases\pbg_gulf_utm14n_mllw",
r"E:\bruty_databases\pbg_gulf_utm14n_mllw_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm14n_mllw_prereview",
r"E:\bruty_databases\pbg_gulf_utm14n_mllw_prereview_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm14n_mllw_sensitive",
r"E:\bruty_databases\pbg_gulf_utm14n_mllw_sensitive_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw_prereview",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw_prereview_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw_sensitive",
r"E:\bruty_databases\pbg_gulf_utm15n_mllw_sensitive_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw_prereview",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw_prereview_not_for_navigation",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw_sensitive",
r"E:\bruty_databases\pbg_gulf_utm16n_mllw_sensitive_not_for_navigation"]

for pth in bruty_paths:
    db = world_raster_database.WorldDatabase.open(pth)
    db.transaction_groups.add_oid_record(("INSERT", datetime.datetime(2021, 9, 25, 0, 0, 0)))
    inc = db.included_ids.cur.execute("UPDATE included set (transaction_id)=(1)").rowcount
    start = db.included_ids.cur.execute("UPDATE started set (transaction_id)=(1)").rowcount
    db.included_ids.conn.commit()
    print(pth, inc, start)
"""

if __name__ == "__main__":
    slow_process()