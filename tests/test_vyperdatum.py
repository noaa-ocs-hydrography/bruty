import pathlib

from osgeo import gdal, osr, ogr
import pyproj
from pyproj import Transformer, CRS
from vyperdatum_register.build_db_from_NOAA import setup_pyproj

setup_pyproj()
proj_data_path = pathlib.Path(pyproj.datadir.get_data_dir().split(';')[0])
path_to_nvdr_json = proj_data_path.parent.joinpath("NOAA_NOS_Vertical_Datum_Register.json")
path_to_mdghgs_json = proj_data_path.parent.joinpath("model_datum_geodetic_height_grids.json")
nwld_url = "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/Test-and-Evaluation/NOAA_NOS_Vertical_Datum_Register/national_model_grids"
geoids_url = "https://noaa-ocs-nationalbathymetry-pds.s3.amazonaws.com/Test-and-Evaluation/NOAA_NOS_Vertical_Datum_Register/additional_geoids"
geoids_outside_proj_cdn = ("GEOID2022",)

def make_height_wkt(horz_epsg, datum):
    datum = datum.upper()
    if datum in ("MLLW",):
        vert_epsg = 5866
        datum_str = "MLLW"
    elif datum in ("IGLD", "IGLD85", "IGLD85LWD"):
        vert_epsg = 5609
        datum_str = "IGLD"
    else:
        raise ValueError(f"datum {datum} not recognized or supported")
    wkt = make_wkt(horz_epsg, vert_epsg)
    # 5866 with GDAL will not accept the Up axis, so have to strip the 5866 epsg authority
    down_string = f'AXIS["Depth",DOWN],AUTHORITY["EPSG","{vert_epsg}"]'
    if down_string in wkt:
        wkt = wkt.replace(down_string, 'AXIS["gravity-related height",UP]').replace(f"{datum_str} depth", f"{datum_str}")

        pass
    return wkt


def make_wkt(horz_epsg, vert_epsg):
    """ Calls gdalsrsinfo with the epsgs supplied.  MLLW (5866) is the default vertical.
    down_to_up flag will change AXIS["Depth",DOWN] to AXIS["gravity-related height",UP]
    see:  https://docs.opengeospatial.org/is/18-010r7/18-010r7.html#47
    """
    # cmd = f"gdalsrsinfo EPSG:{horz_epsg}+{vert_epsg} -o WKT1 --single-line"
    # srs_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # srs_process.wait()
    # wkt_old = srs_process.stdout.read().decode().strip()
    # stderr = srs_process.stderr.read().decode().strip()

    srs = osr.SpatialReference()
    srs.SetFromUserInput(f"EPSG:{horz_epsg}+{vert_epsg}")
    # srs.ExportToWkt(["FORMAT=WKT2"])
    wkt = srs.ExportToWkt(["FORMAT=WKT1"])
    if "COMPD_CS" not in wkt:
        raise Exception("compound CRS not found")

    return wkt

def output_wkt(self, nwlf_code_3d, nwlf_code_2d, vcrs_auth, vcrs_code, sep_wkt_ellipsoid_path, sep_wkt_chartdatum_path, utm_ccrs=None, include_wkt_ellipsoid=True):
    if utm_ccrs:
        if include_wkt_ellipsoid:
            sep_wkt_ellipsoid = pyproj.CRS(f"{utm_ccrs}").to_3d().to_wkt(pretty=True)
        sep_wkt_chartdatum = pyproj.CRS(f"{utm_ccrs}+{vcrs_auth}:{vcrs_code}").to_wkt(pretty=True)
    else:
        if include_wkt_ellipsoid:
            sep_wkt_ellipsoid = pyproj.CRS(f"EPSG:{nwlf_code_3d}").to_wkt(pretty=True)
        sep_wkt_chartdatum = pyproj.CRS(f"EPSG:{nwlf_code_2d}+{vcrs_auth}:{vcrs_code}").to_wkt(pretty=True)
    if include_wkt_ellipsoid:
        with open(sep_wkt_ellipsoid_path, 'w') as wkt_ell_fh:
            wkt_ell_fh.write(sep_wkt_ellipsoid)
    with open(sep_wkt_chartdatum_path, 'w') as wkt_cd_fh:
        wkt_cd_fh.write(sep_wkt_chartdatum)

def run(self):
    gui = self.dialog.get_control_values()
    reference_frame = gui['reference_frame']
    nwlf, coordinate_epoch, nwlf_code_3d, nwlf_code_2d = ("NAD83(2011)", 2010.0, 6319, 6318) if "NAD83" in reference_frame else (
    "ITRF2020", 2020.0, 9989, 9990)
    nwld_coverage_crs = osr.SpatialReference()
    _ = nwld_coverage_crs.ImportFromEPSG(nwlf_code_2d)
    geodesic = pyproj.Geod(ellps='GRS80')
    do_utm = gui['bag_utm_wkt']
    # do_raster, do_vector = gui['raster_output'], gui['vector_output']
    pixel_dm = gui['raster_resolution']
    project_unique_id = gui['project_unique_id']  # <Project Unique ID>_WKT_[ChartDatum, Ellipsoid]_<#>of<#>.txt

    do_ortho, geoid_undulation = gui['orthometric_sep'], gui['geoid_undulation']
    if geoid_undulation:
        nwld, nwld_hydroid = None, None
        # note: auth and coding defined per ortho_sep_ccrs, after roundup_parameters()
        # but otherwise consistent with nwlf_code_3d and nwlf_code_2d
    else:
        nwld_release = gui['nwld_release']
        vcrs_auth, (vcrs_name, vcrs_code) = "NOAA", self.dialog.get_noaa_vcrs_coding(nwld_release, gui["vertical_datum"])
        nwld = vcrs_name.split()[0]
        nwld_hydroid = f"us_noaa_nos_{nwld}-{nwlf}_{coordinate_epoch}_({nwld_release}).tif"

    output_path, messages, aws_override, geoid_vdatum, geoid_sigma, utm_ccrs, sep_param = self.roundup_parameters(gui, nwld_hydroid, nwlf,
                                                                                                                  nwlf_code_3d, nwld,
                                                                                                                  nwld_coverage_crs)
    # sort messages so that any of type "Error" precede any of type "Warning"
    for message_issue, message_type in sorted(messages, key=lambda m: m[1]):
        print(5 * "*", message_issue, 5 * "*")
        self.show_message_box.emit(message_issue, message_type)
        if self._interrupt:
            break

    if not self._interrupt:
        sep_params = (
        "sep_locale_path_stem", "sep_cutline_ds_name", "sep_cutline_geometry", "sep_lyr_bounds", "sep_ccrs", "sep_relation", "sep_file_relation",
        "sep_band_relation", "grid_transform_steps")
        sep_locale_path_stem, sep_cutline_ds_name, sep_cutline_geometry, sep_lyr_bounds, sep_ccrs, sep_relation, sep_file_relation, sep_band_relation, grid_transform_steps = [
            sep_param[k] for k in sep_params]
        if geoid_undulation:
            # resolve auth and coding defined per ortho_sep_ccrs (and consistent with nwlf_code_[3d,2d])
            geoid_hcrs, geoid_vcrs = sep_ccrs.split('+')
            vcrs_auth, vcrs_code = geoid_vcrs.split(':')
            sep_label = sep_relation
            hydroid_wkt_type = "Geoid"
            hydroid_address = (None, f"{nwlf} {coordinate_epoch}", geoid_vdatum)
        else:
            sep_label = f"{sep_relation}: {nwld_release}"
            hydroid_wkt_type = "ChartDatum"
            hydroid_address = (nwld_release, f"{nwlf} {coordinate_epoch}", nwld)

        if aws_override:
            self.dialog.via_aws_cloud.setChecked(True)

        sep_cutline_ds_path = pathlib.Path(sep_cutline_ds_name)
        lon_min, lat_min, lon_max, lat_max = sep_lyr_bounds
        lon_mean, lat_mean = np.mean((lon_min, lon_max)), np.mean((lat_min, lat_max))

        if project_unique_id:
            self.sep_vrt_path = sep_vrt_path = output_path.joinpath(f"{project_unique_id}_{sep_file_relation}.vrt")
            sep_wkt_base_path = output_path.joinpath(f"{project_unique_id}_WKT")
        else:
            self.sep_vrt_path = sep_vrt_path = output_path.joinpath(f"{sep_locale_path_stem}_{sep_file_relation}.vrt")
            sep_wkt_base_path = output_path.joinpath(f"{sep_locale_path_stem}_WKT")

        sep_hydroid_path = sep_vrt_path.with_suffix(".tif")
        sep_wkt_ellipsoid_path = f"{sep_wkt_base_path}_Ellipsoid_1of1.txt"
        sep_wkt_chartdatum_path = f"{sep_wkt_base_path}_{hydroid_wkt_type}_{hydroid_address[-1]}_1of1.txt"
        sep_uncertainty_path = sep_vrt_path.with_name(f"{sep_vrt_path.stem}_Uncertainty.txt")

        self.sep_coverage_mask = sep_coverage_mask = output_path.joinpath(f"{sep_cutline_ds_path.stem}_MASK.tif")
        pixel_size_lon = np.fabs(geodesic.fwd(lon_mean, lat_mean, 270, pixel_dm / 2.)[0] - geodesic.fwd(lon_mean, lat_mean, 90, pixel_dm / 2.)[0])
        if pixel_size_lon > 180:  # i.e., phase wrap about 180/-180
            pixel_size_lon = 360 % pixel_size_lon

        pixel_size_lat = geodesic.fwd(lon_mean, lat_mean, 0, pixel_dm / 2.)[1] - geodesic.fwd(lon_mean, lat_mean, 180, pixel_dm / 2.)[1]

        # extend raster dimensions by 1 pixel (~half a pixel on each side)
        raster_nodata = -32768
        raster_size_lon = int(np.ceil((lon_max - lon_min) / pixel_size_lon)) + 1
        lon_min, lon_max = lon_min - pixel_size_lon / 2., lon_max + pixel_size_lon / 2.
        raster_size_lat = int(np.ceil((lat_max - lat_min) / pixel_size_lat)) + 1
        lat_min, lat_max = lat_min - pixel_size_lat / 2., lat_max + pixel_size_lat / 2.
        tmp_ds.SetMetadataItem('AREA_OR_POINT', 'Point')
        tmp_ds.SetMetadataItem('TYPE', 'VERTICAL_OFFSET_GEOGRAPHIC_TO_VERTICAL')
        tmp_ds.SetMetadataItem('TIFFTAG_COPYRIGHT',
                               f"Derived from work by NOAA, {sep_label}, compiled by Jack L. Riley. Creative Commons Zero 1.0 Universal Public Domain Dedication (CC0 1.0)")
        tmp_ds.SetMetadataItem('TIFFTAG_DATETIME', datetime.now().strftime('%Y:%m:%d %H:%M:%S'))
        tmp_ds.SetMetadataItem('terms_of_use_agreement_uri', "https://vdatum.noaa.gov/download_agreement.php")
        tmp_ds_band = tmp_ds.GetRasterBand(1)
        tmp_ds_band.SetUnitType('m')
        tmp_ds_band.SetNoDataValue(raster_nodata)
        tmp_ds_band.WriteArray(np.zeros((raster_size_lat, raster_size_lon), np.float32))
        tmp_ds.GetRasterBand(1).ComputeStatistics(0)
        tmp_crs = osr.SpatialReference()
        tmp_crs.SetFromUserInput(sep_ccrs)
        tmp_crs.SetCoordinateEpoch(coordinate_epoch)
        tmp_ds.SetSpatialRef(tmp_crs)
        tmp_ds.SetGeoTransform((lon_min, pixel_size_lon, 0, lat_max, 0, -pixel_size_lat))
        tmp_ds.FlushCache()

        kwargs = {
            'format': 'GTiff',
            'outputType': gdal.GDT_Float32,
            'creationOptions': ["COMPRESS=DEFLATE", ],
            'warpOptions': ["CUTLINE_ALL_TOUCHED=TRUE", ],
            'cutlineDSName': sep_cutline_ds_name,
            'cropToCutline': True,
            'callback': self.progress_callback,
            'callback_data': 1,
        }

        ds = gdal.Warp(sep_coverage_mask, tmp_ds, **kwargs)
        ds = None
        tmp_ds = None

        concatenated_transform = f"""\
            +proj=pipeline \
            +step +proj=axisswap +order=2,1 \
            +step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m \
            {grid_transform_steps} \
            +step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m \
            +step +proj=axisswap +order=2,1"""

        # remove whitespace formatting used in proj template strings, as will use it in GeoTIFF metadata
        concatenated_transform = re.sub(r'\s{2,}', ' ', concatenated_transform).strip()
        ds = gdal.Warp(sep_vrt_path, sep_coverage_mask, format="vrt", outputType=gdal.gdalconst.GDT_Float32,
                       warpOptions=["APPLY_VERTICAL_SHIFT=YES", "SAMPLE_GRID=YES", "SAMPLE_STEPS=ALL"], errorThreshold=0,
                       outputBounds=(lon_min, lat_min, lon_max, lat_max), xRes=pixel_size_lon, yRes=pixel_size_lat,
                       coordinateOperation=concatenated_transform,
                       callback=self.progress_callback, callback_data=2)
        ds.SetMetadataItem('TIFFTAG_IMAGEDESCRIPTION', concatenated_transform)
        ds_band = ds.GetRasterBand(1)

        ds_band_nodata = ds_band.GetNoDataValue()
        ds_band_data = ds_band.ReadAsArray()
        no_sep_coverage = (ds_band_data == ds_band_nodata).all()

        if no_sep_coverage:
            print("no_sep_coverage")
            message_issue = "No VDatum model coverage exists for the selected SEP locale.\nYou may report the shortfall to the NOAA VDatum support email:\nvdatum.info@noaa.gov"
            message_type = "Error"
            print(5 * "*", message_issue, 5 * "*")
            self.show_message_box.emit(message_issue, message_type)
        else:
            # NOTE:
            # CARIS SEP Model defaults to a depth orientation, whereas the GeoTIFF SEP data file
            # and band naming is per +up elevations. Such sign-flipping by CARIS nonetheless
            # results in a correct offset applied by HIPS in the Geodetic Referencing Process.
            ds_band.SetDescription(sep_band_relation)
            ds_band.SetUnitType('m')
            ds_band.ComputeStatistics(0)

            # gdal exceptions apply to CE_Error and CE_Fatal messages only
            # in lieu of try/except, trap warnings registered via push/pop
            # of custom_error_handler() and then check for interrupt message
            gdal.PushErrorHandler(self.custom_error_handler)
            sep_ds = gdal.Translate(sep_hydroid_path, ds, format="GTiff", outputType=gdal.GDT_Float32,
                                    creationOptions=["COMPRESS=DEFLATE", "TILED=YES"], callback=self.progress_callback, callback_data=3)
            if self._interrupt_message:
                self.show_message_box.emit(self._interrupt_message, "Error")
            gdal.PopErrorHandler()
            sep_ds = None
            ds = None

            # may as well proceed with sigma and WKT output, in case of gdal "error" false alarm
            if geoid_undulation:
                sep_uncertainty = geoid_sigma
            else:
                weighting, sep_uncertainty = self.dialog.calculate_sep_locale_weighted_uncertainty(sep_cutline_geometry, hydroid_address,
                                                                                                   do_ortho)
            with open(sep_uncertainty_path, 'w') as sep_unc_fh:
                sep_unc_fh.write(f"{round(sep_uncertainty)} cm, 1-sigma")

            if do_utm:
                self.output_wkt(nwlf_code_3d, nwlf_code_2d, vcrs_auth, vcrs_code, sep_wkt_ellipsoid_path, sep_wkt_chartdatum_path, utm_ccrs)
            else:
                self.output_wkt(nwlf_code_3d, nwlf_code_2d, vcrs_auth, vcrs_code, sep_wkt_ellipsoid_path, sep_wkt_chartdatum_path)


source_crs = pyproj.CRS.from_wkt(source_layer.GetSpatialRef().ExportToWkt(["FORMAT=WKT2_2018"]))
target_crs = pyproj.CRS.from_wkt(desired_crs.ExportToWkt(["FORMAT=WKT2_2018"]))
transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
