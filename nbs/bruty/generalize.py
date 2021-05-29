import logging
import os
import pathlib
import shutil
import sys
from datetime import datetime
from tempfile import TemporaryDirectory
from osgeo import gdal
from glob import glob
import csv
import configparser

import numpy
from scipy.ndimage import distance_transform_edt as edt

from transform_dev.datum_transform.reproject import parse_crs, split_compound_crs
from xipe_dev.xipe import get_logger, iter_configs, log_config, set_stream_logging
from xipe_dev.xipe.csar.convert_csar import csar_to_raster
from xipe_dev.xipe.raster import CONTRIBUTOR_BAND_NAME, ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME, \
    raster_band_name_index, reproject_raster, update_raster
import fuse_dev.fuse.interpolator.bag_interpolator as raster_interp
from fuse_dev.fuse.coverage.coverage import interpolation_coverage
from nbs_utils.gdal_utils import where_not_nodata

LOGGER = get_logger('xipe.csar.proc')
CONFIG_SECTION = 'combined_raster_processing'

DEFAULT_CARIS_ENVIRONMENT = 'CARIS35'
default_config_name = "default.config"
_debug = False

if __name__ == '__main__':
    
    config_path_file = 'proc_csar_config_path.config'
    config = {}
    config_file = configparser.ConfigParser()
    config_file.read(config_path_file)
    sections = config_file.sections()
    for section in sections:
        config_file_section = config_file[section]
        for key in config_file_section:
            config[key] = config_file_section[key]
    proc_csar_config_path = config['config_path']
    use_configs = os.path.join(proc_csar_config_path, os.getlogin())

    warnings = ""

    # Run each config
    for config_filename, config_file in iter_configs(use_configs):
        
        stringio_warnings = set_stream_logging("xipe", file_level=logging.WARNING, remove_other_file_loggers=False)
        LOGGER.info('***************************** Start Run  *****************************')
        LOGGER.info(f'reading "{config_filename}"')
        log_config(config_file, LOGGER)

        # Get stuff out of config
        config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
        caris_environment = config['caris_environment'] if 'caris_environment' in config else DEFAULT_CARIS_ENVIRONMENT
        bands = ['Elevation', 'Uncertainty', 'Contributor']
        output_extension = '.tiff'
        output_directory = config['output_directory'].strip('"')
        input_crs = parse_crs(config['input_crs']) if 'input_crs' in config and config['input_crs'] != 'None' else None
        output_crs = parse_crs(config['output_crs']) if 'output_crs' in config and config[
            'output_crs'] != 'None' else None
        vert_crs = split_compound_crs(config['output_crs'])[1].to_string() if 'output_crs' in config and config[
            'output_crs'] != 'None' else 'VERT_CS["unknown", VERT_DATUM["unknown", 2000]]'
        nbs_area_polygons = config['nbs_tile_polygons'].strip('"')
        if not os.path.isfile(nbs_area_polygons):
            raise ValueError("Invalid NBS tile polygons file")
        
        for csar_filename in csar_files:
            # Make the output filename and convert the CSAR file to a TIFF/CSV
            try:

                if not os.path.exists(output_directory):
                    os.makedirs(output_directory, exist_ok=True)

            except Exception as error:
                LOGGER.exception(f'{error.__class__.__name__} - {error}')
            finally:
                raster = None
        print(f"------------------------ repeating warnings for {config_filename} ------------------------------")
        print(stringio_warnings.getvalue())
        warnings += stringio_warnings.getvalue()

    print(f"------------------------ repeating warnings for all configs processed ------------------------------")
    print(warnings)
    print('done')


def generalize(raster_filename, closing_distance, output_crs=None, gen_contributor_idx=0,
               uncertainty_closing_distance_multiplier=.1, uncertainty_elevation_multiplier=.1):
    # get the crs from the raster_filename - if available
    input_crs = parse_crs(gdal.Open(raster_filename).GetProjection())  # or GetSpatialRef() ?

    # match the input crs if no transform is requested
    if output_crs is None:
        output_crs = input_crs
    else:
        output_crs = parse_crs(output_crs)

    # try:
    if 1:
        # reproject happens in Xipe
        # reproject_raster(raster_filename, input_crs, output_crs)
        
        try:
            elevation_band_index = raster_band_name_index(raster_filename, ELEVATION_BAND_NAME)
        except KeyError:
            elevation_band_index = None
        try:
            uncertainty_band_index = raster_band_name_index(raster_filename, UNCERTAINTY_BAND_NAME)
        except KeyError:
            uncertainty_band_index = None
        try:
            contributor_band_index = raster_band_name_index(raster_filename, CONTRIBUTOR_BAND_NAME)
        except KeyError:
            contributor_band_index = None

        raster = gdal.Open(raster_filename)
        band_names = []
        for band_num in range(raster.RasterCount):
            band = raster.GetRasterBand(band_num + 1)
            band_names.append(band.GetDescription().upper())
        elev_idx = band_names.index('ELEVATION') + 1
        elev_band = raster.GetRasterBand(elev_idx)
        raster_transform = raster.GetGeoTransform()
        nodata = elev_band.GetNoDataValue()

        # interpolate the combined raster within the new coverage provided by a binary closing

        resolution = abs(max(raster_transform[1:3]))

        LOGGER.info(f'performing generalized interpolation on "{raster_filename}"')
        LOGGER.info(f'Using a closing distance of {closing_distance} meters for {raster_filename}')
        generalized_interpolation = raster_interp.process.RasterInterpolator().interpolate(raster, 'linear', buffer=closing_distance/resolution)
        driver = gdal.GetDriverByName('GTiff')
        if _debug:
            try:
                new_ds = driver.CreateCopy(r"C:\Data\nbs\Tile12_temp_generalize_interp.tif", generalized_interpolation._dataset,
                                       options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
            except:
                new_ds = driver.CreateCopy(r"C:\Data\nbs\Tile12_temp_generalize_interp.tif", generalized_interpolation,
                                           options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
            del new_ds
        LOGGER.info('generalized interpolation completed.  Begin closing.')
        # This step is to be moved to Xipe

        ## create coverage vector that covers the tiles for the closing process
        tmp_dataset = gdal.GetDriverByName('MEM').CreateCopy('', raster)
        footprint_band = tmp_dataset.GetRasterBand(1)
        footprint_arr = footprint_band.ReadAsArray()
        footprint_arr[:,:] = 1
        footprint_band.WriteArray(footprint_arr)
        srs = gdal.osr.SpatialReference()
        srs.ImportFromWkt(raster.GetProjection())
        outfilename = 'tile_footprint.gpkg'
        with TemporaryDirectory() as tmpdir:
            outfilepath = os.path.join(tmpdir, outfilename)
            gpkg_drv = gdal.ogr.GetDriverByName("GPKG")
            dst_ds = gpkg_drv.CreateDataSource(outfilepath)
            path, filename = os.path.split(outfilename)
            layername, ext = os.path.splitext(filename)
            dst_layer = dst_ds.CreateLayer(layername, srs = srs)
            dst_def = dst_layer.GetLayerDefn()
            IDX = gdal.ogr.FieldDefn('Index', gdal.ogr.OFTInteger)
            dst_layer.CreateField(IDX)
            gdal.Polygonize(footprint_band, None, dst_layer, 0, [], callback=None )
            del dst_layer
            del dst_ds

            for generalized_area in interpolation_coverage(raster, generalized_interpolation, resolution, closing_distances=[closing_distance], support_filenames=[outfilepath]):
                generalized_data, closing_distance, not_present = generalized_area

        # determine change in coverage and calulated uncertainty layer
        LOGGER.info('calculating uncertainty in new coverage')
        if _debug:
            driver = gdal.GetDriverByName('GTiff')
            new_ds = driver.CreateCopy(r"C:\Data\nbs\Tile12_temp_generalize_data.tif", generalized_data._dataset,
                                       options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
            del new_ds
        # @todo use the bruty.utils.iter_gdal functions to avoid reading all the data into memory
        generalized_array = generalized_data.GetRasterBand(1).ReadAsArray()
        raster_array = raster.ReadAsArray()
        raster = None
        changed_idx = numpy.where(generalized_array != nodata)
        if len(changed_idx[0] > 0):
            raster_array[0][changed_idx] = generalized_array[changed_idx]
            above_zero_idx = numpy.where(numpy.logical_and(generalized_array > 0, (generalized_array != nodata)))
            dist_to_data = edt(~where_not_nodata(raster_array[1], nodata)) * resolution
            fixed_uncert = uncertainty_closing_distance_multiplier * dist_to_data
            uncertainty_array = raster_array[1]
            uncertainty_array[changed_idx] =  fixed_uncert + uncertainty_elevation_multiplier * - generalized_array[changed_idx]
            uncertainty_array[above_zero_idx] = fixed_uncert
            raster_array[1] = uncertainty_array

            generalized_interpolated_contributor = numpy.array(
                [str(gen_contributor_idx), 'N/A', 'NBS Generalization',
                 f'{datetime.today():%Y%m%d}', '0', '0.0', 'N/A'])
            LOGGER.info(
                f'adding new contributor {dict([(generalized_interpolated_contributor[3 - 1], generalized_interpolated_contributor[0])])}')
            raster_array[2][changed_idx] = gen_contributor_idx
            # add the new contributor to the contributor table
# @todo - is this not updating since the interpolated dataset seems to have data in it?
            # closed_mask[int(13184/4):int(13184/4)+90, int(0/4):3]
            # interpolated_dataset.GetRasterBand(1).ReadAsArray()[int(13184/4):int(13184/4)+90, int(0/4):3]
            update_raster(raster_filename, raster_array=raster_array)
        # @todo - confirm we are removing the clipping and leave that to Xipe
        # clip_nbs_tile(raster_filename, nbs_area_polygons, nbs_polygon_buffer)
        # modify_csv(raster_filename, contributor_table_filename)
        # sort_contributors(raster_filename, contributor_table_filename)

    # except Exception as error:
    #     LOGGER.exception(f'{error.__class__.__name__} - {error}')
    #     raise
    # finally:
    #     raster = None




