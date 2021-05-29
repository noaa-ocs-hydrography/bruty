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
from tqdm import tqdm

from transform_dev.datum_transform.reproject import parse_crs, split_compound_crs
from xipe_dev.xipe import get_logger, iter_configs, log_config, set_stream_logging
from xipe_dev.xipe.raster import CONTRIBUTOR_BAND_NAME, ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME, \
    raster_band_name_index, reproject_raster, update_raster
import fuse_dev.fuse.interpolator.bag_interpolator as raster_interp
from fuse_dev.fuse.coverage.coverage import interpolation_coverage
from nbs_utils.gdal_utils import where_not_nodata
from nbs.bruty.utils import iterate_gdal_image, BufferedImageOps

LOGGER = get_logger('bruty.generalize')
CONFIG_SECTION = 'combined_raster_processing'

DEFAULT_CARIS_ENVIRONMENT = 'CARIS35'
default_config_name = "default.config"
_debug = False
write_distance = False  # FIXME - this doesn't work if no generalization takes place (need to change code below)
perform_closing = False

def generalize_tile(elev_array, uncert_array, contrib_array, nodata, closing_distance, resolution, gen_contributor_idx=0,
                     uncertainty_closing_distance_multiplier=.1, uncertainty_elevation_multiplier=.1, ext_progress=None):

    progress = ext_progress if ext_progress is not None else tqdm(desc="TileGen", total=5, leave=False)
    # interpolate the combined raster within the new coverage provided by a binary closing
    progress.set_description("Generalize")
    generalized_array = raster_interp.process.RasterInterpolator().interpolate_tile(elev_array, 'linear', nodata)
    progress.update(1)
    driver = gdal.GetDriverByName('GTiff')

    progress.set_description("Closing")
    # LOGGER.info('generalized interpolation completed.  Begin closing.')
    # This step is to be moved to Xipe
    if perform_closing:
        raise NotImplementedError("Removed closing for Xipe to implement")
        for generalized_area in interpolation_coverage(raster, generalized_interpolation, resolution,
                                                       closing_distances=[closing_distance], upsampled=True):  # support_filenames=[outfilepath]
            generalized_data, closing_distance, not_present = generalized_area
        generalized_dataset = generalized_data._dataset  # this is a fuse temp_dataset, so get the real gdal dataset
    progress.update(1)
    # LOGGER.info('closing completed.  Begin uncertainty computation.')
    progress.set_description("Find Empties")

    # use the closing distance to get enough data to perform the distance finding with edt
    # - if there is no data within the closing distance then we don't care what the result is

    # iterate the contributor index since we are modifying the data as we go
    # so checking elevation==nodata will fail as we modify what becomes the buffer area of the next block
    # so check contributor for being non-nan and non-generalization

    empty_data = numpy.logical_or(contrib_array == gen_contributor_idx, ~where_not_nodata(contrib_array, nodata))
    if perform_closing:  # the closing function creates a mask of where interpolation was valid
        # only has data where interp should be kept, nothing at real data and nothing at empty space too far from real data
        changed_idx = numpy.where(where_not_nodata(generalized_array, nodata))
    else:  # find the empty data in the original and make sure the generalize is not empty also
        changed_idx = numpy.where(numpy.logical_and(empty_data, where_not_nodata(generalized_array, nodata)))

    progress.update(1)
    progress.set_description("Distances")

    # use the full buffered data to determine distances
    # iterating the contributor index since we are modifying the data as we go
    # so checking elevation==nodata will fail as we modify what becomes the buffer area of the next block
    # so check contributor for being non-nan and non-generalization
    dist_to_data = edt(empty_data) * resolution
    # now match the array to just the data to modify (remove the buffers)
    # dist_to_data = image_ops.trim_array(dist_to_data)


    progress.update(1)
    progress.set_description("Uncert,Contrib")

    elev_array[changed_idx] = generalized_array[changed_idx]
    above_zero_idx = numpy.where(numpy.logical_and(generalized_array > 0, (generalized_array != nodata)))
    horiz_uncert = uncertainty_closing_distance_multiplier * dist_to_data
    uncert_array[changed_idx] = horiz_uncert[changed_idx] + uncertainty_elevation_multiplier * (-generalized_array[changed_idx])
    uncert_array[above_zero_idx] = horiz_uncert[above_zero_idx] + 1

    contrib_array[changed_idx] = gen_contributor_idx

    progress.update(1)
    if ext_progress is None:
        progress.close()
    return dist_to_data

def generalize(raster_filename, closing_distance, output_crs=None, gen_contributor_idx=0,
               uncertainty_closing_distance_multiplier=.1, uncertainty_elevation_multiplier=.1):
    if _debug:
        import pickle
        fname = "e:\\debug\\generalize.pickle"
        if os.path.exists(fname):
            outfile = open(fname, "rb")
            raster_filename, closing_distance, output_crs, gen_contributor_idx, \
            uncertainty_closing_distance_multiplier, uncertainty_elevation_multiplier = pickle.load(outfile)
            shutil.copyfile(raster_filename + ".orig", raster_filename)
        else:
            outfile = open(fname, "wb")
            pickle.dump((str(raster_filename), closing_distance, output_crs, gen_contributor_idx,
                         uncertainty_closing_distance_multiplier, uncertainty_elevation_multiplier), outfile)
            shutil.copyfile(raster_filename, str(raster_filename) + ".orig")
        outfile.close()

    # try:
    if 1:
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

        raster = gdal.Open(raster_filename, gdal.GA_Update)
        raster_transform = raster.GetGeoTransform()
        # nodata = elev_band.GetNoDataValue()

        # interpolate the combined raster within the new coverage provided by a binary closing
        resolution = abs(max(raster_transform[1:3]))
        cache_name = r"e:\debug\temp_generalize_data.tif"
        if _debug and os.path.exists(cache_name):
            generalized_dataset = gdal.Open(cache_name)
        else:
            LOGGER.info(f'performing generalized interpolation on "{raster_filename}"')
            LOGGER.info(f'Using a closing distance of {closing_distance} meters for {raster_filename}')
            generalized_interpolation = raster_interp.process.RasterInterpolator().interpolate(raster, 'linear', buffer=closing_distance / resolution)
            driver = gdal.GetDriverByName('GTiff')

            if False and _debug:
                try:
                    new_ds = driver.CreateCopy(r"C:\Data\nbs\Tile12_temp_generalize_interp.tif", generalized_interpolation._dataset,
                                               options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
                except:
                    new_ds = driver.CreateCopy(r"C:\Data\nbs\Tile12_temp_generalize_interp.tif", generalized_interpolation,
                                               options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
                del new_ds
            LOGGER.info('generalized interpolation completed.  Begin closing.')
            # This step is to be moved to Xipe
            if perform_closing:
                for generalized_area in interpolation_coverage(raster, generalized_interpolation, resolution,
                                                               closing_distances=[closing_distance],
                                                               upsampled=True):  # support_filenames=[outfilepath]
                    generalized_data, closing_distance, not_present = generalized_area
                generalized_dataset = generalized_data._dataset  # this is a fuse temp_dataset, so get the real gdal dataset
                if _debug:
                    driver = gdal.GetDriverByName('GTiff')
                    new_ds = driver.CreateCopy(cache_name, generalized_dataset,
                                               options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
                    del new_ds
            else:
                generalized_dataset = generalized_interpolation._dataset
        LOGGER.info('closing completed.  Begin uncertainty computation.')
        if write_distance:
            # create an image to store the distance values in
            dst_ds = driver.Create(str(raster_filename) + ".distance.tif", generalized_dataset.RasterXSize, generalized_dataset.RasterYSize, bands=1,
                                   eType=gdal.GDT_Int32, options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
            dst_ds.SetGeoTransform(generalized_dataset.GetGeoTransform())
            dst_ds.SetProjection(generalized_dataset.GetProjection())
            distance_band = dst_ds.GetRasterBand(1)
            distance_band.SetNoDataValue(generalized_dataset.GetRasterBand(1).GetNoDataValue())

        # use the closing distance to get enough data to perform the distance finding with edt
        # - if there is no data within the closing distance then we don't care what the result is
        image_ops = BufferedImageOps(raster)
        buff = closing_distance / resolution
        generalized_band = generalized_dataset.GetRasterBand(1)
        # iterate the contributor index since we are modifying the data as we go
        # so checking elevation==nodata will fail as we modify what becomes the buffer area of the next block
        # so check contributor for being non-nan and non-generalization
        for block in image_ops.iterate_gdal(buff, buff, band_nums=(contributor_band_index,), min_block_size=8192, max_block_size=8192):
            ic, ir, cols, rows, col_buffer_lower, row_buffer_lower, nodata, data = block
            contrib_array = data[0]
            generalized_array = generalized_band.ReadAsArray(ic, ir, cols, rows)  # just the data to modify (not the buffers)
            empty_data = numpy.logical_or(contrib_array == gen_contributor_idx, ~where_not_nodata(contrib_array, nodata))
            if perform_closing:  # the closing function creates a mask of where interpolation was valid
                # only has data where interp should be kept, nothing at real data and nothing at empty space too far from real data
                changed_idx = numpy.where(where_not_nodata(generalized_array, nodata))
            else:  # find the empty data in the original and make sure the generalize is not empty also
                changed_idx = numpy.where(numpy.logical_and(image_ops.trim_array(empty_data), where_not_nodata(generalized_array, nodata)))

            if len(changed_idx[0]) > 0:
                # use the full buffered data to determine distances
                # iterating the contributor index since we are modifying the data as we go
                # so checking elevation==nodata will fail as we modify what becomes the buffer area of the next block
                # so check contributor for being non-nan and non-generalization
                dist_to_data = edt(empty_data) * resolution
                # now match the array to just the data to modify (remove the buffers)
                dist_to_data = image_ops.trim_array(dist_to_data)

                elevations = image_ops.read(elevation_band_index, buffered=False)
                # band = raster.GetRasterBand(1)
                elevations[changed_idx] = generalized_array[changed_idx]
                image_ops.write_array(elevations, elevation_band_index)

                uncertainty_array = image_ops.read(uncertainty_band_index, buffered=False)
                above_zero_idx = numpy.where(numpy.logical_and(generalized_array > 0, (generalized_array != nodata)))
                horiz_uncert = uncertainty_closing_distance_multiplier * dist_to_data
                # uncertainty_array = raster_array[1]
                uncertainty_array[changed_idx] = horiz_uncert[changed_idx] + uncertainty_elevation_multiplier * - generalized_array[changed_idx]
                uncertainty_array[above_zero_idx] = horiz_uncert[above_zero_idx] + 1
                image_ops.write_array(uncertainty_array, uncertainty_band_index)

                contrib_array = image_ops.trim_array(contrib_array)  # use the data we already read above
                contrib_array[changed_idx] = gen_contributor_idx
                image_ops.write_array(contrib_array, contributor_band_index)
                if write_distance:
                    image_ops.write_array(dist_to_data, distance_band)


if __name__ == '__main__':
    if _debug:
        LOGGER.warning("Running cached data for generalize debugging")
        generalize("", 0)
