import os
import pathlib
import csv
import glob
import sys
import logging
import functools
import collections

from osgeo import gdal
import numpy

from nbs.bruty.utils import tqdm, iterate_gdal_image

from xipe_dev.xipe.raster import CONTRIBUTOR_BAND_NAME, copy_raster, raster_band_name_index
from xipe_dev.xipe import get_logger, iter_configs, log_config, set_stream_logging
from xipe_dev.xipe.raster.attribute_tables import RasterAttributeTableDataset, SurveyAttributeRecord, VALUE, \
    DEFAULT_QUALITY_FACTORS, RAT_EXTENSION, BOOLEAN_QUALITY_FACTORS
from xipe_dev.xipe.cell_contributors import read_tile_to_enc_cells
from xipe_dev.xipe.csar.contributor_metadata import contributor_metadata_from_file, DEFAULT_TO_FILENAME_COLUMN
from xipe_dev.xipe.deliverables import CLOUD_OPTIMIZED_GEOTIFF_CREATION_OPTIONS, OVERVIEW_DOWNSAMPLING

LOGGER = get_logger('xipe.scripts.RAT')
CONFIG_SECTION = 'combined_raster_processing'
QUALITY_BAND_NAME = "Quality"
SOURCE_EXT = '.source_information'
QUAL_EXT = '.quality_of_bathymetric_data'

def get_enc_name_from_raster(enc_raster_filename):
    # Determine which ENC we are working on
    raise NotImplementedError
    return enc_cell_name

def find_tile_for_enc(enc_cell_name, tiles_to_cells):
    # For an enc raster figure out what tile it came from
    for t, cells in tiles_to_cells.items():
        if enc_cell_name in cells:
            tile_number = t
            break
    return tile_number

def convert_raster_to_single_band_int(raster_filename, band=CONTRIBUTOR_BAND_NAME):
    # Read a raster (enc) and reduce to the contributor band only
    # raster_filename = r"C:\NBS_Products\OCM\PBC_UTM19N_MLLW\Initial\Band 4\interp\NBS_US4ME1CD_20200923_0_pre_public_interp.tiff"

    # if one band then no-op, if multiband then only keep the contributors
    raster = rasterio.open(raster_filename)
    bands = list(raster.descriptions)
    is_float = 'float' in raster.profile['dtype']
    nodata = raster.profile['nodata']
    kwargs = {}
    if is_float:
        rio = raster.read()
        kwargs['nodata'] = -1
        kwargs['dtype'] = rasterio.int32
        rio[rio==nodata] = kwargs['nodata']
        kwargs['raster_array'] = rio.astype(rasterio.int32)  # numpy.array([rio.astype(rasterio.int32)]*len(bands))
        del rio
    raster.close()
    if len(bands) > 1 or is_float:
        basename, ext = os.path.splitext(raster_filename)
        final_filename = basename + SOURCE_EXT + ext
        try:
            bands.remove(band)  # remove all the other bands by leaving them in the list to omit
        except ValueError:
            raise KeyError(f"Did not find {band} in {raster_filename}, only has {bands}") from None
        copy_raster(raster_filename, final_filename, omit_bands=bands, **kwargs)
    else:
        final_filename = raster_filename
    return final_filename


def sortkey_quality_groups(quality_factors, quality_group):
    # we want to sort True first then False then other numbers (like nodata) for boolean integers
    # so we will make a list of modified values and the original values
    sort_on = []
    for name, val in zip(quality_factors, quality_group):
        if name in BOOLEAN_QUALITY_FACTORS:
            if val in (False, True, 1, 0):
                # Put a zero in the first column so it sorts before a nodata number or other value
                # then flip the bool so True sorts first for ascending order
                sort_on.append((0, not val))
            else:
                # put a one so it sorts after a boolean value then the number value
                sort_on.append((1, val))
        else:
            sort_on.append(val)
    return sort_on

# FIXME -- not reimplemented yet!!!!!!!!!!!!!!!!!!!!!!!!!!!
def make_combined_raster_with_attr_table(raster_filename, quality_factors=DEFAULT_QUALITY_FACTORS):
    """  Create a raster attribute table for a raster file.
    Will create a second single band raster of just contributor info if needed (since Arc seems to limit to single band).

    1) Determine which ENC we are working on
    2) For an enc raster figure out what tile it came from
    3) Get the raster attribute data for the full tile
    4) Read a raster (enc) and reduce to the contributor band only
    5) Trim the raster attribute data back to just the contributors from this raster (enc)
    6) For the single band raster write a raster attribute table to disk

    Parameters
    ----------
    enc_raster_filename
        the enc raster full path to process
    tiles_to_cells
        dictionary of review tile to enc cells to use to reverse the lookup to find a tile based on the enc
    overview_tiles_directory
        overview tiles (numbers 1 to about 50 for PBC)
    quality_factors
        list of SurveyAttributeRecord keys that should be used to make a new raster based on grouping those keys together.
        e.g. if [DATA_ASSESSMENT] was supplied which is derived from CATZOC then all the catzocs would be grouped into one pixel value
    enc_cell_name
        the cell name of the raster file, if not supplied an attempt to read it from the file can be made
    cached_tile_info
        if a dictionary is supplied the tile number is used as a key to read existing data or to add newly read data into
    Returns
    -------

bathymetry in the RAT / BAG should correspond to ~interpolated in our database,
meaning if the node in the raster came from a metadata entry in the database that was interpolated = True then bathymetry = False in the RAT.
If interpolated = False in the database for a node then bathymetry = True.
All nodes from our generalized bathymetry should also have bathyemtry = False

coverage in the RAT / BAG should correspond to ~interpolated or feat_detect, where feat_detect is the feature detection flag in our database.
This assumes the only additional coverage for a survey beyond the source bathymetry is provided by side scan for object detection, but that is the only case I am aware of at this time.
If this logic does not agree with what we discussed please let me know.
    """

    # Determine which ENC we are working on
    if not enc_cell_name:
        enc_cell_name = get_enc_name_from_raster(enc_raster_filename)
    # For an enc raster figure out what tile it came from
    tile_number = find_tile_for_enc(enc_cell_name, tiles_to_cells)

    # Get the raster attribute data for the full tile
    if cached_tile_info is not None and tile_number in cached_tile_info:
        contributor_metadata = cached_tile_info[tile_number]
    else:
        # use the generalized contributor info since it has all the possible contributors including the fake NBS interpolation entry.
        search_glob = os.path.join(overview_tiles_directory, f"Tile{tile_number}_*_generalized_Contributor.csv")
        contributors_files = glob.glob(search_glob)
        if len(contributors_files) == 1:
            contributor_filename = contributors_files[0]
        elif len(contributors_files) == 0:
            raise FileNotFoundError(f"could not find a contributor csv for {enc_raster_filename} using {search_glob}")
        else:
            raise FileExistsError(f"Found multiple contributor csv files for {enc_raster_filename} :\n {contributors_files}")

        if 1:
            contributor_metadata = contributor_metadata_from_file(contributor_filename, DEFAULT_TO_FILENAME_COLUMN, retain_missing_records=True)
        else:
            record = get_record()
            simple_record = fuse_dev.fuse.meta_review.MetadataTable._simplify_record(record)  # todo - make this a static method
            fuse_dev.fuse.meta_review.meta_review.records_to_fusemetadata(simple_record)  # re-casts the db values into other/desired types


        if cached_tile_info is not None:
            cached_tile_info[tile_number] = contributor_metadata
    src_data = RasterAttributeTableDataset.from_contributor_metadata(contributor_metadata)
    # Read a raster (enc) and reduce to the contributor band only
    src_filename = convert_raster_to_single_band_int(enc_raster_filename, CONTRIBUTOR_BAND_NAME)

    # Open the dataset, compute counts then figure out what quality values there are (catzoc + horizontal_uncert)
    dataset = gdal.Open(src_filename, gdal.GA_ReadOnly)
    contributor_band_index = raster_band_name_index(src_filename, CONTRIBUTOR_BAND_NAME)
    raster_band = dataset.GetRasterBand(contributor_band_index)
    contrib_array = raster_band.ReadAsArray()
    quality_array = numpy.zeros(contrib_array.shape, dtype=contrib_array.dtype)
    # Compute the histogram of pixels/contributor and trim the raster attribute data back to just the contributors to this raster (enc)
    src_data.compute_counts(raster_band)
    src_data.remove_count_le_zero()
    # For the single band raster write a raster attribute table to disk
    del raster_band, dataset
    src_data.write_to_raster(src_filename, CONTRIBUTOR_BAND_NAME)

    # group the contributors together based on quality metrics and make a new raster (or layer) for that
    consolidated_quality_contributors = {}
    for record in src_data:
        key = tuple([record[k] for k in quality_factors])
        consolidated_quality_contributors.setdefault(key, []).append(record[VALUE])

    key_func = functools.partial(sortkey_quality_groups, quality_factors)
    quality_groups = list(consolidated_quality_contributors.keys())
    quality_groups.sort(key=key_func)
    quality_data = RasterAttributeTableDataset()
    # For each quality level/grouping, make a record in the raster attributes and fill that record value into the quality array
    for i, quality_group in enumerate(quality_groups):
        record = SurveyAttributeRecord([(k, v) for k, v in zip(quality_factors, quality_group)])
        record[VALUE] = i + 1  # make sure VALUE is populated and is a one based index instead of zero based (zero is our nodata value)
        quality_data.append(record)
        # wherever the contributor was in the original array set the new quality index
        for contributor_index in consolidated_quality_contributors[quality_group]:
            quality_array[contrib_array==contributor_index] = record[VALUE]
    if SOURCE_EXT in src_filename:
        qual_filename = src_filename.replace(SOURCE_EXT, QUAL_EXT)
    else:
        basename, ext = os.path.splitext(src_filename)
        qual_filename = basename + QUAL_EXT + ext

    copy_raster(src_filename, qual_filename, raster_array=numpy.array([quality_array]), nodata=0)
    # rename Contributor to Quality for the band name
    dataset = gdal.Open(qual_filename, gdal.GA_Update)
    raster_band = dataset.GetRasterBand(1)
    raster_band.SetDescription(QUALITY_BAND_NAME)
    del raster_band, dataset

    # make sure VALUE is in the list otherwise ARC will add its own using zero which is the no-data value
    if VALUE not in quality_factors:
        use_fields = [VALUE]
        use_fields.extend(quality_factors)
    else:
        use_fields = quality_factors
    quality_data.write_to_raster(qual_filename, QUALITY_BAND_NAME, fields=use_fields)
    return src_filename, qual_filename

def add_nbs_description(filename):
    # rename the band from contributor to Quality and add tags to both files
    dataset = gdal.Open(filename, gdal.GA_Update)
    raster_band = dataset.GetRasterBand(1)
    d = dataset.GetMetadata()
    d['TIFFTAG_IMAGEDESCRIPTION'] = """This multi-layered image is part of NOAA Office of Coast Survey’s National Bathymetry. The National Bathymetric Source is created to serve chart production and support navigation, but these specific files may include unqualified data and have been transformed from chart datum and are therefore not suitable for navigation.\n
The bathymetry is compiled from multiple sources with varying quality and includes forms of interpolation. The bathymetric quality and interpolation is described through the vertical uncertainty and data quality layers."""
    d['TIFFTAG_COPYRIGHT'] = 'cc0-1.0, https://creativecommons.org/share-your-work/public-domain/cc0/'
    dataset.SetMetadata(d)
    # bd = raster_band.GetMetadata()
    # bd['TIFFTAG_IMAGEDESCRIPTION'] = "Test band description"
    # raster_band.SetMetadata(bd)

    # hack = raster_band.ReadAsArray(0, 0, 1, 1)  # gdal wasn't writing the metadata unless we change a pixel too, not sure why
    # raster_band.WriteArray(hack, 0, 0)
    del raster_band
    del dataset
    # 1. The NBS was created to serve chart production and support the mariners. The data is publicly available to support the broader end-user community; 2. The data includes uncertainty (vertical and horizontal) based on work conducted at office of Coast Survey and the Joint Hydrographic Center.

# if __name__ == '__main__':
#     if len(sys.argv) > 1:
#         use_configs = sys.argv[1:]
#     else:
#         use_configs = pathlib.Path(__file__).parent.resolve()  # (os.path.dirname(os.path.abspath(__file__))
#
#     warnings = ""
#     for config_filename, config_file in iter_configs(use_configs):
#         stringio_warnings = set_stream_logging("xipe", file_level=logging.WARNING, remove_other_file_loggers=False)
#         LOGGER.info(f'***************************** Start Run  *****************************')
#         LOGGER.info(f'reading "{config_filename}"')
#         log_config(config_file, LOGGER)
#
#         try:
#             config = config_file[CONFIG_SECTION if CONFIG_SECTION in config_file else 'DEFAULT']
#             # values read from the config file -- this is a backwards lookup to see where process_csar wrote the data
#             output_overview_directory = os.path.abspath(os.path.expanduser(config['output_overview_directory'].strip('"'))) if 'output_overview_directory' in config else config['output_directory']
#             output_tiles_directory = os.path.abspath(os.path.expanduser(config['output_tiles_directory'].strip('"'))) if 'output_tiles_directory' in config else output_overview_directory
#             filter_cells_filename = config['review_tile_to_ENC_cell'] if 'review_tile_to_ENC_cell' in config else ""
#
#             tiles_to_cells = read_tile_to_enc_cells(filter_cells_filename)
#             if not tiles_to_cells:
#                 raise FileNotFoundError(f"'tile to cell lookup' shapefile not read correctly from {filter_cells_filename}")
#             cache = {}
#             for subtype in ['generalized', 'source', "interp", ]:
#                 LOGGER.info(f"processing directory {subtype}")
#                 enc_csv = glob.glob(os.path.join(output_tiles_directory, subtype, '*_cells.csv'))
#                 for csv_path in enc_csv:
#                     LOGGER.info(f"processing csv file {csv_path}")
#                     with open(csv_path) as csvfile:
#                         reader = csv.DictReader(csvfile)
#                         for enc_row in reader:
#                             LOGGER.info(f"processing cell {enc_row}")
#                             enc_full_path = os.path.join(output_tiles_directory, subtype, enc_row['filename'])
#                             begin, end = os.path.splitext(enc_full_path)  # encname, ".src.tiff"
#                             expected_final_name = os.path.join(begin + "_p" + QUAL_EXT + end + RAT_EXTENSION)
#                             if os.path.exists(expected_final_name):
#                                 LOGGER.info(f"Skipping {enc_row['filename']} since the output already exists")
#                                 continue
#
#                             src_filename, qual_filename = make_enc_raster_attr_table(enc_full_path, tiles_to_cells, output_overview_directory,
#                                                                                      enc_cell_name=enc_row['cell'], cached_tile_info=cache)
#
#                             # add NBS description, add an _p to filenames to match Glen's datum change naming
#                             for filename in (src_filename, qual_filename):
#                                 add_nbs_description(filename)
#                                 dir, fname = os.path.split(filename)
#                                 begin, end = fname.split(".", 1)  # encname, ".src.tiff"
#                                 new_name = os.path.join(dir, begin + "_p." + end)
#                                 os.replace(filename, new_name)
#                                 os.replace(filename+RAT_EXTENSION, new_name+RAT_EXTENSION)
#         except Exception as error:
#             LOGGER.exception(f'{error.__class__.__name__} - {error}')
#         print(f"------------------------ repeating warnings for {config_filename} ------------------------------")
#         print(stringio_warnings.getvalue())
#         warnings += stringio_warnings.getvalue()
#
#     print(f"------------------------ repeating warnings for all configs processed ------------------------------")
#     print(warnings)
#     print('done')


def make_raster_attr_table(tile_filename, meta_records, contrib_band=3):
    """  Create a raster attribute table for a raster file.
    Will create a second single band raster of just contributor info if needed (since Arc seems to limit to single band).

    Parameters
    ----------
    tile_filename
        raster tile
    tile_contributors_filename
        raster tile contrib csv
    quality_factors
        list of SurveyAttributeRecord keys that should be used to make a new raster based on grouping those keys together.
        e.g. if [DATA_ASSESSMENT] was supplied which is derived from CATZOC then all the catzocs would be grouped into one pixel value
    Returns
    -------

bathymetry in the RAT / BAG should correspond to ~interpolated in our database,
meaning if the node in the raster came from a metadata entry in the database that was interpolated = True then bathymetry = False in the RAT.
If interpolated = False in the database for a node then bathymetry = True.
All nodes from our generalized bathymetry should also have bathyemtry = False

coverage in the RAT / BAG should correspond to ~interpolated or feat_detect, where feat_detect is the feature detection flag in our database.
This assumes the only additional coverage for a survey beyond the source bathymetry is provided by side scan for object detection, but that is the only case I am aware of at this time.
If this logic does not agree with what we discussed please let me know.
    """

    # contributor_metadata = contributor_metadata_from_file(tile_contributors_filename, DEFAULT_TO_FILENAME_COLUMN, retain_missing_records=True)
    # contributor_table = numpy.loadtxt(tile_contributors_filename, delimiter=',', dtype=str)
    # contributors = contributor_table[:, 2]
    # contributor_tables = contributor_table[:, 6]

    # @todo compile counts using loop instead of reading entire band -- need to change RasterAttributeTable class for this
    # ds = gdal.Open(tile_filename)
    # for ic, ir, nodata, [contrib] in iterate_gdal_image(ds, [contrib_band]):

    """
    from osgeo import gdal
    fname=r"C:\Data\bruty_databases\output\Tile20_PBC18_4.tif"
    ds = gdal.Open(fname)
    band = ds.GetRasterBand(3)
    band.ComputeRasterMinMax(False)
    Out[6]: (0.0, 721643.0)
    hist = band.GetHistogram(0.0 - 0.5, 721643 + 0.5, 721643 - 0 + 1,  approx_ok=False)
    hist[0]
    Out[9]: 22114845
    hist[721643]
    Out[10]: 7786
    import numpy
    hist = numpy.array(band.GetHistogram(0.0 - 0.5, 721643 + 0.5, 721643 - 0 + 1,  approx_ok=False), numpy.int32)
    hist[0]
    Out[14]: 22114845
    numpy.where(hist>0)
    Out[15]: 
    (array([     0, 720105, 720109, 720110, 720142, 720166, 720271, 720323,
            720421, 720430, 720488, 720536, 720537, 720692, 720826, 720925,
            720941, 720955, 720959, 720991, 721017, 721053, 721094, 721180,
            721192, 721199, 721291, 721303, 721449, 721509, 721611, 721635,
            721643], dtype=int64),)
    """
    ds = gdal.Open(tile_filename)
    contributor_band_index = raster_band_name_index(str(tile_filename), CONTRIBUTOR_BAND_NAME)
    band = ds.GetRasterBand(contributor_band_index)
    contributor_metadata = {}
    try:
        minv, maxv = band.ComputeRasterMinMax(False)
    except RuntimeError:
        used_contributors = []  # no data found so make a blank RAT
        # have to add a record in order to find the fields to use for the RAT, even though there are no data points
        contributor_metadata["NBS Generalization"] = {'record': None, 'sensitive': False, 'index': 0}
    else:
        hist = numpy.array(band.GetHistogram(minv - 0.5, maxv + 0.5, int(maxv - minv) + 1,  approx_ok=False), numpy.int32)
        # close the gdal file
        band = None; ds = None
        used_contributors = (numpy.where(hist > 0)[0] + minv).astype(numpy.int32)
        for n, nbs_id in enumerate(used_contributors):
            if nbs_id > 0:
                rec = meta_records[nbs_id]
                contributor_metadata[rec['from_filename']] = {'record': rec, 'index': int(nbs_id)}  # ugh, int(nbs_id) to cast out of numpy.int32 for use later
            else:
                contributor_metadata["NBS Generalization"] = {'record': None, 'sensitive': False, 'index': int(nbs_id)}
            # if contributor_metadata[contributor]['record'] is None or contributor_metadata[contributor]['record'] == "Missing":
            #     print(f"{contributor} did not have record, sensitive value is not adjusted by database table name")
            #     continue
            # if contributors[n] == contributor:
            #     if "SENSITIVE" in contributor_tables[n].upper():
            #         contributor_metadata[contributor]['record']['sensitive'] = True
            #     else:
            #         contributor_metadata[contributor]['record']['sensitive'] = False
            # else:
            #     raise ValueError("contributors not aligned")
    src_data = RasterAttributeTableDataset.from_contributor_metadata(contributor_metadata)  # this will reduce the records to the desired fields
    src_data.write_to_raster(tile_filename, CONTRIBUTOR_BAND_NAME, positive_counts_only=True)
    return tile_filename