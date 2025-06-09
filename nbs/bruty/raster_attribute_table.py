import os
import pathlib
import glob
import functools

from osgeo import gdal
import numpy

from nbs.bruty.utils import tqdm, iterate_gdal_image

from nbs.bruty.attribute_tables import RasterAttributeTableDataset, SurveyAttributeRecord, VALUE, \
    DEFAULT_QUALITY_FACTORS, RAT_EXTENSION, BOOLEAN_QUALITY_FACTORS
from nbs.bruty.contributor_metadata import contributor_metadata_from_file, DEFAULT_TO_FILENAME_COLUMN
from nbs.configs import get_logger, iter_configs, log_config, set_stream_logging
from nbs.bruty.constants import CONTRIBUTOR_BAND_NAME, ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME
from nbs.bruty.raster_funcs import raster_band_name_index

gdal.DontUseExceptions()

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