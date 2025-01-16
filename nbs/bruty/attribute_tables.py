# From xipe_dev originally
import os
import collections
from typing import Union
import copy

from osgeo import gdal
import numpy

from nbs.configs import get_logger
from nbs.bruty.contributor_metadata import contributor_metadata_from_file, DEFAULT_TO_FILENAME_COLUMN
from nbs.bruty.constants import CONTRIBUTOR_BAND_NAME
from nbs.bruty.raster_funcs import raster_band_name_index

# constants for field names in the raster attribute tables, based on the BAG update spec derived and S101
VALUE = 'value'
COUNT = 'count'
SOURCE_SURVEY = 'source_survey_id'
SOURCE_INSTITUTION = 'source_institution'
LICENSE = 'license_name'
LICENSE_URL = 'license_url'
SURVEY_DATE_START = 'survey_date_start'
SURVEY_DATE_END = 'survey_date_end'

DATA_ASSESSMENT = 'data_assessment'
FEATURE_LEAST_DEPTH = FEATURE_LD = 'feature_least_depth'
SIGNIFICANT_FEATURES = 'significant_features'
FEATURE_SIZE = 'feature_size'
FEATUTE_SIZE_VAR = 'feature_size_var'
COVERAGE = 'coverage'
BATHY_COVERAGE = 'bathy_coverage'
HORZ_UNCERT_FIXED = 'horizontal_uncert_fixed'
HORZ_UNCERT_VAR = 'horizontal_uncert_var'
VERT_UNCERT_FIXED = 'vertical_uncert_fixed'
VERT_UNCERT_VAR = 'vertical_uncert_var'
RAT_EXTENSION = ".aux.xml"

SOURCE_INDICATOR = 'sorind'
# SENSITIVE = 'sensitive'

CATZOC = 'catzoc'
SUPERCESSION = 'supercession_score'
DECAY = 'decay_score'
SOURCE_DB = 'source_table'
AREA_BATHY_COVERAGE = 'area_bathy_coverage'
AREA_COVERAGE = 'area_complete_coverage'

# These quality factors are being used to make a raster attribute table that is grouped by similar 'survey quality'
# the idea is to allow an automated contouring/sounging selection based on surveys with the right set of 'quality'
BOOLEAN_QUALITY_FACTORS = (COVERAGE, BATHY_COVERAGE, FEATURE_LD, SIGNIFICANT_FEATURES)
DEFAULT_QUALITY_FACTORS = (DATA_ASSESSMENT, BATHY_COVERAGE, COVERAGE, SIGNIFICANT_FEATURES, FEATURE_LD, FEATURE_SIZE, HORZ_UNCERT_FIXED, HORZ_UNCERT_VAR)

LOGGER = get_logger('xipe.raster.attr')

class RasterAttributeRecord(dict):
    TYPE_INDEX, USAGE_INDEX, DEFAULT_VALUE = range(3)
    fields = collections.OrderedDict()

    @classmethod
    def names(cls):
        return list(cls.fields.keys())
    def reset_to_defaults(self):
        for field, val in self.fields.items():
            self[field] = val[self.DEFAULT_VALUE]
    def default(self, field):
        return self.fields[field][self.DEFAULT_VALUE]


def read_rat(fname):
    # from osgeo import gdal
    ds = gdal.Open(fname)
    for n in range(1, ds.RasterCount+1):
        band = ds.GetRasterBand(n)
        rat = band.GetDefaultRAT()
        if rat:
            break
    # rat.GetColumnCount()
    # Out[16]: 16
    # rat.GetNameOfCol(0)
    # Out[17]: 'Value'
    # rat.GetNameOfCol(1)
    # Out[18]: 'Count'
    col_names = []
    for ncol in range(rat.GetColumnCount()):
        col_names.append(rat.GetNameOfCol(ncol))
    # rat.ReadAsArray(1)
    # Out[19]: array([   997, 2109541, 51037158])
    # rat.ReadAsArray(0)
    # Out[20]: array([14, 36, 38])
    col_index = {}
    for meta_name, col_name in [('count', COUNT), ('feat_detect', SIGNIFICANT_FEATURES), ('coverage', COVERAGE),
                                ('horiz_uncert_fixed', HORZ_UNCERT_FIXED),('horiz_uncert_vari', HORZ_UNCERT_VAR),
                                ('nbs_id', VALUE), ('institution', SOURCE_INSTITUTION), ('Source_Survey_ID', SOURCE_SURVEY)]:
        col_index[meta_name] = col_names.index(col_name)

    records = [{} for n in range(rat.GetRowCount())]
    for meta_name, col in col_index.items():
        data = rat.ReadAsArray(col)
        for n in range(rat.GetRowCount()):
            records[n][meta_name] = data[n]

    # records[0]
    # Out[44]:
    # {'feat_detect': 0,
    # 'complete_coverage': 1,
    # 'horiz_uncert_fixed': 500.0,
    # 'vert_uncert_fixed': 0.0,
    # 'horiz_uncert_vari': 0.0,
    # 'vert_uncert_vari': 0.0,
    # 'Source_Survey_ID': b'h05153.csar'}
    return records


class SurveyAttributeRecord(RasterAttributeRecord):
    fields = collections.OrderedDict(
        [(VALUE, (gdal.GFT_Integer, gdal.GFU_MinMax, 1000000)),
         (COUNT, (gdal.GFT_Integer, gdal.GFU_PixelCount, 0)),
         (DATA_ASSESSMENT, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),  # Valid numbers are 1 to 3, corresponding to S-101 encoding
         (FEATURE_LD, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),  # 0=False, 1=True  See S-101 least depth of detected feature measured.
         (SIGNIFICANT_FEATURES, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),  # 0=False, 1=True See S-101 significant features detected.
         (FEATURE_SIZE, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See S-101 feature size.
         # (FEATUTE_SIZE_VAR, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See further discussion (2)
         (COVERAGE, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),  # 0=False, 1=True  See S-101 full seafloor coverage achieved
         (BATHY_COVERAGE, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),  # 0=False, 1=True  See further discussion (3)
         (HORZ_UNCERT_FIXED, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See S-101 horizontal position uncertainty fixed
         (HORZ_UNCERT_VAR, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See S-101 horizontal position uncertainty variable factor
         (VERT_UNCERT_FIXED, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See S-101 horizontal position uncertainty fixed
         (VERT_UNCERT_VAR, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)),  # See S-101 horizontal position uncertainty variable factor
         (LICENSE, (gdal.GFT_String, gdal.GFU_Generic, "")),
         # (SENSITIVE, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)),
         (SOURCE_INDICATOR, (gdal.GFT_String, gdal.GFU_Generic, "")),
         (LICENSE_URL, (gdal.GFT_String, gdal.GFU_Generic, "")),
         (SOURCE_SURVEY, (gdal.GFT_String, gdal.GFU_Generic, "")),
         (SOURCE_INSTITUTION, (gdal.GFT_String, gdal.GFU_Generic, "")),
         (SURVEY_DATE_START, (gdal.GFT_String, gdal.GFU_Generic, "")),  # See S-101 Survey date start
         (SURVEY_DATE_END, (gdal.GFT_String, gdal.GFU_Generic, "")),  # See S-101 Survey date end
         (AREA_BATHY_COVERAGE, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)), # bathy area based coverage field populated by the branches
         (AREA_COVERAGE, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)), # area based coverage field populated by the branches
         (CATZOC, (gdal.GFT_Integer, gdal.GFU_Generic, 1000000)), # not in S-101 or BAG spec, but needed for Hydro Health
         (SUPERCESSION, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)), # not in S-101 or BAG spec, but needed for Hydro Health
         (DECAY, (gdal.GFT_Real, gdal.GFU_Generic, 1000000)), # not in S-101 or BAG spec, but needed for Hydro Health
         (SOURCE_DB, (gdal.GFT_String, gdal.GFU_Generic, "")), # not in S-101 or BAG spec, but needed for Hydro Health
         ])

    @staticmethod
    def from_contributor(contributor_name, contributor):
        """ Maps a record from the postgres database as read from meta_review.py into a record that will be written to raster attribute table.

        Parameters
        ----------
        contributor_name
            The survey name
        contributor
            dictionary that meta_review creates.  e.g.
            {'index': pixel-value, 'record': {postgres data}}

        Returns
        -------

        """
        try:
            record = SurveyAttributeRecord()
            # record.reset_to_defaults()
            record[VALUE] = contributor['index']
            record[COUNT] = record.fields[COUNT][record.DEFAULT_VALUE]
            record[SOURCE_SURVEY] = contributor_name
            contributor_record = contributor['record']
            if contributor_record is None and (contributor_name == "NBS interpolation" or contributor_name == "NBS Generalization"):
                record[SOURCE_SURVEY] = "NBS Generalization"
                record[SOURCE_INSTITUTION] = 'DOC/NOAA/NOS/OCS -- Office of Coast Survey'
                record[SURVEY_DATE_START] = 'N/A'
                record[SURVEY_DATE_END] = 'N/A'
                record[LICENSE] = 'cc0-1.0'
                record[LICENSE_URL] = 'https://creativecommons.org/publicdomain/zero/1.0/legalcode'
                record[DATA_ASSESSMENT] = 3  # this is 1,2, or 3 (not boolean)
                record[SOURCE_INDICATOR] = "N/A"
                record[CATZOC] = 5
                record[SOURCE_DB] = "N/A"
                for bool_field in (COVERAGE, BATHY_COVERAGE, SIGNIFICANT_FEATURES, FEATURE_LD, AREA_BATHY_COVERAGE, AREA_COVERAGE):  # SENSITIVE,
                    record[bool_field] = False
                for field in (VERT_UNCERT_VAR, VERT_UNCERT_FIXED, HORZ_UNCERT_FIXED,  HORZ_UNCERT_VAR, FEATURE_SIZE,
                              DECAY, SUPERCESSION, SOURCE_DB):
                    record[field] = record.default(field)
            elif contributor_record == "Missing":
                for field in (SOURCE_INSTITUTION, SURVEY_DATE_START, SURVEY_DATE_END, LICENSE, LICENSE_URL, DATA_ASSESSMENT,
                              SOURCE_INDICATOR, COVERAGE, BATHY_COVERAGE, SIGNIFICANT_FEATURES, FEATURE_LD,  # SENSITIVE,
                              VERT_UNCERT_VAR, VERT_UNCERT_FIXED, HORZ_UNCERT_FIXED, HORZ_UNCERT_VAR, FEATURE_SIZE, CATZOC,
                              AREA_BATHY_COVERAGE, AREA_COVERAGE, DECAY, SUPERCESSION, SOURCE_DB):
                    record[field] = record.default(field)
                    record[SOURCE_INSTITUTION] = "Missing"
                    record[SOURCE_DB] = "Missing"
            else:
                # map from the raster attribute (extended bag) keys to the postgres database names
                mapping = {SOURCE_INSTITUTION: 'survey_authority',
                           SURVEY_DATE_START: 'start_date',
                           SURVEY_DATE_END: 'end_date',
                           LICENSE: 'license',
                           LICENSE_URL: 'license_url',
                           VERT_UNCERT_VAR: 'vert_uncert_vari',
                           VERT_UNCERT_FIXED: 'vert_uncert_fixed',
                           HORZ_UNCERT_FIXED: 'horiz_uncert_fixed',
                           HORZ_UNCERT_VAR: 'horiz_uncert_vari',
                           SIGNIFICANT_FEATURES: 'feat_detect',
                           FEATURE_SIZE: 'feat_size',
                           # FEATURE_SIZE_VAR: 'feat_size_var',
                           FEATURE_LD: 'feat_least_depth',
                           # SENSITIVE: 'sensitive',
                           SOURCE_INDICATOR: 'source_indicator',
                           CATZOC: 'catzoc',
                           SUPERCESSION: 'supersession_score',
                           DECAY: 'decay_score',
                           SOURCE_DB: 'tablename',
                           AREA_BATHY_COVERAGE: 'bathymetry',
                           AREA_COVERAGE: 'complete_coverage',
                           }
                # right now everything is "assessed" in NBS and this would have to be changed in the future if other s101 values were desired/appropriate
                record[DATA_ASSESSMENT] = 1  # this is 1,2, or 3 (not boolean)
                record[BATHY_COVERAGE] = not contributor_record['interpolated']
                record[COVERAGE] = record[BATHY_COVERAGE] or contributor_record['feat_detect']
                for rat_key, contrib_key in mapping.items():
                    record[rat_key] = contributor_record.get(contrib_key, record.default(rat_key))
                    # if contrib_key == "catzoc":  # convert catzon from a string to an integer
                    #     record[rat_key] = int(record[rat_key])
        except Exception as e:
            raise type(e)(f"There was likely a problem with the contributor record, see exception above for the contributor:\n{contributor_name}: {contributor}") from e
        return record

class RasterAttributeTableDataset(list):
    """ A list of RasterAttributeRecord of all the same subtype (WARNING - do NOT mix record types)
    """
    def compute_counts(self, band_data: gdal.Band):
        self.sort(key=lambda d: d[VALUE])
        min_val = self[0][VALUE]
        max_val = self[-1][VALUE]
        # hist = band_data.GetHistogram(min_val - 0.5, max_val + 0.5, max_val - min_val + 1,  approx_ok=False)
        hist = numpy.array(band_data.GetHistogram(min_val - 0.5, max_val + 0.5, int(max_val - min_val) + 1, approx_ok=False), numpy.int32)
        # used_contributors = (numpy.where(hist > 0)[0] + minv).astype(numpy.int32)
        for record in self:
            record[COUNT] = int(hist[record[VALUE] - min_val])  # cast to python int from numpy int

    def remove_count_le_zero(self):
        for i in range(len(self) - 1, -1, -1):
            if self[i][COUNT] <= 0:
                self.pop(i)

    def write_to_raster(self, raster_filename: str, band: Union[str, int], fields: [str] = None,
                        compute_counts: bool = False, positive_counts_only: bool = False):
        """ Write a record attribute table using GDAL - which ends up as an xml file.
        THIS WILL OVERWRITE ANY EXISTING RASTER ATTRIBUTES FOR ALL BANDS (until we figure out how to update via gdal)

        Parameters
        ----------
        raster_filename
            full path filename of the raster to create a raster attribute table for
        band
            integer or string name of the band to create the table for
        fields
            list of strings of the fields from the contained records to write.  If None then all fields are written
        compute_counts
            compute the count of each value in the band
        positive_counts_only
            only export rows for records that occur in the band, will compute_counts as well to determine which records to retain

        Returns
        -------
            None

        """
        try:
            first_record = self[0]
        except IndexError:
            LOGGER.warning(f"No data records to write into raster attribute table for {raster_filename}")
        else:
            if not fields:
                fields = first_record.names()  # assumes all records are the same type
            dataset = gdal.Open(raster_filename, gdal.GA_Update)
            if isinstance(band, str):
                contributor_band_index = raster_band_name_index(raster_filename, band)
            else:
                contributor_band_index = band
            band_data = dataset.GetRasterBand(contributor_band_index)
            if compute_counts or positive_counts_only:
                self.compute_counts(band_data)
            raster_attribute_table = gdal.RasterAttributeTable()
            field_indices = {}
            for index, field in enumerate(fields):
                raster_attribute_table.CreateColumn(field,
                                                    first_record.fields[field][first_record.TYPE_INDEX],
                                                    first_record.fields[field][first_record.USAGE_INDEX])
                field_indices[field] = index
            if positive_counts_only:
                reduced_dataset = copy.copy(self)
                reduced_dataset.remove_count_le_zero()
            else:
                reduced_dataset = self
            raster_attribute_table.SetRowCount(len(reduced_dataset))
            for field_index, field in enumerate(fields):
                use_type = first_record.fields[field][first_record.TYPE_INDEX]
                if use_type == gdal.GFT_String:
                    add_data = raster_attribute_table.SetValueAsString
                elif use_type == gdal.GFT_Integer:
                    add_data = raster_attribute_table.SetValueAsInt
                elif use_type == gdal.GFT_Real:
                    add_data = raster_attribute_table.SetValueAsDouble
                else:
                    raise TypeError(f"Raster Attribute Field has unknown type {field} - {use_type}")

                for index, record in enumerate(reduced_dataset):
                    try:
                        add_data(index, field_index, record[field])
                    except Exception as e:
                        try:
                            if field == CATZOC:
                                add_data(index, field_index, int(record[field]))
                            elif field == SUPERCESSION:
                                add_data(index, field_index, float(record[field]))
                            else:
                                raise e
                        except Exception as e:
                            print("failure with", index, field, record)
                            raise e
            try:
                os.remove(raster_filename+RAT_EXTENSION)
            except FileNotFoundError:
                pass

            band_data.SetDefaultRAT(raster_attribute_table)
            # w = raster_attribute_table.ChangesAreWrittenToFile()
            # deleting the objects closes the dataset and writes the xml file
            del band_data
            del dataset

    @staticmethod
    def from_contributor_metadata(contributor_metadata):
        data = RasterAttributeTableDataset()
        for contributor_name, contributor in contributor_metadata.items():
            data.append(SurveyAttributeRecord.from_contributor(contributor_name, contributor))
        return data

    @staticmethod
    def from_contributor_file(contributor_filename, to_filename_column=DEFAULT_TO_FILENAME_COLUMN):
        contributor_metadata = contributor_metadata_from_file(contributor_filename, to_filename_column)
        return RasterAttributeTableDataset.from_contributor_metadata(contributor_metadata)


def create_RAT(raster_filename, contributor_metadata, fields=None, compute_counts: bool = True, positive_counts_only=True):
    """ For a tiff that has a band named CONTRIBUTOR_BAND_NAME  (e.g. 'Contributor')
    Add a raster attribute table (RAT).  Fill the RAT with metadata values looked up in the databases.
    Set the GFU_MinMax with the pixel value of the survey.
    The rest of each row record in the RAT will have the metadata as described at https://github.com/OpenNavigationSurface/BAG/issues/2

    temporal_variability Unsigned Integer # Valid numbers are 1 to 6, corresponding to S-101 encoding
    data_assessment Unsigned Integer # Valid numbers are 1 to 3, corresponding to S-101 encoding
    feature_least_depth Boolean # See S-101 least depth of detected feature measured.
    significant_features Boolean # See S-101 significant features detected.
    feature_size Float # See S-101 feature size.
    feature_size_var Float # See further discussion (2)
    coverage Boolean # See S-101 full seafloor coverage achieved
    bathy_coverage Boolean # See further discussion (3)
    horizontal_uncert_fixed Float # See S-101 horizontal position uncertainty fixed
    horizontal_uncert_var Float # See S-101 horizontal position uncertainty variable factor
    survey_date_start String # See S-101 Survey date start
    survey_date_end String # See S-101 Survey date end
    Source Institution String
    Source Survey ID String
    License Name String
    License URL or DOI String

    Parameters
    ----------
    raster_filename
    contributor_metadata
    fields
    compute_counts

    Returns
    -------

    """
    data = RasterAttributeTableDataset.from_contributor_metadata(contributor_metadata)
    data.write_to_raster(raster_filename, CONTRIBUTOR_BAND_NAME, fields, compute_counts, positive_counts_only=True)

    # dataset = gdal.Open(raster_filename, gdal.GA_Update)
    # contributor_band_index = raster_band_name_index(raster_filename, CONTRIBUTOR_BAND_NAME)
    # band = dataset.GetRasterBand(contributor_band_index)
    #
    # raster_attribute_table = gdal.RasterAttributeTable()
    # raster_attribute_table.CreateColumn('Value', gdal.GFT_Integer, gdal.GFU_MinMax)
    # raster_attribute_table.CreateColumn('Source_Survey_ID', gdal.GFT_String, gdal.GFU_Generic)
    # raster_attribute_table.CreateColumn('Source_Institution', gdal.GFT_String, gdal.GFU_Generic)
    # raster_attribute_table.CreateColumn('License_Name', gdal.GFT_String, gdal.GFU_Generic)
    # raster_attribute_table.CreateColumn('License_URL', gdal.GFT_String, gdal.GFU_Generic)
    # raster_attribute_table.CreateColumn('temporal_variability', gdal.GFT_Integer,
    #                                     gdal.GFU_Generic)  # Valid numbers are 1 to 6, corresponding to S-101 encoding
    # raster_attribute_table.CreateColumn('data_assessment', gdal.GFT_Integer,
    #                                     gdal.GFU_Generic)  # Valid numbers are 1 to 3, corresponding to S-101 encoding
    # raster_attribute_table.CreateColumn('feature_least_depth', gdal.GFT_Integer,
    #                                     gdal.GFU_Generic)  # 0=False, 1=True  See S-101 least depth of detected feature measured.
    # raster_attribute_table.CreateColumn('significant_features', gdal.GFT_Integer,
    #                                     gdal.GFU_Generic)  # 0=False, 1=True See S-101 significant features detected.
    # raster_attribute_table.CreateColumn('feature_size', gdal.GFT_Real, gdal.GFU_Generic)  # See S-101 feature size.
    # raster_attribute_table.CreateColumn('feature_size_var', gdal.GFT_Real, gdal.GFU_Generic)  # See further discussion (2)
    # raster_attribute_table.CreateColumn('full_coverage', gdal.GFT_Integer,
    #                                     gdal.GFU_Generic)  # 0=False, 1=True  See S-101 full seafloor coverage achieved
    # raster_attribute_table.CreateColumn('bathy_coverage', gdal.GFT_Integer, gdal.GFU_Generic)  # 0=False, 1=True  See further discussion (3)
    # raster_attribute_table.CreateColumn('horizontal_uncert_fixed', gdal.GFT_Real, gdal.GFU_Generic)  # See S-101 horizontal position uncertainty fixed
    # raster_attribute_table.CreateColumn('horizontal_uncert_var', gdal.GFT_Real,
    #                                     gdal.GFU_Generic)  # See S-101 horizontal position uncertainty variable factor
    # raster_attribute_table.CreateColumn('survey_date_start', gdal.GFT_String, gdal.GFU_Generic)  # See S-101 Survey date start
    # raster_attribute_table.CreateColumn('survey_date_end', gdal.GFT_String, gdal.GFU_Generic)  # See S-101 Survey date end
    # if contributor_metadata is None:
    #     contributor_metadata = contributor_metadata_from_file(interpolated_raster_filename, to_filename_column)
    # raster_attribute_table.SetRowCount(max([int(contributor['index']) for contributor in contributor_metadata.values()]) + 1)
    # for contributor_name, contributor in contributor_metadata.items():
    #     index = contributor['index']
    #     raster_attribute_table.SetValueAsInt(index, 0, index)
    #     raster_attribute_table.SetValueAsString(index, 1, contributor_name)
    #     # raster_attribute_table.SetValueAsInt(index, 1, contributor_name)
    #     # raster_attribute_table.SetValueAsDouble(index, 1, contributor_name)
    #
    # band.SetDefaultRAT(raster_attribute_table)
    #
    # del band
    # del dataset
