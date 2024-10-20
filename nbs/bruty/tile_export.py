import argparse
import os
from stat import S_IREAD, S_IRGRP, S_IROTH
import time
import datetime
import tempfile
import re
import sys
import subprocess
import pathlib
import shutil
import traceback
import logging
import glob
from functools import total_ordering

from shapely import wkt, wkb
from osgeo import ogr, osr, gdal
import pickle
from functools import partial

import numpy
import psycopg2
try:
    from numba import jit
    has_numba = True
except ModuleNotFoundError:
    has_numba = False


from data_management.db_connection import connect_with_retries
from fuse_dev.fuse.meta_review import meta_review
from nbs.configs import iter_configs, read_config
from nbs.bruty.utils import affine, get_crs_transformer, make_mllw_height_wkt, user_action, tqdm, remove_file, \
    iterate_gdal_image, BufferedImageOps, QUIT, HELP, contributor_int_to_float, contributor_float_to_int
from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, ConnectionInfo, connection_with_retries, connect_params_from_config
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, INTERNAL, NAVIGATION, PUBLIC, SCORING_METADATA_COLUMNS, EXPORT_METADATA_COLUMNS
from nbs.bruty.world_raster_database import WorldDatabase, use_locks, AdvisoryLock, BaseLockException
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError
from nbs.configs import get_logger
from nbs.bruty.nbs_locks import LockNotAcquired, AreaLock, FileLock, EXCLUSIVE, SHARED, NON_BLOCKING, SqlLock, NameLock, start_server, current_address
from nbs.bruty.generalize import generalize, generalize_tile
from nbs.bruty.raster_attribute_table import make_raster_attr_table
from xipe_dev.xipe.raster import where_not_nodata
from nbs.bruty.raster_data import LayersEnum
from fuse_dev.fuse.fuse_processor import git_head_commit_id
import nbs.scripts.combine
from xipe_dev.xipe.raster import CONTRIBUTOR_BAND_NAME, ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME, raster_band_name_index
from nbs.scripts.tile_specs import create_world_db, SUCCEEDED, TILE_LOCKED, UNHANDLED_EXCEPTION, DATA_ERRORS, \
    TileInfo, ResolutionTileInfo, CombineTileInfo


LOGGER = get_logger('nbs.bruty.export')
VERSION = (1, 0, 0)
__version__ = '.'.join(map(str, VERSION))
DEFAULT_BLOCK_SIZE = 8192  # roughly a gig of memory usage: 16 * 512 (gdal tile size) which when taken as a square of three bands of floats 8192^2 * 3 * 4 = 800MB

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
interactive_debug = False
if interactive_debug and sys.gettrace() is None:  # this only is set when a debugger is run (?)
    interactive_debug = False

# The export types are not using these data structures anymore.
#   For resource efficiency the exports are broken up by hand.
#   Does navigation then appends data to make public then appends again for internal.
#   We could spend the time to get fancy and automate ways to build efficiently or go back to building all separately.
# each datasets is a tuple of (datatype, for_nav)
# NOTE:  It is assumed that public builds on navigation and internal builds on public by using set differences in the combine_and_export function
export_definitions = {
    PUBLIC: {'datasets': {(REVIEWED, True), (PREREVIEW, True), (ENC, True), (GMRT, True),
                          (REVIEWED, False), (PREREVIEW, False), (ENC, False), (GMRT, False)},
             "name": PUBLIC},  # public use
    NAVIGATION: {'datasets': {(REVIEWED, True), (ENC, True), (GMRT, True)},
                 "name": NAVIGATION},  # MCD charting use
    INTERNAL: {'datasets': {(REVIEWED, True), (PREREVIEW, True), (SENSITIVE, True), (ENC, True), (GMRT, True),
                            (REVIEWED, False), (PREREVIEW, False), (SENSITIVE, False), (ENC, False), (GMRT, False)},
               "name": INTERNAL},  # HSD planning use
}


@total_ordering
class BrutyExportVersion:
    domain = "Bruty"

    def __init__(self):
        self.db_version = nbs.bruty.world_raster_database.VERSION
        self.combine_version = nbs.scripts.combine.VERSION
        self.export_version = VERSION
        self.commit_id = git_head_commit_id()

    def __gt__(self, other):
        return all([self.db_version > other.db_version,
                    self.combine_version > other.combine_version,
                    self.export_version > other.export_version])

    def __eq__(self, other):
        return all([self.db_version == other.db_version,
                    self.combine_version == other.combine_version,
                    self.export_version == other.export_version])

    def to_dataset(self, ds):
        existing = ds.GetMetadata(self.domain)
        existing.update(self.__dict__)
        for k in existing:
            existing[k] = str(existing[k])
        ds.SetMetadata(existing, self.domain)

    def to_gdal(self, filename):
        ds = gdal.Open(filename, gdal.GA_Update)
        self.to_dataset(ds)

    @classmethod
    def from_dataset(cls, ds):
        data = cls()
        file_dict = ds.GetMetadata(cls.domain)
        if file_dict:
            # convert strings stored in the geotiffs to tuples
            for key in ("db_version", "combine_version", 'export_version'):
                file_dict[key] = eval(file_dict[key])
            data.__dict__.update(file_dict)
        else:
            raise ValueError("Bruty version info not found")
        return data

    @classmethod
    def from_gdal(cls, filename):
        ds = gdal.Open(filename)
        return cls.from_dataset(ds)

    def same_or_newer(self, data):
        try:
            data = self.from_gdal(data)
        except:
            try:
                data = self.from_dataset(data)
            except:
                pass
        ret = False
        if self.db_version >= data.db_version and self.combine_version >= data.combine_version and self.export_version >= data.export_version:
            ret = True
        return ret


class RasterExport:
    COG_SUFFIX = ".tif"
    EXTRA_RAT_SUFFIX = ".aux.xml"
    RAT_SUFFIX = ".tif.aux.xml"
    ORIG_SUFFIX = ".original.tif"
    GENERALIZED_SUFFIX = ".generalized.tif"
    CONTRIB_SUFFIX = ".revise_contrib.tif"
    SCORE_SUFFIX = ".score.tif"

    def __init__(self, parent_dir, tile_info, export_type, data_time="", export_time=""):
        self.output_base_path = pathlib.Path(parent_dir)
        self.tile_info = tile_info
        os.makedirs(self.output_base_path, exist_ok=True)
        self.data_time = data_time
        self.export_time = export_time
        self.type = export_type

    @property
    def type(self):
        return self._export_type

    @type.setter
    def type(self, val):
        self._export_type = val

    def filename_with_type_root(self, export_type) -> pathlib.Path:
        """no export time included"""
        t = self.tile_info
        use_res = t.resolution
        if t.resolution == int(t.resolution):
            use_res = int(t.resolution)
        base = str(self.output_base_path.joinpath(f"{t.pb}{t.utm}{t.hemi}_{t.locality}", export_type, f"Tile{t.tile}_{t.pb}{t.utm}{t.hemi}_{t.locality}_{use_res}m_{export_type}"))
        if self.data_time:
            base += "_" + self.data_time
        use_closing = t.closing_dist
        if t.closing_dist == int(t.closing_dist):
            use_closing = int(t.closing_dist)
        base += f"_{use_closing}m"
        return pathlib.Path(base)

    def filename_with_type(self, export_type: str) -> pathlib.Path:
        base = str(self.filename_with_type_root(export_type))
        if self.export_time:
            base += "_" + self.export_time
        return pathlib.Path(base)

    @property
    def base_filename(self):
        return self.filename_with_type(self.type)

    @property
    def extracted_filename(self):
        return self.base_filename.with_suffix(self.ORIG_SUFFIX)

    @property
    def generalized_filename(self):
        return self.base_filename.with_suffix(self.GENERALIZED_SUFFIX)

    @property
    def contributors_filename(self):
        return self.base_filename.with_suffix(self.CONTRIB_SUFFIX)

    @property
    def cog_filename(self):
        return self.base_filename.with_suffix(self.COG_SUFFIX)

    @property
    def score_filename(self):
        return self.extracted_filename.with_suffix(self.SCORE_SUFFIX)

    @property
    def rat_filename(self):
        return self.cog_filename.with_suffix(self.RAT_SUFFIX)

    def cleanup_tempfiles(self, allow_permission_fail=False, silent=True):
        if not interactive_debug:
            remove_file(self.extracted_filename, allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(self.score_filename, allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(str(self.extracted_filename) + ".aux.xml", allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(str(self.score_filename) + ".aux.xml", allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(self.generalized_filename, allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(self.contributors_filename, allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(self.contributors_filename.with_suffix(".pickle"), allow_permission_fail=allow_permission_fail, silent=silent)

    def cleanup_finalfiles(self, allow_permission_fail=False, silent=True):
        if not interactive_debug:
            remove_file(self.cog_filename, allow_permission_fail=allow_permission_fail, silent=silent)
            remove_file(self.rat_filename, allow_permission_fail=allow_permission_fail, silent=silent)


def make_unreviewed_notes(all_simple_records, tile_info, combine_tiles):
    """ Look through all the records and find the ones that are not reviewed and make a note of them.
    This is used to populate the xbox table with the unreviewed data that is being included or excluded from the export.

    Parameters
    ----------
    all_simple_records
        list of the records from the nbs database
    tile_info
        A tile_info object that has the information about the tile being exported so the tablename can be found
    dtypes_and_for_nav
        list of tuples of (datatype, for_nav) that are to be included in the notes.
        ex: [(REVIEWED, True), (ENC, True), (GMRT, True)]

    Returns
    -------
    str
        A string that has the count of records which were unreviewed and then a list of the unreviewed filenames

    """
    # Find all the records that are not reviewed and make a dictionary of lists of them
    unreviewed = {}
    unreviewed_notes = ""
    IGNORED = 1
    COMBINED = 0
    labels = {IGNORED: "Skipped due to never_post=True", COMBINED: "Included in combined data (never_post=False)"}
    for combine_tile in combine_tiles:
        database_name = combine_tile.metadata_table_name()
        for key, rec in all_simple_records.items():
            if not rec.get('nbs_reviewed', False):  # when simple_records had a null in postgres they end up as a missing key, so treat null as False
                if rec['tablename'] == database_name:
                    if rec.get('for_navigation', False) == combine_tile.for_nav:
                        index = IGNORED if rec.get('never_post', False) else COMBINED
                        table = unreviewed.setdefault((rec['tablename']), [[], []])
                        table[index].append(rec)
    # Make a description of the amount of unreviewed data and then list the survey names
    total_unreviewed = 0
    for tablename, record_groups in unreviewed.items():
        for records in record_groups:
            total_unreviewed += len(records)
    if total_unreviewed:
        unreviewed_notes = f"Unreviewed data found: {total_unreviewed} records (nbs_reviewed=False)\n"
        # Now list the records that are unreviewed, first the ones that are included and then the ones that are skipped
        for tablename, record_groups in unreviewed.items():
            for index in (COMBINED, IGNORED):
                records = record_groups[index]
                if records:
                    unreviewed_notes += f"  {tablename} - {len(records)} records {labels[index]}\n"
                    for record in records:
                        unreviewed_notes += f"    {record['from_filename']}\n"
    return unreviewed_notes


def make_read_only(fname):
    """ Change the read only attribute of a file and ignore any errors

    Parameters
    ----------
    fname
        str or pathlib.Path to file

    Returns
    -------
    None
    """
    try:
        os.chmod(fname, S_IREAD | S_IRGRP | S_IROTH)
    except Exception:
        pass


def get_metadata(combine_tiles, conn_info, use_bruty_cached=False):
    metadata_fields = []
    metadata_records = []
    conn, cursor = connection_with_retries(conn_info)
    for combine_tile in combine_tiles:
        query_database = False
        tablename = combine_tile.metadata_table_name()
        if use_bruty_cached:
            root_dir = pathlib.Path(combine_tile.data_location)
            db_name = root_dir.joinpath(combine_tile.bruty_db_name())
            cache_fname = pathlib.Path(db_name).joinpath(f"last_used_{conn_info.database}_{tablename}.pickle")
            try:
                cache_file = open(cache_fname, "rb")
                mfields = pickle.load(cache_file)
                mrecords = pickle.load(cache_file)
            except:
                query_database = True
        else:
            query_database = True
        if query_database:
            try:
                query_fields = set(SCORING_METADATA_COLUMNS)
                query_fields.update(EXPORT_METADATA_COLUMNS)
                mfields, mrecords = get_nbs_records(tablename, cursor, query_fields=query_fields)
            except psycopg2.errors.UndefinedTable:
                print(f"{tablename} doesn't exist.")
                continue
        metadata_records.append(mrecords)
        metadata_fields.append(mfields)
    # @todo move this record stuff into a function in nbs_postgres.py
    all_meta_records = {}
    all_simple_records = {}
    for n, meta_table_recs in enumerate(metadata_records):
        # id_col = metadata_fields[n].index('nbs_id')
        for record_dict in meta_table_recs:
            # record_dict = dict(zip(metadata_fields[n], record))
            simple_record = meta_review.MetadataDatabase._simplify_record(record_dict)
            # re-casts the db values into other/desired types but then we just want a plain dictionary for pickle
            simple_fuse_record = dict(meta_review.records_to_fusemetadata(simple_record))
            all_simple_records[record_dict['nbs_id']] = simple_fuse_record
        all_meta_records.update({rec['nbs_id']: rec for rec in meta_table_recs})

    sorted_recs, names_list, sort_dict = id_to_scoring(metadata_records, for_navigation_flag=(False, False),
                                                       never_post_flag=(False, False))
    return all_simple_records, sort_dict


def combine_and_export(config, tile_info, decimals=None, use_caches=False):
    # conn_info.database = "metadata"
    conn_info_exports = connect_params_from_config(config)
    conn_info_exports.database = config.get('export_database', None)
    conn_info_exports.tablenames = [config.get('export_table', "")]
    bruty_dir = config['data_dir']
    export_dir = config['export_dir']
    time_format = "%Y%m%d_%H%M%S"
    export_time = datetime.datetime.now().strftime(time_format)

    ret_code = 0
    try:
        # lock the resolution record so we can update the export times
        # Find the spec_combine records
        # lock them (which will also tell if any are in use)
        conn_info = connect_params_from_config(config)
        tile_info.acquire_lock_and_combine_locks(conn_info)  # will raise a BaseLockException if the locks can't be acquired
        # lck = AdvisoryLock(all_paths, conn_info, flags=SHARED | NON_BLOCKING)

        combine_tiles = tile_info.get_related_combine_info(tile_info.sql_obj)
        all_simple_records, sort_dict = get_metadata(combine_tiles, conn_info, use_bruty_cached=use_caches)
        comp = partial(nbs_survey_sort, sort_dict)
        warnings_log = "''"  # single quotes for postgres
        info_log = "''"
        for h in logging.getLogger("nbs").handlers:
            if isinstance(h, logging.FileHandler):
                if f"{os.getpid()}" in h.baseFilename:
                    if ".warnings" in h.baseFilename:
                        warnings_log = f"'{h.baseFilename}'"
                    elif ".log" in h.baseFilename:
                        info_log = f"'{h.baseFilename}'"  # single quotes for postgres

        tile_info.update_table_record(**{tile_info.export.START_TIME: "NOW()", tile_info.export.TRIES: f"COALESCE({tile_info.export.TRIES}, 0) + 1",
                                         tile_info.export.DATA_LOCATION: f"'{export_dir}'", tile_info.export.INFO_LOG: info_log,
                                         tile_info.export.WARNINGS_LOG: warnings_log})
        # 3) Extract data from all necessary Bruty DBs
        if tile_info.public or tile_info.internal or tile_info.navigation:
            fobj, cache_file = tempfile.mkstemp(".cache.tif")
            os.close(fobj)
            databases = []

            # Make an export file for all the databases everything uses (for_nav of qualified/public, enc, gmrt) "original"
            # Make a copy and add things for navigation (sensitive) and complete the export
            # Take original and add not_for_nav of the first datasets (qualified/reviewed, enc, gmrt)
            # Next add unqualified/prereview (both nav and not_for_nav)
            # Export as bluetopo
            # Then add sensitive and export as internal

            # @todo if there is no sensitive then internal and bluetopo are the same (don't export twice)

            # Find the latest data time available so we can tell if an existing export was recent enough
            dataset, dataset_score = None, None
            all_times = []
            root_dir = pathlib.Path(bruty_dir)
            for combine_tile in combine_tiles:
                try:
                    db = WorldDatabase.open(combine_tile.combine.data_location)
                except FileNotFoundError as e:
                    LOGGER.warning(f"Bruty data not found:\n  {combine_tile} at {combine_tile.data_location}")
                else:
                    if dataset is None:
                        dataset, dataset_score = setup_export_raster(cache_file, tile_info, db)
                    all_times.extend([rec.ttime for rec in db.transaction_groups.values() if rec.modified_data])
                    del db
            if not dataset:
                raise FileNotFoundError(f"No bruty data was found under{root_dir}")

            time_format = "%Y%m%d_%H%M%S"
            if all_times:
                last_time = max(all_times)
                timestamp = last_time.strftime(time_format)
            else:
                timestamp = ""

            # Check the disk and see if all desired exports already exist - if any fail then remake the whole set of requested data
            find_export = RasterExport(export_dir, tile_info, NAVIGATION, timestamp)
            current_version = BrutyExportVersion()
            done = []
            # Check all files with any export time listed in its name and see if it was made with the current Bruty algorithms
            for exp, wanted, kwargs in ((NAVIGATION, tile_info.navigation, {'navigation': True}),
                                        (PUBLIC, tile_info.public, {'public': True}),
                                        (INTERNAL, tile_info.internal, {'internal': True})):
                success = False
                if wanted:
                    # FIXME - instead of checking the export directory we should check all the export table records for the nav/internal/public=True (as appropriate) and see if the files match the pattern of the name (not the folder so much as nav might be used for all 3 exports)
                    # note, there should only be one tablename but for completeness we'll allow for multiples
                    for tablename, recs in zip(conn_info_exports.tablenames, get_tile_records(conn_info_exports, tile_info, select="id,data_location", combine_time=timestamp, **kwargs)):
                        for rec_id, possible_file in recs:
                            if os.path.exists(possible_file):
                                try:
                                    # If this fails, we don't care - just check remaining files
                                    stored_version = BrutyExportVersion.from_gdal(possible_file)
                                except:  # file didn't load or no bruty version info was in it
                                    pass  # keep looking
                                else:  # the version info is in the file
                                    if current_version >= stored_version and os.path.exists(possible_file + find_export.EXTRA_RAT_SUFFIX):
                                        success = (exp, possible_file, rec_id)
                                        break
                else:  # not wanted
                    continue
                done.append(success)

            if all(done):
                LOGGER.info(f"Already exported - {timestamp} data located at:")
                del dataset, dataset_score  # release the file handle for the score file
                for exp, loc, exp_rec_id in done:
                    LOGGER.info(f"  {exp}  --  {loc}")
                    needed_tiles = set([tile for tile in combine_tiles if (tile.datatype, tile.for_nav) in export_definitions[exp]['datasets']])
                    unreviewed_notes = make_unreviewed_notes(all_simple_records, tile_info, needed_tiles)
                    # update the notes fields in case the nbs_reviewed column has changed which wouldn't change the actual data
                    update_export_record(conn_info_exports, exp_rec_id, notes=unreviewed_notes)
            else:
                # re-export in case the previous export didn't finish
                nav_export = RasterExport(export_dir, tile_info, NAVIGATION, timestamp, export_time)
                internal_export = RasterExport(export_dir, tile_info, INTERNAL, timestamp, export_time)
                public_export = RasterExport(export_dir, tile_info, PUBLIC, timestamp, export_time)

                for exp, wanted in ((nav_export, tile_info.navigation), (internal_export, tile_info.internal), (public_export, tile_info.public)):
                    if wanted:
                        exp.cleanup_tempfiles()
                        exp.cleanup_finalfiles()
                combines_used = set()
                needed_tiles = set([tile for tile in combine_tiles if (tile.datatype, tile.for_nav) in export_definitions[NAVIGATION]['datasets']])
                add_tiles = needed_tiles - combines_used
                base_cnt = add_databases(add_tiles, dataset, dataset_score, comp)
                combines_used.update(add_tiles)

                if tile_info.navigation:
                    os.makedirs(nav_export.extracted_filename.parent, exist_ok=True)
                    nav_dataset = copy_dataset(dataset, nav_export.extracted_filename)
                    nav_score = copy_dataset(dataset_score, nav_export.score_filename)
                    # databases = [tile_info.bruty_db_name(SENSITIVE, True)]
                    # databases = [root_dir.joinpath(db_name) for db_name in databases]
                    # nav_sensitive_cnt = add_databases(databases, nav_dataset, nav_score, comp)
                    del nav_score, nav_dataset  # closes the files so they can be copied
                    complete_export(nav_export, all_simple_records, tile_info.closing_dist, tile_info.epsg, decimals=decimals)
                    unreviewed_notes = make_unreviewed_notes(all_simple_records, tile_info, combines_used)
                    navigation_id = write_export_record(conn_info_exports, nav_export, navigation=True, notes=unreviewed_notes)

                if tile_info.public or tile_info.internal:
                    # @todo the counts returned by add_databases should allow to tell if any changes were added
                    #   or possibly the same export would apply to multiple configurations, like public and internal are the same
                    # add the not for navigation versions that were already added
                    # also add the prereview (unqualified data)
                    needed_tiles = set([tile for tile in combine_tiles if (tile.datatype, tile.for_nav) in export_definitions[PUBLIC]['datasets']])
                    add_tiles = needed_tiles - combines_used
                    if export_definitions[NAVIGATION]['datasets'] - export_definitions[PUBLIC]['datasets']:
                        raise ValueError("PUBLIC export must include all NAVIGATION datasets")
                    # [(REVIEWED, False), (ENC, False), (GMRT, False), (PREREVIEW, True), (PREREVIEW, False)]
                    prereview_cnt = add_databases(add_tiles, dataset, dataset_score, comp)
                    combines_used.update(add_tiles)

                    if tile_info.public:
                        os.makedirs(public_export.extracted_filename.parent, exist_ok=True)
                        unreviewed_notes = make_unreviewed_notes(all_simple_records, tile_info, combines_used)
                        if prereview_cnt == 0 and nav_export.cog_filename.exists():
                            try:
                                os.symlink(nav_export.cog_filename, public_export.cog_filename)
                                os.symlink(nav_export.rat_filename, public_export.rat_filename)
                            except OSError:
                                LOGGER.warning("Symlinks failed, only the Navigation file will exist, public will not")
                            update_export_record(conn_info_exports, navigation_id, public=True, notes=unreviewed_notes)
                            public_id = navigation_id
                        else:
                            copy_dataset(dataset, public_export.extracted_filename)
                            complete_export(public_export, all_simple_records, tile_info.closing_dist, tile_info.epsg, decimals=decimals)
                            public_id = write_export_record(conn_info_exports, public_export, public=True, notes=unreviewed_notes)

                    if tile_info.internal:
                        os.makedirs(internal_export.extracted_filename.parent, exist_ok=True)
                        # internal is the same as public with SENSITIVE added.
                        needed_tiles = set([tile for tile in combine_tiles if (tile.datatype, tile.for_nav) in export_definitions[INTERNAL]['datasets']])
                        add_tiles = needed_tiles - combines_used
                        if export_definitions[PUBLIC]['datasets'] - export_definitions[INTERNAL]['datasets']:
                            raise ValueError("INTERNAL export must include all PUBLIC datasets")
                        sensitive_cnt = add_databases(add_tiles, dataset, dataset_score, comp)
                        combines_used.update(add_tiles)
                        unreviewed_notes = make_unreviewed_notes(all_simple_records, tile_info, combines_used)
                        if sensitive_cnt == 0 and prereview_cnt == 0 and nav_export.cog_filename.exists():
                            try:
                                os.symlink(nav_export.cog_filename, internal_export.cog_filename)
                                os.symlink(nav_export.rat_filename, internal_export.rat_filename)
                            except OSError:
                                LOGGER.warning("Symlinks failed, only the Navigation file will exist, internal will not")
                            update_export_record(conn_info_exports, navigation_id, internal=True, notes=unreviewed_notes)
                            internal_id = navigation_id
                        elif sensitive_cnt == 0 and public_export.cog_filename.exists():
                            try:
                                os.symlink(public_export.cog_filename, internal_export.cog_filename)
                                os.symlink(public_export.rat_filename, internal_export.rat_filename)
                            except OSError:
                                LOGGER.warning("Symlinks failed, only the public file will exist, internal will not")
                            update_export_record(conn_info_exports, public_id, internal=True, notes=unreviewed_notes)
                            internal_id = public_id
                        else:
                            copy_dataset(dataset, internal_export.extracted_filename)
                            complete_export(internal_export, all_simple_records, tile_info.closing_dist, tile_info.epsg, decimals=decimals)
                            internal_id = write_export_record(conn_info_exports, internal_export, internal=True, notes=unreviewed_notes)

                for exp, wanted in ((nav_export, tile_info.navigation), (internal_export, tile_info.internal), (public_export, tile_info.public)):
                    if wanted:
                        exp.cleanup_tempfiles(allow_permission_fail=True)
                del dataset, dataset_score  # release the file handle for the score file
                for fname in (nav_export.cog_filename, internal_export.cog_filename,public_export.cog_filename):
                    make_read_only(fname)
            remove_file(cache_file, allow_permission_fail=True, limit=4, tdelay=15)
            remove_file(cache_file.replace(".tif", ".score.tif"), allow_permission_fail=True, limit=4, tdelay=15)
    except BaseLockException:
        LOGGER.warning(f"Could not get locks for {tile_info}")
        ret_code = TILE_LOCKED
    except Exception as e:
        traceback.print_exc()
        msg = f"{tile_info.full_name} had an unhandled exception - see message above"
        print(msg)
        LOGGER.error(traceback.format_exc())
        LOGGER.error(msg)
        ret_code = UNHANDLED_EXCEPTION
    try:
        tile_info.update_table_record(**{tile_info.export.END_TIME: "NOW()", tile_info.export.EXIT_CODE: ret_code})
        tile_info.release_lock()
    except:
        pass
    return ret_code


def copy_dataset(dataset, filename):
    # FlushCache isn't gauranteed to write the file and so I don't trust FlushCache() + shutil.copy
    # so do a gdal CreateCopy then close the dataset and then run the rest of the export function
    driver = dataset.GetDriver()
    new_dataset = driver.CreateCopy(str(filename), dataset,
                                    options=("BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"))
    return new_dataset


def add_databases(combine_tiles, dataset, dataset_score, comp):
    cnt = 0
    with tqdm(desc="merging databases", total=len(combine_tiles)) as progress:
        for combine_tile in combine_tiles:
            db_name = combine_tile.combine.data_location
            progress.set_description("merging "+str(pathlib.Path(db_name).name))
            try:
                db = WorldDatabase.open(db_name)
                try:
                    cnt += db.export_into_raster(dataset, dataset_score, compare_callback=comp)
                except KeyError as e:
                    print("KeyError", str(e))
                    # open(export.output_base_path.joinpath(export.fname_base+f" - keyerror {str(e)}"), "w")
                    raise BrutyMissingScoreError(str(e))
            except FileNotFoundError:
                LOGGER.warning(f"Bruty data not found:\n  {db_name}")
            progress.update()

    dataset.FlushCache()
    dataset_score.FlushCache()
    return cnt


def setup_export_raster(filename, tile_info, db):
    """ Given filenames in export object, tile specs in tile_info and bruty database in db,
    create and return two gdal dataset objects - one for the depth/uncertainty and one for the scores.
    """
    crs_transform = get_crs_transformer(tile_info.epsg, db.db.epsg)
    minx, maxx = tile_info.minx, tile_info.maxx
    miny, maxy = tile_info.miny, tile_info.maxy
    if crs_transform:
        xs, ys = [minx, maxx], [miny, maxy]
        cx, cy = crs_transform.transform(xs, ys)
        minx = min(cx)
        maxx = max(cx)
        miny = min(cy)
        maxy = max(cy)
    wkt = make_mllw_height_wkt(db.db.tile_scheme.epsg)
    res = tile_info.resolution  # tile_record[resolution_index]
    closing_dist = tile_info.closing_dist
    # center the output cells at origin of UTM like Coast Survey standard -- this will align with Coast Survey Bruty tiles
    # basically a cell center would fall at 0,0 of the coordinate system
    use_minx = ((minx - closing_dist) // res) * res - res / 2.0
    use_miny = ((miny - closing_dist) // res) * res - res / 2.0
    # don't use edge alignment as we want the centers to line up.
    # @todo change make_export_rasters align parameter to support none/center/edge (currently matches edge alignment)
    dataset, dataset_score = db.make_export_rasters(filename, use_minx, use_miny,
                                                    maxx + closing_dist, maxy + closing_dist, res, align=None)
    dataset.SetProjection(wkt)
    return dataset, dataset_score


def complete_export(export, all_simple_records, closing_dist, epsg, decimals=None):
    # return complete_export_sequential(export, all_simple_records, closing_dist, epsg, decimals)
    return complete_export_tiled(export, all_simple_records, closing_dist, epsg, decimals)

if has_numba:
    @jit(nopython=True)
    def replace_nodata(src, dst, reduction_factor, nodata=numpy.nan):
        src_col = 0
        src_row = 0
        # numba appears to allocate memory for every range call, so making an object here instead of inside the loop
        cols = range(dst.shape[1])
        for row in range(dst.shape[0]):
            src_row = row // reduction_factor
            for col in cols:
                src_col = col // reduction_factor
                replace = False
                if not numpy.isnan(nodata):
                    if dst[row, col] == nodata:
                        replace = True
                else:
                    if numpy.isnan(dst[row, col]):
                        replace = True
                if replace:
                    dst[row, col] = src[src_row, src_col]
else:
    raise ModuleNotFoundError("Numba not found, need to re-implement a non-numba version of this function")
    # def replace_nodata(src, dst, reduction_factor, nodata=numpy.nan):

@jit(nopython=True)
def reduce_array_in_place(src, dst, reduction_factor, row_offset, col_offset, nodata=numpy.nan):
    """ Reduce an array to a lower resolution - similar to a resample except the max value for a cell is retained.

    Also to make the reduced arrays align with reduced arrays produced from other blocks within a gdal iteration,
    can supply a col_offset and row_offset.  So if all cells are desired to align to the 0,0 cell of the original image and
    the current src is not modulo reduction_factor from the origin then pass in that modulo number.  For example, if a 1000x1000 image is being
    reduced by a factor of 10 to make a final 100x100 but being iterater by 512 for read speed.
    The first chunk would be from 0,0 to 511,511 and would have a zero offset.
    The second chunk would be from 512,0 to 1000,511 and would need a row_offset=2 and col_offset=0
    The third chunk would be from 0,512 to 511,1000 and would need a row_offset=0 and col_offset=2
    The fourth chunk would be from 512,512 to 1000,1000 and would need a row_offset=2 and col_offset=2

    Parameters
    ----------
    src
        full res source array
    dst
        reduced resolution destination array
    reduction_factor
        number of pixels to cell to one cell
    col_offset
        to make reduced array align, supply a modulo offset (i.e. number less than the reduction_factor) to shift by
    row_offset
        to make reduced array align, supply a modulo offset (i.e. number less than the reduction_factor) to shift by
    nodata
        numpy.nan or the bag value of 1000000 are most common

    Returns
    -------
    array
        the reduced resolution array
    """
    # numba appears to allocate memory for every range call, so making an object here instead of inside the loop
    cols = range(dst.shape[1])
    for row in range(dst.shape[0]):
        for col in cols:
            src_row = row * reduction_factor - row_offset
            if src_row < 0:
                src_row = 0
            src_col = col * reduction_factor - col_offset
            if src_col < 0:
                src_col = 0
            sub_data = src[src_row: row * reduction_factor - row_offset + reduction_factor,
                           src_col: col * reduction_factor - col_offset + reduction_factor]
            # change nodata to nans so nanmax works and a nodata value doesn't affect max computation
            if not numpy.isnan(nodata):
                sub_data = sub_data.copy()
                for i in range(reduction_factor):
                    for j in range(reduction_factor):
                        try:
                            if sub_data[i, j] == nodata:
                                sub_data[i, j] = numpy.nan
                        except:
                            pass  # the end sub_data may not have all elements
                        # sub_data[sub_data == nodata] = numpy.nan
            dst[row, col] = numpy.nanmax(sub_data)
            # now change nodata back to correct value, if needed
            if not numpy.isnan(nodata):
                if numpy.isnan(dst[row, col]):
                    dst[row, col] = nodata

@jit(nopython=True)
def replace_high_res_nodata_with_low_res(row, col, high_res_array, clipped_rows, clipped_cols, low_res_array, offset_row, offset_col, reduction, nodata):
    replace = False
    if numpy.isnan(nodata):
        if numpy.isnan(high_res_array[row, col]):
            replace = True
    else:
        if high_res_array[row, col] == nodata:
            replace = True
    if replace:
        high_res_array[row, col] = low_res_array[int((row + clipped_rows + offset_row) / reduction), int((col + clipped_cols + offset_col) / reduction)]


@jit(nopython=True)
def replace_borders(high_res_array, clipped_rows, clipped_cols, low_res_array, offset_row, offset_col, reduction, nodata):
    cols = numpy.arange(high_res_array.shape[1], dtype=numpy.int64)
    high_col = cols[-1]
    rows = numpy.arange(high_res_array.shape[0], dtype=numpy.int64)
    high_row = rows[-1]
    # replace the high_res_array nodata values with the low_res_array equivalents
    # revise the left and right edge
    for col in (0, high_col):
        for row in rows:
            replace_high_res_nodata_with_low_res(row, col, high_res_array, clipped_rows, clipped_cols, low_res_array, offset_row, offset_col, reduction, nodata)

    # revise the top and bottom edge
    for row in (0, high_row):
        for col in cols:
            replace_high_res_nodata_with_low_res(row, col, high_res_array, clipped_rows, clipped_cols, low_res_array, offset_row, offset_col, reduction, nodata)


def reduce_array(src, reduction_factor, ir, ic, nodata=numpy.nan):
    col_offset = ic % reduction_factor
    row_offset = ir % reduction_factor
    red_rows = int(numpy.ceil((src.shape[0] + row_offset) / reduction_factor))
    red_cols = int(numpy.ceil((src.shape[1] + col_offset)/ reduction_factor))
    coarse_array = numpy.zeros([red_rows, red_cols])
    reduce_array_in_place(src, coarse_array, reduction_factor, row_offset, col_offset, nodata=nodata)
    return row_offset, col_offset, coarse_array


def create_subdataset(ds, filename_ext, ir, ic, array, nodata, names=(ELEVATION_BAND_NAME, UNCERTAINTY_BAND_NAME, CONTRIBUTOR_BAND_NAME)):
    p = pathlib.Path(ds.GetFileList()[0])
    filename = p.stem + "." + filename_ext + p.suffix
    full_path = str(p.with_name(filename))
    if len(array.shape) == 3:
        bands, rows, cols = array.shape
    else:
        bands = 1
        rows, cols = array.shape
    new_ds = ds.GetDriver().Create(full_path, xsize=cols, ysize=rows, bands=bands, eType=gdal.GDT_Float32, options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
    new_ds.SetProjection(ds.GetProjection())
    x1, resx, dxy, y1, dyx, resy = ds.GetGeoTransform()
    x1 += resx * ic
    y1 += ir * resy
    new_ds.SetGeoTransform((x1, resx, dxy, y1, dyx, resy))
    for n in range(bands):
        band = new_ds.GetRasterBand(n + 1)
        if bands == 1:
            band.WriteArray(array)
        else:
            band.WriteArray(array[n])
        band.SetNoDataValue(nodata)
        band.SetDescription(names[n])
        try:
            band.ComputeStatistics(0)
        except (ValueError, RuntimeError):
            LOGGER.warning("Could not compute statistics, empty data?")
        del band
    new_ds.FlushCache()
    return new_ds


def complete_export_tiled(export, all_simple_records, closing_dist, epsg, decimals=None, block_size=DEFAULT_BLOCK_SIZE, debug_plots=False):
    """ for the export object, take the extracted filename and perform generalization, cloud overviews and consolidate the raster attribute table
    all_simple_records is a list of the postgres metadata records for operating on the raster attributes.
    closing_dist is used in generalization.
    epsg is the target epsg used in generalization.
    """
    # FIXME - remove this change of nodata value once NBS is fixed to accept NaN in addition to 1000000
    #   also this give a problem in the HACK of storing large int32 in float32 tiffs
    #   the 1000000 becomes 1232348160 when translated back, so a contributor of 1232348160 would get stepped on by NaN
    #   numpy.frombuffer(numpy.array(1000000).astype(numpy.float32).tobytes(), numpy.int32)
    LOGGER.debug("change nodata value (Caris doesn't like NAN)")
    shutil.copyfile(export.extracted_filename, export.generalized_filename)
    elevation_band_index = raster_band_name_index(str(export.generalized_filename), ELEVATION_BAND_NAME)
    uncertainty_band_index = raster_band_name_index(str(export.generalized_filename), UNCERTAINTY_BAND_NAME)
    contributor_band_index = raster_band_name_index(str(export.generalized_filename), CONTRIBUTOR_BAND_NAME)
    original_ds = gdal.Open(str(export.extracted_filename))
    generalized_ds = gdal.Open(str(export.generalized_filename), gdal.GA_Update)
    raster_transform = generalized_ds.GetGeoTransform()
    resolution = abs(max(raster_transform[1:3]))

    new_nodata = 1000000
    bands = (elevation_band_index, uncertainty_band_index, contributor_band_index)
    image_ops = BufferedImageOps(original_ds)

    # iterate the contributor index since we are modifying the data as we go
    # so checking elevation==nodata will fail as we modify what becomes the buffer area of the next block
    # so check contributor for being non-nan and non-generalization
    # read about 1GB at a time
    for b in bands:
        band = generalized_ds.GetRasterBand(b)
        band.SetNoDataValue(new_nodata)
        del band
    generalized_ds.GetRasterBand(elevation_band_index).SetDescription(ELEVATION_BAND_NAME.capitalize())
    generalized_ds.GetRasterBand(uncertainty_band_index).SetDescription(UNCERTAINTY_BAND_NAME.capitalize())
    generalized_ds.GetRasterBand(contributor_band_index).SetDescription(CONTRIBUTOR_BAND_NAME.capitalize())
    # make a copy of the dictionary so we don't modify the caller's data, or while converting the contributor numbers make a new dict
    orig_simple_records = all_simple_records
    all_simple_records = {}
    old_to_new_mapping = {0: 0.0}  # add the generalized contributor since we know the conversion and there is no matching orig_simple_record
    inew = 1

    buff = closing_dist / resolution
    LOGGER.debug("iterate blocks to generalize data")
    for block_cnt, block in enumerate(image_ops.iterate_gdal(buff, buff, band_nums=bands, min_block_size=block_size, max_block_size=block_size)):
        # if True or not (block_cnt==0 or block_cnt==1 or (block_cnt >= 4 and block_cnt <= 7)):
        #     continue
        with tqdm(desc="processing steps", total=4+5, leave=False) as progress:  # 5 steps in the generalize code
            ic, ir, cols, rows, col_buffer_lower, row_buffer_lower, nodata, data = block
            if debug_plots:
                outline = numpy.zeros([rows, cols], dtype=numpy.float32)
                outline.fill(numpy.nan)
                outline[0:100, :] = 1
                outline[-101:, :] = 1
                outline[:, 0:100] = 1
                outline[:, -101:] = 1
                temp_ds = create_subdataset(original_ds, f"{block_cnt}_outline", ir, ic, outline, numpy.nan);
                del temp_ds
            has_data = False
            elevation_array, uncert_array, contrib_array = data
            # change the nodata value and write back to the file
            progress.set_description("Change nodata")
            for band_num, subarray in zip(bands, data):
                subarray[numpy.isnan(subarray)] = new_nodata
                if band_num == elevation_band_index:
                    has_data = has_data or where_not_nodata(subarray, new_nodata).any()
            progress.update(1)
            if has_data and closing_dist > 0:
                # FIXME - HACK  we should send a float version of the contributor number since we have a hack
                #   of storing int32 inside the float32 tiff, but zero is the same in both so we can get away with it for now
                # numpy.array(0, dtype=numpy.float32).tobytes()
                # Out[7]: b'\x00\x00\x00\x00'
                # numpy.array(0, dtype=numpy.int32).tobytes()
                # Out[8]: b'\x00\x00\x00\x00'

                # LOGGER.info(f'performing generalized interpolation on "{raster_filename}"')
                # LOGGER.info(f'Using a closing distance of {closing_distance} meters for {raster_filename}')
                reduction = 8  # low res cell size is N times normal res
                reduced_buffer_size = int(block_size / 4)  # Buffer size as ratio to to normal block size
                if buff > reduced_buffer_size:

                    # Make a raster that is N times smaller and generalize that to supply rough values.
                    # generalize the whole coarse (downsampled) array.
                    # generalize just the block of data with a reduced buffer around it (retain some better interpolation for data that was close).
                    # fill any remaining empty space in the full res output array with the interpolation from the coarse array computation.

                    # numpy doesn't appear to have a downsample technique that allow selected min/max for a square,
                    # we could just skip using range() with steps but might miss the actual data points
                    # Option 1, make two gdal datasets and use Warp with resampling_method
                    # Option 2, use numba to loop quickly
                    USE_NUMBA = has_numba
                    USE_WARP = not USE_NUMBA
                    if USE_WARP:
                        red_rows = int(numpy.ceil(elevation_array.shape[0] / reduction))
                        red_cols = int(numpy.ceil(elevation_array.shape[1] / reduction))
                        geotransform = list(generalized_ds.GetGeoTransform())
                        mem_ds = gdal.GetDriverByName('MEM').Create('', elevation_array.shape[1] , elevation_array.shape[0], len(bands), gdal.GDT_Float32)
                        mem_ds.SetGeoTransform((0, geotransform[1], 0, 0, 0, geotransform[5]))
                        band = mem_ds.GetRasterBand(1)
                        band.WriteArray(elevation_array)
                        reduced_ds = gdal.GetDriverByName('MEM').Create('', red_cols, red_rows, len(bands), gdal.GDT_Float32)
                        reduced_ds.SetGeoTransform((0, geotransform[1] * reduction, 0, 0, 0, geotransform[5] * reduction))
                        for band_num in bands:
                            mem_ds.GetRasterBand(band_num).SetNoDataValue(new_nodata)
                            reduced_ds.GetRasterBand(band_num).SetNoDataValue(new_nodata)
                        gdal.Warp(reduced_ds, mem_ds, resampleAlg="max")
                        coarse_array = reduced_ds.ReadAsArray()
                        coarse_elev, coarse_uncert, coarse_contrib = coarse_array
                    if USE_NUMBA:
                        row_offset, col_offset, coarse_elev = reduce_array(elevation_array, reduction, ir - row_buffer_lower, ic - col_buffer_lower, nodata=new_nodata)
                        row_offset, col_offset, coarse_uncert = reduce_array(uncert_array, reduction, ir - row_buffer_lower, ic - col_buffer_lower, nodata=new_nodata)
                        row_offset, col_offset, coarse_contrib = reduce_array(contrib_array, reduction, ir - row_buffer_lower, ic - col_buffer_lower, nodata=new_nodata)
                        if debug_plots:
                            coarse_ds = create_subdataset(original_ds, f"{block_cnt}_raw_coarse", ir - row_buffer_lower, ic - col_buffer_lower,
                                                      coarse_elev, new_nodata)
                            coarse_geo = list(coarse_ds.GetGeoTransform())
                            # shift the local origin so it lines up with the global origin of the overall image
                            coarse_geo[0] -= ((ic - col_buffer_lower) % reduction) * coarse_geo[1]
                            coarse_geo[3] -= ((ir - row_buffer_lower) % reduction) * coarse_geo[5]
                            # increase the pixel size for the reduced image
                            coarse_geo[1] *= reduction
                            coarse_geo[5] *= reduction
                            coarse_ds.SetGeoTransform(coarse_geo)
                            del coarse_ds
                    # compute the low res answer
                    coarse_dist_array = generalize_tile(coarse_elev, coarse_uncert, coarse_contrib,
                                                 new_nodata, closing_dist, resolution * reduction, ext_progress=None)
                    if debug_plots:
                        temp_ds = create_subdataset(original_ds, f"{block_cnt}_raw", ir - row_buffer_lower, ic - col_buffer_lower,
                                                    elevation_array, new_nodata); del temp_ds
                        coarse_ds = create_subdataset(original_ds, f"{block_cnt}_coarse", ir - row_buffer_lower, ic - col_buffer_lower,
                                                      coarse_elev, new_nodata)
                        coarse_geo = list(coarse_ds.GetGeoTransform())
                        # shift the local origin so it lines up with the global origin of the overall image
                        coarse_geo[0] -= ((ic - col_buffer_lower) % reduction) * coarse_geo[1]
                        coarse_geo[3] -= ((ir - row_buffer_lower) % reduction) * coarse_geo[5]
                        # increase the pixel size for the reduced image
                        coarse_geo[1] *= reduction
                        coarse_geo[5] *= reduction
                        coarse_ds.SetGeoTransform(coarse_geo)
                        del coarse_ds
                    # shrink the buffer size and compute the interpolation for the remaining high res data
                    low_row_buff, low_col_buff, trimmed_elev = image_ops.trim_buffer(elevation_array, reduced_buffer_size)
                    low_row_buff, low_col_buff, trimmed_uncert = image_ops.trim_buffer(uncert_array, reduced_buffer_size)
                    low_row_buff, low_col_buff, trimmed_contrib = image_ops.trim_buffer(contrib_array, reduced_buffer_size)
                    # compute the edge indices of the trimmed data
                    low_row = row_buffer_lower - low_row_buff
                    high_row = low_row + trimmed_elev.shape[0]
                    low_col = col_buffer_lower - low_col_buff
                    high_col = low_col + trimmed_elev.shape[1]
                    # if debug_plots:
                    #     _temp_elev, _temp_uncert, _temp_contrib = trimmed_elev.copy(), trimmed_uncert.copy(), trimmed_contrib.copy()
                    #     _dist_array = generalize_tile(_temp_elev, _temp_uncert, _temp_contrib,
                    #                                          new_nodata, closing_dist, resolution, ext_progress=progress)
                    #     temp_ds = create_subdataset(original_ds, f"{block_cnt}_prior_res", ir - low_row_buff, ic - low_col_buff,
                    #                                 trimmed_elev, new_nodata); del temp_ds
                    #     del _temp_elev, _temp_uncert, _temp_contrib, _dist_array

                    replace_borders(trimmed_elev, low_row, low_col, coarse_elev, row_offset, col_offset, reduction, new_nodata)
                    if debug_plots:
                        temp_ds = create_subdataset(original_ds, f"{block_cnt}_borders", ir - low_row_buff, ic - low_col_buff,
                                                    trimmed_elev, new_nodata); del temp_ds

                    trimmed_dist_array = generalize_tile(trimmed_elev, trimmed_uncert, trimmed_contrib, new_nodata,
                                                         closing_dist, resolution, ext_progress=progress)
                    if debug_plots:
                        temp_ds = create_subdataset(original_ds, f"{block_cnt}_hi_res", ir - low_row_buff, ic - low_col_buff,
                                                    trimmed_elev, new_nodata); del temp_ds
                    # copy the trimmed data into the original array
                    # FIXME - make a unit test for this behavior or just always copy (which is a small waste)
                    # Because the numpy array is a view the generalized data goes right back into the original data -
                    #   if the trim_buffer function returns a copy then we would need to copy the data from the trimmed to the original data
                    elevation_array[low_row:high_row, low_col:high_col] = trimmed_elev
                    uncert_array[low_row:high_row, low_col:high_col] = trimmed_uncert
                    contrib_array[low_row:high_row, low_col:high_col] = trimmed_contrib
                    dist_array = numpy.zeros(contrib_array.shape)
                    dist_array.fill(new_nodata)
                    dist_array[low_row:high_row, low_col:high_col] = trimmed_dist_array

                    # Fill any gaps with the low res data
                    replace_nodata(coarse_elev, elevation_array, reduction, new_nodata)
                    replace_nodata(coarse_uncert, uncert_array, reduction, new_nodata)
                    replace_nodata(coarse_contrib, contrib_array, reduction, new_nodata)
                    replace_nodata(coarse_dist_array, dist_array, reduction, new_nodata)
                    if debug_plots:
                        temp_ds = create_subdataset(original_ds, f"{block_cnt}_hi_low", ir - low_row_buff, ic - low_col_buff,
                                                    trimmed_elev, new_nodata); del temp_ds

                else:
                    dist_array = generalize_tile(elevation_array, uncert_array, contrib_array, new_nodata,
                                                 closing_dist, resolution, ext_progress=progress)
                dist_array = image_ops.trim_array(dist_array)
            else:
                progress.update(5)

            # NOTE - this is important since we are changing data size!!!!!!!
            # Trim the buffers off the data as we are done with the generalization and distances
            for i, band_array in enumerate(data):
                data[i] = image_ops.trim_array(band_array)
            elevation_array, uncert_array, contrib_array = data

            # LOGGER.debug("Copy to 'contribs' filename to start modifying the contributor from float to int")
            progress.set_description("Decimal precision")
            if decimals is not None:
                # LOGGER.debug("Limit decimal precision")
                decimal_multiplier = 10**decimals
                # do operations in place to save memory and make writing back out from data[] work
                numpy.ceil(elevation_array * decimal_multiplier, out=elevation_array)
                elevation_array /=  decimal_multiplier
                numpy.ceil(uncert_array * decimal_multiplier, out=uncert_array)
                uncert_array /= decimal_multiplier
            progress.update(1)
            # LOGGER.debug("Consolidate contributor IDs")

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

            # # @TODO - couldn't this convert all the floats to ints and then recover the nan or 1000000
            # #    by looking at the original float data?
            # #    Two loops would be faster than looping the data N times
            # #    However, since we are now also consolidating the values as 0,1,2,3,4... it needs a loop.

            progress.set_description("Contributor IDs")
            icontrib = contrib_array.copy()  # make a copy of the data in case we ever had a value in the float version equaled something mapped to an int
            contribs = numpy.unique(contrib_array)
            int_contribs = numpy.sort(contributor_float_to_int(contribs)).tolist()
            # leave the 1000000 value (Nan replacement) in the contributor space
            try:
                int_contribs.remove(contributor_float_to_int(1000000))
            except ValueError:
                pass
            # if not int_contribs:  # there is no data in this tile - make a fake contributor list so it continues on.
            #     int_contribs = [0]
            # if int_contribs[0] != 0:  # make sure there is a zero entry for the generalization so enumerate below doesn't move a real contributor into the zero spot
            #     int_contribs = [0] + int_contribs
            # @TODO - couldn't this convert all the floats to ints and then recover the nan or 1000000
            #    by looking at the original float data?
            #    Two loops would be faster than looping the data N times
            #    However, since we are now also consolidating the values as 0,1,2,3,4... it needs a loop.
            for iold in int_contribs:
                iold_as_float = contributor_int_to_float(iold)
                if iold not in old_to_new_mapping:
                    old_to_new_mapping[iold] = inew
                    if inew != 0:  # the generalization has no change from int to float and would cause a keyerror in all_simple_records
                        all_simple_records[inew] = orig_simple_records[iold]
                    inew += 1
                if iold_as_float != old_to_new_mapping[iold]:  # don't replace zero with zero
                    icontrib[contrib_array == iold_as_float] = old_to_new_mapping[iold]

            progress.update(1)
            progress.set_description("Write tile to tif")
            for band_num, band_array in zip(bands, data):
                band = generalized_ds.GetRasterBand(band_num)
                if band_num == contributor_band_index:
                    band_array = icontrib
                image_ops.write_array(band_array, band)
                del band
            progress.update(1)

    del original_ds
    # regenerate the stats since we replaced the floats with integers
    # Set the statistics in the exported data
    try:
        # generalized_contrib_band.SetStatistics(float(0), float(inew), float(inew/2), float(numpy.std(range(inew))))
        generalized_ds.GetRasterBand(contributor_band_index).ComputeStatistics(0)
        generalized_ds.GetRasterBand(elevation_band_index).ComputeStatistics(0)
        generalized_ds.GetRasterBand(uncertainty_band_index).ComputeStatistics(0)
        # band = ds.GetRasterBand(2)
        # m = band.GetMetadata()
        # m['STATISTICS_VALID_PERCENT'] = '50'
        # band.SetMetadata(m)
    except (ValueError, RuntimeError):
        LOGGER.warning("Could not compute statistics, empty data?")

    del image_ops

    LOGGER.debug("Make initial Cloud Optimized Geotiff")
    cogdriver = gdal.GetDriverByName("COG")
    # from xipe_dev.xipe.deliverables import CLOUD_OPTIMIZED_GEOTIFF_CREATION_OPTIONS
    #     'PREDICTOR': 3,  # floating point predictor, change this to 2 for horizontal if using an integer tiff
    # use NEAREST to eliminate some artifacts from cubic overview at nodata gaps and
    # also reduces file size by keeping the 2 decimal values from nearest rather than averages full res data
    cog_ds = cogdriver.CreateCopy(str(export.cog_filename), generalized_ds, 0, options=['TILED=YES', 'PREDICTOR=3', 'RESAMPLING=NEAREST', 'OVERVIEW_RESAMPLING=NEAREST', 'COMPRESS=LZW', "BIGTIFF=YES", 'OVERVIEWS=IGNORE_EXISTING'])
    # Add the Bruty metadata version here.  Adding after the COG overviews are added genereated a warning by GDAL but CreateCopy didn't retain the metadata.
    current_ver = BrutyExportVersion()
    current_ver.to_dataset(cog_ds)

    # # the overview of the contributor needs 'nearest' (otherwise it interpolates to non-contributor numbers like 15.65)
    # # which is what paletted data uses, while elevation and uncertainty needs "cubic" to display best (see fliers)
    # layers = [cog_ds.GetRasterBand(b + 1).GetDescription().lower() for b in range(cog_ds.RasterCount)]
    # if LayersEnum.CONTRIBUTOR.name.lower() in layers:
    #     LOGGER.debug("Replace contributor overview in Cloud Optimized Geotiff")
    #     band_num = layers.index(LayersEnum.CONTRIBUTOR.name.lower()) + 1
    #     band = cog_ds.GetRasterBand(band_num)
    #
    #     # Change the overviews on the contributor banc to use "nearest"
    #     # bug in gdal <3.3 doesn't compute overviews consistently, work around by doing it twice and copying ourselves
    #     major, minor = list(map(int, gdal.__version__.split(".")))[:2]
    #     if major > 3 or major == 3 and minor > 2:
    #         # in gdal 3.3+ just tell it to regenerate on the contributor band
    #         gdal.RegenerateOverviews(band, [band.GetOverview(n) for n in range(band.GetOverviewCount())], 'NEAREST')
    #     else:
    #         # in gdal 3.2- make a second tif that uses "nearest"
    #         # then copy those overviews into the original 'cubic' tif replacing the contributor overviews
    #         nearest_filename = str(export.cog_filename)+".nearest.tif"
    #         cog_ds_near = cogdriver.CreateCopy(nearest_filename, generalized_ds, 0,
    #                                            options=['OVERVIEW_RESAMPLING=NEAREST', 'OVERVIEWS=IGNORE_EXISTING', 'COMPRESS=LZW', "BIGTIFF=YES"])
    #         for i in range(cog_ds_near.GetRasterBand(band_num).GetOverviewCount()):
    #             band.GetOverview(i).WriteArray(cog_ds_near.GetRasterBand(band_num).GetOverview(i).ReadAsArray())
    #         del cog_ds_near
    #         remove_file(nearest_filename, allow_permission_fail=True)
    # band3 = cog_ds.GetRasterBand(3)  # try doing statistics here, they didn't seem to work up above
    # band3.ComputeStatistics(0)  # This didn't seem better -- maybe a gdal version or qgis issue?
    # del band3
    # del band
    del generalized_ds
    LOGGER.debug("Make Raster Attribute Table")
    # Note that we are renumbering the contributors above so this is no longer the nbs_id but just revised indices stored in the contributor layer
    make_raster_attr_table(str(export.cog_filename), all_simple_records)  # make a raster attribute table for the generalized dataset
    # FIXME - remove when Caris is fixed
    #  -- change raster attributes for Caris which is failing on 'metre'
    rat_text = open(export.rat_filename, 'rb').read()
    new_rat_text = rat_text.replace(b"<UnitType>metre</UnitType>", b"<UnitType>m</UnitType>")
    if b"<UnitType>m</UnitType>" not in new_rat_text:
        new_rat_text = rat_text.replace(b"</Description>", b"</Description><UnitType>m</UnitType>")
    open(export.rat_filename, 'wb').write(new_rat_text)
    # create_RAT(dataset)  # make a raster attribute table for the raw dataset

    # remove the score and extracted files
    export.cleanup_tempfiles(allow_permission_fail=True)


def complete_export_sequential(export, all_simple_records, closing_dist, epsg, decimals=None, block_size=DEFAULT_BLOCK_SIZE):
    """ for the export object, take the extracted filename and perform generalization, cloud overviews and consolidate the raster attribute table
    all_simple_records is a list of the postgres metadata records for operating on the raster attributes.
    closing_dist is used in generalization.
    epsg is the target epsg used in generalization.
    """

    # FIXME - remove this change of nodata value once NBS is fixed to accept NaN in addition to 1000000
    #   also this give a problem in the HACK of storing large int32 in float32 tiffs
    #   the 1000000 becomes 1232348160 when translated back, so a contributor of 1232348160 would get stepped on by NaN
    #   numpy.frombuffer(numpy.array(1000000).astype(numpy.float32).tobytes(), numpy.int32)
    LOGGER.debug("change nodata value (Caris doesn't like NAN)")
    shutil.copyfile(export.extracted_filename, export.generalized_filename)
    elevation_band_index = raster_band_name_index(str(export.generalized_filename), ELEVATION_BAND_NAME)
    uncertainty_band_index = raster_band_name_index(str(export.generalized_filename), UNCERTAINTY_BAND_NAME)
    contributor_band_index = raster_band_name_index(str(export.generalized_filename), CONTRIBUTOR_BAND_NAME)
    ds = gdal.Open(str(export.generalized_filename), gdal.GA_Update)
    new_nodata = 1000000
    # read about 1GB at a time
    has_data = False
    bands = (elevation_band_index, uncertainty_band_index, contributor_band_index)
    for band_num in tqdm(bands, "change nodata values"):
        for ic, ir, nodata, data in iterate_gdal_image(ds, [band_num], block_size, block_size, leave_progress_bar=False):
            # limit the depth and uncertainty to two decimals
            subarray = data[0]
            # change the nodata value and write back to the file
            subarray[numpy.isnan(subarray)] = new_nodata
            ds.GetRasterBand(band_num).WriteArray(subarray, ic, ir)
            if band_num == elevation_band_index:
                has_data = has_data or where_not_nodata(subarray, new_nodata).any()
        data, subarray = None, None  # delete the array data to possibly release the memory (based on garbage collection timing)
    for b in bands:
        band = ds.GetRasterBand(b)
        band.SetNoDataValue(new_nodata)
    del band, ds  # close the dataset so generalize can work on it.


    # if interactive_debug:
    #     shutil.copyfile(export_generalized_filename, export_generalized_filename.with_suffix(".convert_nodata.tif"))
    LOGGER.debug("start generalize")
    if has_data:
        # FIXME - HACK  we should send a float version of the contributor number since we have a hack
        #   of storing int32 inside the float32 tiff, but zero is the same in both so we can get away with it for now
        # numpy.array(0, dtype=numpy.float32).tobytes()
        # Out[7]: b'\x00\x00\x00\x00'
        # numpy.array(0, dtype=numpy.int32).tobytes()
        # Out[8]: b'\x00\x00\x00\x00'

        # call the generalize function which used to be process_csar script
        generalize(str(export.generalized_filename), closing_dist, output_crs=epsg)

    LOGGER.debug("Copy to 'contribs' filename to start modifying the contributor from float to int")
    shutil.copyfile(export.generalized_filename, export.contributors_filename)
    generalized_ds = gdal.Open(str(export.contributors_filename), gdal.GA_Update)
    # limit values to X decimals (for compression) generalization will return full precision which we may not want for compression reasons
    if decimals is not None:
        LOGGER.debug("Limit decimal precision")
        decimal_multiplier = 10**decimals
        bands = (elevation_band_index, uncertainty_band_index)
        for band_num in bands:
            for ic, ir, nodata, data in iterate_gdal_image(generalized_ds, [band_num], block_size, block_size):
                # limit the depth and uncertainty to two decimals
                subarray = data[0]
                subarray = numpy.ceil(subarray * decimal_multiplier) / decimal_multiplier
                generalized_ds.GetRasterBand(band_num).WriteArray(subarray, ic, ir)
    generalized_ds.GetRasterBand(elevation_band_index).SetDescription(ELEVATION_BAND_NAME.capitalize())
    generalized_ds.GetRasterBand(uncertainty_band_index).SetDescription(UNCERTAINTY_BAND_NAME.capitalize())
    generalized_ds.GetRasterBand(contributor_band_index).SetDescription(CONTRIBUTOR_BAND_NAME.capitalize())
    LOGGER.debug("Consolidate contributor IDs")
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


    # # @TODO - couldn't this convert all the floats to ints and then recover the nan or 1000000
    # #    by looking at the original float data?
    # #    Two loops would be faster than looping the data N times
    # #    However, since we are now also consolidating the values as 0,1,2,3,4... it needs a loop.

    # make a copy of the dictionary so we don't modify the caller's data, or while converting the contributor numbers make a new dict
    orig_simple_records = all_simple_records
    all_simple_records = {}
    old_to_new_mapping = {0: 0.0}  # add the generalized contributor since we know the conversion and there is no matching orig_simple_record
    generalized_contrib_band = generalized_ds.GetRasterBand(contributor_band_index)
    inew = 1
    for ic, ir, nodata, data in iterate_gdal_image(generalized_ds, [contributor_band_index], block_size, block_size):
        float_contrib = data[0]
        icontrib = float_contrib.copy()  # make a copy of the data in case we ever had a value in the float version equaled something mapped to an int
        contribs = numpy.unique(float_contrib)
        int_contribs = numpy.sort(contributor_float_to_int(contribs)).tolist()
        # leave the 1000000 value (Nan replacement) in the contributor space
        try:
            int_contribs.remove(contributor_float_to_int(1000000))
        except ValueError:
            pass
        # if not int_contribs:  # there is no data in this tile - make a fake contributor list so it continues on.
        #     int_contribs = [0]
        # if int_contribs[0] != 0:  # make sure there is a zero entry for the generalization so enumerate below doesn't move a real contributor into the zero spot
        #     int_contribs = [0] + int_contribs
        # @TODO - couldn't this convert all the floats to ints and then recover the nan or 1000000
        #    by looking at the original float data?
        #    Two loops would be faster than looping the data N times
        #    However, since we are now also consolidating the values as 0,1,2,3,4... it needs a loop.
        for iold in int_contribs:
            iold_as_float = contributor_int_to_float(iold)
            if iold not in old_to_new_mapping:
                old_to_new_mapping[iold] = inew
                if inew != 0:  # the generalization has no change from int to float and would cause a keyerror in all_simple_records
                    all_simple_records[inew] = orig_simple_records[iold]
                inew += 1
            icontrib[float_contrib == iold_as_float] = old_to_new_mapping[iold]

        generalized_contrib_band.WriteArray(icontrib, ic, ir)
        # regenerate the stats since we replaced the floats with integers
    # Set the statistics in the exported data
    try:
        # generalized_contrib_band.SetStatistics(float(0), float(inew), float(inew/2), float(numpy.std(range(inew))))
        generalized_contrib_band.ComputeStatistics(0)
        generalized_ds.GetRasterBand(elevation_band_index).ComputeStatistics(0)
        generalized_ds.GetRasterBand(uncertainty_band_index).ComputeStatistics(0)
        generalized_ds.FlushCache()
        # band = ds.GetRasterBand(2)
        # m = band.GetMetadata()
        # m['STATISTICS_VALID_PERCENT'] = '50'
        # band.SetMetadata(m)
    except ValueError:
        LOGGER.warning("Could not compute statistics, empty data?")
    del generalized_contrib_band



    LOGGER.debug("Make initial Cloud Optimized Geotiff")
    cogdriver = gdal.GetDriverByName("COG")
    # from xipe_dev.xipe.deliverables import CLOUD_OPTIMIZED_GEOTIFF_CREATION_OPTIONS
    #     'PREDICTOR': 3,  # floating point predictor, change this to 2 for horizontal if using an integer tiff
    # use NEAREST to eliminate some artifacts from cubic overview at nodata gaps and
    # also reduces file size by keeping the 2 decimal values from nearest rather than averages full res data
    cog_ds = cogdriver.CreateCopy(str(export.cog_filename), generalized_ds, 0, options=['TILED=YES', 'PREDICTOR=3', 'RESAMPLING=NEAREST', 'OVERVIEW_RESAMPLING=NEAREST', 'COMPRESS=LZW', "BIGTIFF=YES", 'OVERVIEWS=IGNORE_EXISTING'])
    # Add the Bruty metadata version here.  Adding after the COG overviews are added genereated a warning by GDAL but CreateCopy didn't retain the metadata.
    current_ver = BrutyExportVersion()
    current_ver.to_dataset(cog_ds)

    # # the overview of the contributor needs 'nearest' (otherwise it interpolates to non-contributor numbers like 15.65)
    # # which is what paletted data uses, while elevation and uncertainty needs "cubic" to display best (see fliers)
    # layers = [cog_ds.GetRasterBand(b + 1).GetDescription().lower() for b in range(cog_ds.RasterCount)]
    # if LayersEnum.CONTRIBUTOR.name.lower() in layers:
    #     LOGGER.debug("Replace contributor overview in Cloud Optimized Geotiff")
    #     band_num = layers.index(LayersEnum.CONTRIBUTOR.name.lower()) + 1
    #     band = cog_ds.GetRasterBand(band_num)
    #
    #     # Change the overviews on the contributor banc to use "nearest"
    #     # bug in gdal <3.3 doesn't compute overviews consistently, work around by doing it twice and copying ourselves
    #     major, minor = list(map(int, gdal.__version__.split(".")))[:2]
    #     if major > 3 or major == 3 and minor > 2:
    #         # in gdal 3.3+ just tell it to regenerate on the contributor band
    #         gdal.RegenerateOverviews(band, [band.GetOverview(n) for n in range(band.GetOverviewCount())], 'NEAREST')
    #     else:
    #         # in gdal 3.2- make a second tif that uses "nearest"
    #         # then copy those overviews into the original 'cubic' tif replacing the contributor overviews
    #         nearest_filename = str(export.cog_filename)+".nearest.tif"
    #         cog_ds_near = cogdriver.CreateCopy(nearest_filename, generalized_ds, 0,
    #                                            options=['OVERVIEW_RESAMPLING=NEAREST', 'OVERVIEWS=IGNORE_EXISTING', 'COMPRESS=LZW', "BIGTIFF=YES"])
    #         for i in range(cog_ds_near.GetRasterBand(band_num).GetOverviewCount()):
    #             band.GetOverview(i).WriteArray(cog_ds_near.GetRasterBand(band_num).GetOverview(i).ReadAsArray())
    #         del cog_ds_near
    #         remove_file(nearest_filename, allow_permission_fail=True)
    # band3 = cog_ds.GetRasterBand(3)  # try doing statistics here, they didn't seem to work up above
    # band3.ComputeStatistics(0)  # This didn't seem better -- maybe a gdal version or qgis issue?
    # del band3
    # del band
    del cog_ds
    del generalized_ds

    LOGGER.debug("Make Raster Attribute Table")
    # Note that we are renumbering the contributors above so this is no longer the nbs_id but just revised indices stored in the contributor layer
    make_raster_attr_table(str(export.cog_filename), all_simple_records)  # make a raster attribute table for the generalized dataset
    # FIXME - remove when Caris is fixed
    #  -- change raster attributes for Caris which is failing on 'metre'
    rat_text = open(export.rat_filename, 'rb').read()
    new_rat_text = rat_text.replace(b"<UnitType>metre</UnitType>", b"<UnitType>m</UnitType>")
    if b"<UnitType>m</UnitType>" not in new_rat_text:
        new_rat_text = rat_text.replace(b"</Description>", b"</Description><UnitType>m</UnitType>")
    open(export.rat_filename, 'wb').write(new_rat_text)
    # create_RAT(dataset)  # make a raster attribute table for the raw dataset

    # remove the score and extracted files
    export.cleanup_tempfiles(allow_permission_fail=True)

def get_tile_records(conn_info: ConnectionInfo, tile_info, select="id", **kwargs):
    """ Return a list of the record values for the given tile_info and kwargs.  Returns a list for each tablename in conn_info.tablenames.

    Parameters
    ----------
    conn_info
        user, password, database, port for the postgres connection and tablenames to query from.
    tile_info
        Uses the tile, utm, hemi, datum, pb (product_branch), locality, res for the query.
    select
        String of columns to retrieve.  Ex: "id" or "id, data_location"
    kwargs
        keys and values to use in the sql statement.  All use "equals" for the comparison.
    Returns
    -------
    list of lists
    """
    connection, cursor = connection_with_retries(conn_info)
    rids = []
    for tablename in conn_info.tablenames:
        record = [tile_info.tile, tile_info.utm, tile_info.hemi.upper(), tile_info.datum, tile_info.pb, tile_info.locality, tile_info.resolution, tile_info.closing_dist]
        query = f"""select {select} FROM {tablename} WHERE tile=%s AND utm=%s AND hemisphere=%s AND datum=%s AND production_branch=%s AND locality=%s AND resolution=%s AND closing_dist=%s"""
        for key, val in kwargs.items():
            query += f" AND {key}=%s"
            record.append(val)
        cursor.execute(query, record)
        try:
            rid = cursor.fetchall()
        except (TypeError, IndexError):
            rid = None
        rids.append(rid)
    return rids

def fill_xbox_closing_dist(conn_info: ConnectionInfo):
    """ Fill any empth closing_dist values with the value parsed from the filename.
    This was helpful since the closing dist was not originally in the table but only encoded in the filename

    Parameters
    ----------
    conn_info
        ConnectionInfo object with the database connection information and tablenames to query from.

    Returns
    -------
    None

    """
    connection, cursor = connection_with_retries(conn_info)
    for tablename in conn_info.tablenames:
        query = f"""select id,data_location FROM {tablename} WHERE closing_dist IS NULL"""
        cursor.execute(query)
        rids = cursor.fetchall()
        for rid, pth in rids:
            v = re.search(r"\dm_(Navigation|Internal|Public)_\d{8}_\d{6}_(?P<close>\d+)m", pth)
            if v:
                cursor.execute(f"UPDATE {tablename} SET closing_dist=%s WHERE id=%s", (int(v.group("close")), rid))
            else:
                print('failed', pth)
    connection.commit()
    connection.close()

def update_export_record(conn_info: ConnectionInfo, row_id: int, **to_update):
    if conn_info.database is not None:
        connection, cursor = connection_with_retries(conn_info)
        # @TODO this would insert a dictionary but the UPDATE command doesn't work the same.
        # columns = to_update.keys()
        # values = [to_update[column] for column in columns]
        # insert_statement = 'INSERT INTO xbox (%s) values (%s) where id = %s '
        # cursor.execute(insert_statement, (AsIs(','.join(columns)), tuple(values), row_id))
        for key, val in to_update.items():
            for tablename in conn_info.tablenames:
                cursor.execute(f"UPDATE {tablename} SET {key}=%s WHERE id=%s", (val, row_id))
        connection.commit()
        connection.close()


def write_export_record(conn_info: ConnectionInfo, export: RasterExport, internal: bool = False, navigation: bool = False, public: bool = False, notes=""):
    spec_table = "spec_tiles"
    id_of_new_row = None
    if conn_info.database is not None:
        connection, cursor = connection_with_retries(conn_info)
        ti = export.tile_info
        # @TODO this should be a dictionary but there is no clear syntax for writing dicts directly.
        #   make or find a wrapper which is like: con.execute(insert into table (dict.keys()) values (dict.values()))
        record = [ti.tile, ti.utm, ti.hemi.upper(), ti.datum, ti.pb, ti.locality, ti.resolution, ti.closing_dist,
                  str(export.cog_filename), str(export.rat_filename), export.data_time, export.export_time,
                  ti.raw_geometry, internal, navigation, public, notes]
        q_str = ", ".join(["%s"] * len(record))
        cursor.execute(f"""INSERT INTO xbox (tile, utm, hemisphere, datum, production_branch, locality, resolution, closing_dist, 
            data_location, data_aux_location, combine_time, export_time, 
            geometry, internal, navigation, public, notes) VALUES ({q_str}) RETURNING id""", record)

        id_of_new_row = cursor.fetchone()[0]
        connection.commit()
        connection.close()
    return id_of_new_row


def make_parser():
    parser = argparse.ArgumentParser(description='Combine a NBS postgres table(s) into a Bruty dataset')
    parser.add_argument("-?", "--show_help", action="store_true",
                        help="show this help message and exit")

    parser.add_argument("-c", "--config", type=str, metavar='bruty_dir', default="",
                        help="path to root folder of bruty data")
    parser.add_argument("-u", "--use_caches", action='store_true', dest='use_caches', default=False,
                        help="Used cached metadata stored in the bruty database directories")
    parser.add_argument("-k", "--res_tile_pk_id", type=int, metavar='res_tile_pk_id',
                        help=f"primary key of the tile to export from the {ResolutionTileInfo.SOURCE_TABLE} table")
    parser.add_argument("-d", "--decimals", type=int, metavar='decimals', default=None,  # nargs="+"
                        help="number of decimals to keep in elevation and uncertainty bands")
    parser.add_argument("-f", "--fingerprint", type=str, metavar='fingerprint', default="",
                        help="fingerprint to store success/fail code with in sqlite db within the REVIEWED (qualified), for_navigation database")
    return parser



if __name__ == "__main__":
    """ This is not really a command line app.
    This is just a way to run the export in a new console so uses a pickle file to pass cached data.
    Similar to running using multiprocessing.Process
    """
    parser = make_parser()
    args = parser.parse_args()
    proc_start = time.time()
    if args.show_help or not args.config or not args.res_tile_pk_id:
        parser.print_help()
        ret = 1

    if args.config and args.res_tile_pk_id:
        config_file = read_config(args.config)
        config = config_file['EXPORT']
        # use_locks(args.lock_server)
        conn_info = connect_params_from_config(config)

        tile_info = ResolutionTileInfo.from_table(conn_info, args.res_tile_pk_id)
        try:
            LOGGER.debug(f"Processing {tile_info.full_name}")
            ret = combine_and_export(config, tile_info, args.decimals, args.use_caches)
        except Exception as e:
            traceback.print_exc()
            msg = f"{tile_info.full_name} had an unhandled exception - see message above"
            print(msg)
            LOGGER.error(traceback.format_exc())
            LOGGER.error(msg)
            ret = UNHANDLED_EXCEPTION
        # @TODO We don't need this since we are tracking in the postgres tables?
        # if args.fingerprint:
        #     try:
        #         db = create_world_db(config['data_dir'], tile_info, REVIEWED, True)
        #         d = db.completion_codes.data_class()
        #         d.ttime = datetime.datetime.now()
        #         d.ttype = "EXPORT"
        #         d.code = ret
        #         d.fingerprint = args.fingerprint
        #         db.completion_codes[args.fingerprint] = d
        #     except:
        #         traceback.print_exc()
    LOGGER.debug(f"Exiting with code {ret} after {int(time.time()-proc_start)} seconds")
    sys.exit(ret)


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

