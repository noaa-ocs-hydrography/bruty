# From xipe_dev originally
import os
from datetime import datetime
from typing import Any

import numpy

from fuse_dev.fuse.meta_review.meta_review import MetadataDatabase
from nbs.configs import get_logger
from nbs.bruty.constants import CONTRIBUTOR_BAND_NAME

LOGGER = get_logger('xipe.csar.meta')

# indices of fields in the CARIS Contributor string
DEFAULT_INTERPOLATED_COLUMN = 19
DEFAULT_TO_FILENAME_COLUMN = 3
DEFAULT_TABLENAME_COLUMN = 7
DATABASE_NAME = "metadata"

# build os-independent default location
DEFAULT_DRIVE = os.path.join('//', 'OCS-S-HyperV17', 'NBS_Store1')
URL_FILENAME = os.path.join(DEFAULT_DRIVE, 'credentials', 'postgres_hostname.txt')
CREDENTIALS_FILENAME = os.path.join(DEFAULT_DRIVE, 'credentials', 'postgres_scripting.txt')

# override with user-specific settings if they exist
USER_HOME = os.path.expanduser('~')
if os.path.exists(os.path.join(USER_HOME, 'credentials', 'postgres_hostname.txt')):
    URL_FILENAME = os.path.join(USER_HOME, 'credentials', 'postgres_hostname.txt')
if os.path.exists(os.path.join(USER_HOME, 'credentials', 'postgres_scripting.txt')):
    CREDENTIALS_FILENAME = os.path.join(USER_HOME, 'credentials', 'postgres_scripting.txt')


def contributor_metadata_from_file(filename: str, tables: str, to_tablename_column: int = None, to_filename_column: int = None, url: str = None, username: str = None, password: str = None,
                                   retain_missing_records: bool = False) -> {str: Any}:
    """
    Get metadata from the given contributor table.

    Parameters
    ----------
    filename
        file path of contributor table
    to_filename_column
        column number of `to_filename` field in contributor string
    tables
        list of metadata table names
    url
        hostname of metadata database
    username
        username to connect to metadata database
    password
        password to connect to metadata database

    Returns
    -------
    {str: Any}
        contributor filenames with respective raster values and metadata
    """

    if os.path.splitext(filename)[1] != '.csv':
        filename = f'{os.path.splitext(filename)[0]}_{CONTRIBUTOR_BAND_NAME}.csv'

    if to_filename_column is None:
        to_filename_column = DEFAULT_TO_FILENAME_COLUMN

    if to_tablename_column is None:
        to_tablename_column = DEFAULT_TABLENAME_COLUMN

    contributor_table = numpy.loadtxt(filename, delimiter=',', dtype=str)
    LOGGER.info(f'found {len(contributor_table)} contributors in "{filename}"')

    contributors = contributor_table[:, to_filename_column - 1]
    contributor_tables = contributor_table[:, to_tablename_column - 1]
    contributor_indices = dict(zip(contributors, contributor_table[:, 0]))

    contributors = metadata_by_contributor(contributors.tolist(), contributor_tables, url, username, password)

    contributors_with_records = {filename: contributor for filename, contributor in contributors.items() if contributor is not None and contributor != "Missing"}

    if len(contributors) > len(contributors_with_records):
        contributors_without_records = {filename: contributor_indices[filename] for filename in contributors if filename not in contributors_with_records}
        LOGGER.warning(f'could not find metadata records for {len(contributors_without_records)} contributors: {contributors_without_records}')
        if not retain_missing_records:
            contributors = contributors_with_records
    del contributors_with_records

    return {contributor: {'index': int(contributor_indices[contributor]), 'record': record} for contributor, record in contributors.items()}


def metadata_by_contributor(contributors: [str], table_names: [str], url: str, username: str = None, password: str = None) -> {str: {str: Any}}:
    """
    Get metadata for the given contributors.

    Parameters
    ----------
    contributors
        list of contributor filenames, aligning with the `to_filename` field in the provided metadata tables
    tables
        list of metadata table names
    url
        hostname of metadata database
    username
        username to connect to metadata database
    password
        password to connect to metadata database

    Returns
    -------
    {str: {str: Any}}
        contributor filenames with respective metadata
    """

    if url is None:
        with open(URL_FILENAME) as hostname_file:
            url = hostname_file.readline()

    if username is None or password is None:
        with open(CREDENTIALS_FILENAME) as database_credentials_file:
            username, password = [line.strip() for line in database_credentials_file][:2]

    contributor_metadata = {}
    for n, contributor_filename in enumerate(contributors):
        record = None
        if contributor_filename == "NBS Generalization":
            contributor_metadata[contributor_filename] = None
            continue
        table_name = table_names[n]
        metadata_table = MetadataDatabase(url, DATABASE_NAME, username, password, table_name)
        records = metadata_table.records_where({'to_filename': f'%{contributor_filename}%'})
        if len(records) > 1:
            LOGGER.error(f"{contributor_filename} returned more than one result {records}")
        if len(records) > 0:
            if record is not None:
                LOGGER.error(f"{contributor_filename} was found in more than one table:\n{record['found_in_table']}\nand in {[metadata_table.hostname, metadata_table.database_name, metadata_table.table_name]}")
            lengths = {len(os.path.basename(record['to_filename'])): index for index, record in enumerate(records)}
            record = records[lengths[min(lengths)]]
            record['found_in_table'] = [metadata_table.hostname, metadata_table.database_name, metadata_table.table_name]
            # break
        if record == None:
            record = "Missing"
        contributor_metadata[contributor_filename] = record

    return contributor_metadata


def raster_decay_date(filename: str, to_filename_column: int = None) -> datetime:
    contributor_metadata = contributor_metadata_from_file(filename, to_filename_column)

    decay_dates = []
    for contributor_filename, record in contributor_metadata.items():
        if record['record'] is not None:
            if 'decay_date' in record['record']:
                decay_dates.append(record['record']['decay_date'])
            else:
                LOGGER.warning(f'record does not have a decay date: "{contributor_filename}"')
        else:
            LOGGER.warning(f'record does not exist in metadata: "{contributor_filename}"')
    decay_dates = numpy.unique(decay_dates)

    decay_date = decay_dates[0]
    if len(decay_dates) > 1:
        raise ValueError(f'found multiple decay dates ({decay_dates}) in contributor metadata of "{filename}"')

    return decay_date


def raster_contains_sensitive_data(filename: str) -> bool:
    return any(record['sensitive'] for record in contributor_metadata_from_file(filename))
