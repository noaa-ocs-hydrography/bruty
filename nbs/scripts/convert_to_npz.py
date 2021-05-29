import pathlib

from psycopg2 import ProgrammingError
from tqdm import tqdm
import numpy
import caris.coverage as cc

from data_management.db_connection import connect_with_retries
from nbs_utils.points_utils import to_npz, from_npz, npy_to_npz


def convert_bruty_npy(csar_path, npy_path, npz_path):
    data_type = cc.identify(str(csar_path))
    if data_type == cc.DatasetType.CLOUD:
        cloud = cc.Cloud(str(csar_path))
        npy_to_npz(npz_path, cloud.wkt_cosys, npy_path)
        # print("Made point cloud npz")
    else:
        print(csar_path, "wasn't a point cloud")


def existing_npy_name(path):
    """For the given path, see if there is an existing npy file without a npz file.
    If so, read the csar and make a npz file with the WKT coordinate system included.
    Non-CSAR filenames will be returned unchanged.
    """
    ret = ""
    if str(path).lower().endswith(".csar"):
        csar_path = pathlib.Path(path)
        npz_path = csar_path.with_suffix(".bruty.npz")
        npy_path = csar_path.with_suffix(".bruty.npy")
        if npz_path.exists():
            # ret = str(npz_path)
            pass
        elif npy_path.exists():
            # if the npy file is newer than the geopackage or within 10 minutes of the geopackage then use the numpy file
            # FIXME - how much tolerance do we give the npy file and do we check modified time or creation?
            if (npy_path.stat().st_mtime - csar_path.stat().st_mtime) > -600:
                ret = str(npy_path)
            else:
                print("time mismatch", path)

    return ret


def convert_npys_for_table(tablename):
    """For the given table loop all the filenames that would be active (manual if it exists, script otherwise).
    For each file, create an npz if there is only a npy in place.
    """
    user = ""
    password = ""
    host = ''
    port = ''
    connection = connect_with_retries(database="metadata", user=user, password=password, host=host, port=port)
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT * FROM ' + tablename)
    except ProgrammingError:
        print("table", tablename, "not found")
        return

    records = cursor.fetchall()
    fields = [desc[0] for desc in cursor.description]

    filename_col = fields.index('from_filename')
    path_col = fields.index('script_to_filename')
    manual_path_col = fields.index('manual_to_filename')

    for rec in tqdm(records, desc=tablename, mininterval=.7):
        path = rec[manual_path_col]
        # A manual string can be an empty string (not Null) and also protect against it looking empty (just a space " ")
        if path is None or not path.strip():
            path = rec[path_col]
            if path is None or not path.strip():
                continue  # no valid filename found, skip
        path = pathlib.Path(path)
        export_path = existing_npy_name(path)  # see if there is a npy file
        if export_path:
            export_path = pathlib.Path(export_path)
            convert_bruty_npy(path, export_path, export_path.with_suffix(".npz"))


if __name__ == '__main__':
    table_families = ['pbg_puertorico_utm20n_mllw',
                      'pbg_gulf_utm14n_mllw',
                      'pbg_gulf_utm15n_mllw',
                      'pbg_gulf_utm16n_mllw',
                      'pbc_utm18n_mllw',
                      'pbc_utm19n_mllw',
                      'pbd_utm11n_mllw_lalb',
                      ]
    for table_family in table_families:
        for subtype in ["", "_sensitive", "_prereview"]:
            convert_npys_for_table(table_family+subtype)

