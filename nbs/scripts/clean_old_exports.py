import os
import argparse
import pathlib

from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, ConnectionInfo, connect_params_from_config, make_contributor_csv
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, NOT_NAV, INTERNAL, NAVIGATION, PUBLIC, connect_params_from_config, connection_with_retries
from nbs.configs import iter_configs
from nbs.bruty.utils import remove_file
import nbs.scripts

script_dir = pathlib.Path(nbs.scripts.__path__[0])
config_filename, config_file = [x for x in iter_configs([script_dir.joinpath(r'base_configs\temp_xbox.config')])][0]
conn_info = connect_params_from_config(config_file['DEFAULT'])
conn_info.database = "tile_specifications"
fields, recs = get_nbs_records("xbox", conn_info, exclude_fields=['geometry', 'geometry_modified'])
connection, cursor = connection_with_retries(conn_info)
export_types = ['internal', 'navigation', 'public']
keeps = {}
deletes = {}
retain = config_file['DEFAULT'].getint('retain', 2)

def make_key(r):
    return tuple(r[k] for k in ('production_branch', 'utm', 'hemisphere', 'tile', 'datum', 'resolution'))


def main(dryrun=True):
    # We'll keep all the row records in groups.  Each distinct export category should keep it's own files on disk
    # First we'll gather all exported files into dictionaries of lists that specify all the exports that exist for a utm zone/product branch/datum/res/tile
    exports = {export_type: {} for export_type in export_types}
    for rec in recs:
        k = make_key(rec)
        for export_type in export_types:
            if rec[export_type]:
                exports[export_type].setdefault(k, []).append(rec)

    # Now we'll boil all the exports down to two options: keep or delete (if not in the keep list)
    # a file could be used by more than one record and a record might use the same file in more than one internal/public/navigation export
    # so after getting the two lists we have to see if the delete names are in the keep list
    # We are going to assume that we can compare using case insensitive (like windows) for if there is any overlap between the keep vs delete dicts
    # -- keep the lower case for comparison but the original for deletion (Linux is case sensitive but Windows is not)
    for export_type in export_types:  # Bluetope, Navigation, Public etc
        all_keys = list(exports[export_type].keys())
        all_keys.sort()
        for k in all_keys:
            vals = exports[export_type][k]
            place_filename_in = keeps
            num_accepted = 0
            # sort the exported files by descneing export time
            vals.sort(key=lambda r: r["export_time"], reverse=True)
            for rec in vals:
                # count the number of approved exports so that we retain the most recent two and any non-accepted files that are newer than the last accepted file
                if rec['approved']:
                    num_accepted += 1
                # if the number of accepts is greater than the user selected number of files to keep, then add to the set of possible deletes
                # This will include any non-accepted files that are older than the last accepted file
                if num_accepted > retain:
                    place_filename_in = deletes
                # keep the lower case for comparison but the original for deletion (Linux is case sensitive but Windows is not)
                place_filename_in[rec['data_location'].lower()] = rec
    would_keep_overlap = set(deletes).intersection(set(keeps))
    # If an export appears in both the keep and delete listings then keep those exports
    delete_keys = list(set(deletes).difference(set(keeps)))  # the set logic loses the ordering, so re-sort the data
    delete_keys.sort(key=lambda k: make_key(deletes[k])+(deletes[k]['export_time'],))
    print(f"Preparing to delete {len(delete_keys)} records and associated files")
    for filename in delete_keys:
        rec = deletes[filename]
        print("delete", make_key(rec), rec['export_time'], rec['id'], rec['data_location'])
        if not dryrun:
            del_tif = remove_file(rec['data_location'], allow_permission_fail=True)
            del_aux = remove_file(rec['data_aux_location'], allow_permission_fail=True)
            if del_tif and del_aux:  # remove row from the xbox table
                cursor.execute(f"""delete from xbox where id={rec['id']}""")

    if not dryrun:
        connection.commit()
    else:
        print('***********************************************************************************')
        print('********************************    Keeping   *************************************')
        print('***********************************************************************************')
        keys = list(set(keeps))
        keys.sort(key=lambda k: make_key(keeps[k])+(keeps[k]['export_time'],))
        for filename in keys:
            rec = keeps[filename]
            print("keeping", make_key(rec), rec['export_time'], rec['id'], rec['data_location'])


def make_parser():
    parser = argparse.ArgumentParser(description='This script will clean up old exports from the xbox table and the file system.\nThe most recent two APPROVED exports, and any not yet approved that are more recent than the two approved, will be kept and the older ones will be deleted.')
    parser.add_argument("-?", "--show_help", action="store_true",
                        help="show this help message and exit")

    # parser.add_argument("-t", "--export_time", type=str, metavar='export_time', default='',  # nargs="+"
    #                     help="export time to append to filenames")
    # parser.add_argument("-d", "--decimals", type=int, metavar='decimals', default=None,  # nargs="+"
    #                     help="number of decimals to keep in elevation and uncertainty bands")
    parser.add_argument("-d", "--dryrun", action="store_true", help="remove the records cache file after reading it")
    return parser



if __name__ == "__main__":
    """ This script will clean up old exports from the xbox table and the file system
    """
    parser = make_parser()
    args = parser.parse_args()
    if args.show_help:
        parser.print_help()
    else:
        main(dryrun=args.dryrun)