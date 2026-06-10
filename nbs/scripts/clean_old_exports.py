import os
import argparse
import pathlib
import shutil

from nbs.bruty.nbs_postgres import id_to_scoring, get_nbs_records, nbs_survey_sort, ConnectionInfo, connect_params_from_config, make_contributor_csv
from nbs.bruty.nbs_postgres import REVIEWED, PREREVIEW, SENSITIVE, ENC, GMRT, NOT_NAV, INTERNAL, NAVIGATION, PUBLIC, connect_params_from_config, connection_with_retries
from nbs.configs import iter_configs, read_config
from nbs.bruty.utils import remove_file
import nbs.scripts

def cmppath(pth):
    pathobj = pathlib.Path(pth)
    return os.path.join(pathobj.parent.parent.name, pathobj.parent.name, pathobj.name).lower()

def move_orphans(conn_info, in_dir, out_dir, remove=False, dry_run=True):
    # FIXME - This will move the hardlinks that are now being created by the export script
    """ FIXME - This will move the hardlinks that are now being created by the export script

    Find files that are not referenced in the xbox table and move them to a new location.
    The out_dir will replace the in_dir so that the rest of the directory structure is preserved.
    All directories will be created as needed.

    Ex: move_orphans(r'W:\bruty_tile_exports', r'W:\bruty_tile_exports_orphans', dry_run=True)

    Parameters
    ----------
    in_dir
        Directory to search for orphan files
    out_dir
        Directory to move orphan files to
    dry_run
        If True, only print the files that would be moved

    Returns
    -------
    None
    """
    print("retrieving xbox records")
    fields, recs = get_nbs_records(conn_info.table, conn_info, exclude_fields=['geometry', 'geometry_modified'])
    print("finding files")
    disk_files = [str(pth) for pth in pathlib.Path(in_dir).rglob('*') if pth.is_file()]

    db_paths = set()
    for rec in recs:
        for pth in (rec['data_location'], rec['data_aux_location']):
            recdir = pathlib.Path(pth)
            rel_path = cmppath(recdir)
            db_paths.add(rel_path)
    action = "move to" if not remove else "delete"
    for disk_file in disk_files:
        rel_path = cmppath(disk_file)
        if rel_path not in db_paths:

            new_path = disk_file.replace(in_dir, out_dir)
            if dry_run:
                print(f"    Would {action} {new_path}")
            else:

                if not remove:
                    print(f"    Moving to {new_path}")
                    os.makedirs(pathlib.Path(new_path).parent, exist_ok=True)
                    shutil.move(disk_file, new_path)
                else:
                    print(f"    Deleting {disk_file}")
                    remove_file(disk_file, allow_permission_fail=True)
        else:
            print(f"File in database: {disk_file}")


def make_key(r):
    return tuple(r[k] for k in ('production_branch', 'utm', 'hemisphere', 'tile', 'datum', 'resolution'))

# TODO - The export was making files that were not captured in the xbox table.
#  We should look at the ouptut directories to remove files not referenced in the xbox table
def main(conn_info, retain=2, dryrun=True):
    # We'll keep all the row records in groups.  Each distinct export category should keep its own files on disk
    # First we'll gather all exported files into dictionaries of lists that specify all the exports that exist for a utm zone/product branch/datum/res/tile
    fields, recs = get_nbs_records(conn_info.table, conn_info, exclude_fields=['geometry', 'geometry_modified'])
    connection, cursor = connection_with_retries(conn_info)
    export_types = ['internal', 'navigation', 'public']
    keeps = {}
    deletes = {}
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
                # if the number of accepts is greater than the user selected number of files to keep, then add to the set of possible deletes
                # This will include any non-accepted files that are older than the last accepted file
                if num_accepted >= retain:
                    place_filename_in = deletes
                # count the number of approved exports so that we retain the most recent two and any non-accepted files that are newer than the last accepted file
                # @NOTE Do this after the count check above so that if the retain number was reached that either approved or non-approved get removed
                if rec['approved']:
                    num_accepted += 1
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
    parser = argparse.ArgumentParser(description='This script will clean up old exports.  If orphan_source is specified then a directory will be cleaned of all files not in the export table.  Otherwise, the xbox table and the file system will be searched.\nThe most recent two APPROVED exports, and any not yet approved that are more recent than the two approved, will be kept and the older ones will be deleted.')
    parser.add_argument("-?", "--show_help", action="store_true",
                        help="show this help message and exit")

    # parser.add_argument("-t", "--export_time", type=str, metavar='export_time', default='',  # nargs="+"
    #                     help="export time to append to filenames")
    # parser.add_argument("-d", "--decimals", type=int, metavar='decimals', default=None,  # nargs="+"
    #                     help="number of decimals to keep in elevation and uncertainty bands")
    parser.add_argument("-d", "--dryrun", action="store_true", help="remove the records cache file after reading it")
    parser.add_argument("-o", "--orphan_src", type=str, default="", help="move or delete orphan files, requires -r or -m flag as well")
    parser.add_argument("-r", "--remove_orphans", action="store_true", help="remove orphan files")
    parser.add_argument("-m", "--orphan_dest", type=str, default="", help="move orphan files")
    parser.add_argument("-c", "--config_path", type=str, metavar='config_path', default="",
                        help="location to config file for connection info")
    return parser



if __name__ == "__main__":
    """ This script will clean up old exports from the xbox table and the file system
    """
    parser = make_parser()
    args = parser.parse_args()
    if args.show_help:
        parser.print_help()
    else:
        config_path = args.config_path
        if not config_path:
            script_dir = pathlib.Path(nbs.scripts.__path__[0])
            config_path = script_dir.joinpath(r'base_configs\nbs_postgres.config')
        config_file = read_config(config_path)
        connect_info = connect_params_from_config(config_file['DEFAULT'])
        connect_info.database = connect_info.export_database
        retain_cnt = config_file['DEFAULT'].getint('retain', 2)
        connect_info.table = config_file['DEFAULT'].get("export_table", "xbox")
        if args.orphan_src:
            move_orphans(connect_info, args.orphan_src, args.orphan_dest, args.remove_orphans, args.dryrun)
        else:
            main(connect_info, retain_cnt, dryrun=args.dryrun)