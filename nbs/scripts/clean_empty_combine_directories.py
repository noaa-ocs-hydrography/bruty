""" There are a lot of empty directories in the combine which make it slow to move/delete directories.
Not sure where they are coming from.  Doesn't seem to be combine or export.  Maybe the tile_status script?
In any event, this script will simply recurse the combine directories and if a tile directory is empty or only has the metadata file
  then delete the directory.  Would look like XXX/YYY/metadata.json
After checking all the sub directories then check the top levels for empties as well.

"""

import os
import re
import shutil
import pathlib

data_dir = pathlib.Path(r"X:\bruty_databases")
# data_dir = pathlib.Path(r"D:\debug\combines")
remove_empty_subtiles = False
remove_old_accum_directories = True

if remove_empty_subtiles:
    for parent, dirs, files in os.walk(data_dir, topdown=False):
        cur_dir = pathlib.Path(parent)
        # make sure we are three levels down (database/rowYYY/colXXX)
        if cur_dir.parent == data_dir:
            print(parent)
        if cur_dir.name.isdigit() and cur_dir.parent.parent == data_dir:
            print()  # newline per row
        if cur_dir.name.isdigit() and cur_dir.parent.name.isdigit() and cur_dir.parent.parent.parent == data_dir:
            if not files or files == ["metadata.json"]:
                # print(f"column dir {parent} only has metadata.json")
                print(".", end="")
                shutil.rmtree(parent, ignore_errors=True)
            else:
                pass  # print(f"{parent} has data")

    # iterate again to find rows that are now empty (os.walk caches the sub directories so can't do this in the same loop as above
    for parent, dirs, files in os.walk(data_dir, topdown=False):
        cur_dir = pathlib.Path(parent)
        if cur_dir.parent == data_dir:
            print(parent)
        if cur_dir.name.isdigit() and cur_dir.parent.parent == data_dir:
            if not files and not dirs:
                print(",", end="")
                # print(f"row dir {parent} is empty")
                shutil.rmtree(parent, ignore_errors=True)
            else:
                pass  # print(f"{parent} has data")

if remove_old_accum_directories:
    for parent, dirs, files in os.walk(data_dir, topdown=True):
        cur_dir = pathlib.Path(parent)
        # make sure we are three levels down (database/rowYYY/colXXX)
        if cur_dir.parent == data_dir:
            print(parent)
        if re.search("tmp.*_accum", cur_dir.name) and cur_dir.parent.parent == data_dir:
            print("  ",cur_dir.name)
            shutil.rmtree(parent, ignore_errors=True)
