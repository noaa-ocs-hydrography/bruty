import os
import re
import shutil
import pathlib

from nbs.bruty.world_raster_database import WorldDatabase

data_dir = pathlib.Path(r"X:\bruty_databases")
data_dir = pathlib.Path(r"D:\debug\combines")
qual = [r"([\\/])[\\/]nos.noaa[\\/]OCS[\\/]HSD[\\/]Projects[\\/]NBS[\\/]NBS_Data_Qualified",
r"\\\\OCS-S-HyperV17\\NBS_Store2\\fuse\\NBS_Data_Qualified"]  # remember to give extra backslashes for regular expressions
sens = [r"([\\/])[\\/]nos.noaa[\\/]OCS[\\/]HSD[\\/]Projects[\\/]NBS[\\/]NBS_Data_Sensitive",
r"\\\\OCS-S-HyperV17\\NBS_Store2\\fuse\\NBS_Data_Sensitive"]
unqual = [r"([\\/])[\\/]nos.noaa[\\/]OCS[\\/]HSD[\\/]Projects[\\/]NBS[\\/]NBS_Data_Unqualified",
r"\\\\OCS-S-HyperV17\\NBS_Store2\\fuse\\NBS_Data_Unqualified"]

for search_root, dirs, files in os.walk(data_dir, topdown=True):
    break
for subdir in dirs:
    if "PBB" not in subdir or "utm16" not in subdir:
        continue
    db_dir = pathlib.Path(search_root).joinpath(subdir)
    print(db_dir)
    try:
        db = WorldDatabase.open(db_dir)
    except FileNotFoundError:
        pass
    else:
        for old_path, new_path in (qual, sens, unqual):
            db.revise_root_paths(old_path, new_path)
        raise Exception("This script is not finished yet")