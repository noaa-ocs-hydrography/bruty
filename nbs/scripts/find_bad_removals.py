from pathlib import Path
from nbs.bruty import world_raster_database
import traceback

# Replace with the path to your target folder
target_folder = Path(r"X:\bruty_databases")
bad_list = {}
f = open("d:\\debug\\bad_removals.txt", "w")
for entry in target_folder.iterdir():
  if entry.is_dir() and entry.name.upper().startswith("PB"):  #  and "PBB_Southeast_utm17n_MLLW_Tile106_res16_qualified".upper() in entry.name.upper()
    print(entry.name)
    try:
        db = world_raster_database.WorldDatabase.open(entry)
        tile_missing, tile_extra, contributor_missing, contributor_extra = db.validate()
        if tile_missing or tile_extra or contributor_missing or contributor_extra:
            bad_list[entry.name] = (tile_missing, tile_extra, contributor_missing, contributor_extra)
            f.write(f"{entry.name}: {bad_list[entry.name]}\n")
            f.flush()
    except:
        f.write(f"{entry.name} had exception: {traceback.format_exc()}\n")
print(bad_list)