import json
import platform
import time
import os
import sys
import pathlib

import numpy
from osgeo import gdal
gdal.UseExceptions()
from nbs.bruty.world_raster_database import WorldDatabase

"""
## PBG19     3 180    size GB : 11.45   file count:  27247
## PBG20     3 180    size GB:  14.98   file count:  32831
## PBC18     9   513       size GB:  27.71   file count:  87686
## PBC19    18  575        size GB:  87.47   file count: 186597
## PBC20    1  
## PBG14     6  400       size GB:  20.24   file count:  66713
# PBG15    18          size GB:  47.09   file count: 208974
## PBG16    12   367+       size GB:  40.77   file count: 122119
## PBG18     1  17        size GB:   0.64   file count:   2918
# PBB16    12          size GB:  48.24   file count: 121057
# PBB17    36          size GB:  15.13   file count: 441098
## PBB18     6  288        size GB:  20.02   file count:  70469 
## PBB19     3  364       size GB:  96.61   file count:  33379
## PBA       7  931        size GB:  26.12   file count:  70123
## PBD       1  61        size GB:   4.12   file count:   9482
## PBE17     1  75        size GB:   4.23   file count:  10052
# PBE18    36   889+       size GB: 128.72   file count: 203433
## PBE19     3   488       size GB:  17.37   file count:  23440
"""
runs = """
PBC utm19 Tile[1-3]
PBC utm19 Tile[4-9]
PBG utm15 _
PBG utm16 _
PBB utm16 _
PBB utm17 Tile[1-4]
PBG utm15 Tile5[6-9]
PBB utm17 Tile[5-9]
PBE utm18 Tile1
PBE utm18 Tile[2-9]

PBC utm18 _
PBC utm20 _
PBG utm14 _
PBG utm18 _
PBB utm18 _
PBB utm19 _
PBA _ _
PBD _ _
PBE utm17 _
PBE utm19 _

"""

def set_bruty_tiffs_orientation(bruty_path, positive_ns=False, just_size=False):
    """
    Flip the tiffs in the bruty database
    """
    sz = 0
    cnt = 0
    bruty_path = pathlib.Path(bruty_path)
    db = WorldDatabase.open(bruty_path)
    msg = ""
    changed_data = False
    for tx, ty in db.db.iterate_filled_tile_indices():
        data_dir = bruty_path.joinpath(f"{tx}", f"{ty}")
        try:
            meta = json.load(open(data_dir.joinpath("metadata.json")))
            min_y = meta["min_y"]
            max_y = meta["max_y"]
        except:
            continue
        for fname in os.scandir(data_dir):
            if str(fname.name).lower().endswith(".tif"):
                sz += pathlib.Path(fname).stat().st_size
                cnt += 1
                if not just_size:
                    try:
                        ds = gdal.Open(str(data_dir.joinpath(fname.name)), gdal.GA_Update)
                    except Exception as e:
                        continue
                    else:
                        x1, resx, dxy, y1, dyx, resy = ds.GetGeoTransform()
                        if positive_ns and y1 != min_y:
                            ds.SetGeoTransform((x1, resx, dxy, min_y, dyx, resy))
                            changed_data = True
                        if not positive_ns and y1 != max_y:
                            ds.SetGeoTransform((x1, resx, dxy, max_y, dyx, resy))
                            changed_data = True
                        if (resy < 0 and positive_ns) or (resy > 0 and not positive_ns):
                            for lyr in range(ds.RasterCount):
                                band = ds.GetRasterBand(lyr + 1)
                                arr = band.ReadAsArray()
                                band.WriteArray(numpy.flipud(arr))
                                del band
                            ds.SetGeoTransform((x1, resx, dxy, y1 + resy * arr.shape[0], dyx, -resy))
                            changed_data = True
                        else:
                            pass
                        del ds
    if False and not just_size and changed_data:
        try:
            db.create_vrt()
        except Exception as e:
            msg += f"!!! ------------ Error creating VRT {bruty_path}\n {str(e)}"
    return cnt, sz, msg


if __name__ == "__main__":
    db = WorldDatabase.open(tile_info.combine.data_location)
    db.create_vrt()

    """ arguments are production_branch (PBG) and utm zone (18n), if an underscore is put in for utm then all areas of a production branch will be processed.
     If "python flip_bruty_tiffs.py PB _" was run then all bruty databases would be processed"""
    if platform.system() == 'Windows':
        root = pathlib.Path(r"X:\bruty_databases")
    else:
        root = pathlib.Path(r"/OCS-S-HyperV17/NBS_Store2/bruty_databases/")
    ts = time.time()
    sz = 0
    cnt = 0
    if not sys.argv[1] or not sys.argv[2] or not sys.argv[3]:
        print("Please provide a production branch and utm zone and Tile numbers (or an underscore)")
        sys.exit()
    log_file = open(f"{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_conversion.log", "a")
    for p in root.glob(f"{sys.argv[1]}*{sys.argv[2]}*{sys.argv[3]}*"):  # probably run multiple times with different PBX paths
        if p.is_dir():
            print("reversing (as needed)", p)
            try:
                t = time.time()
                folder_cnt, folder_sz, msg = set_bruty_tiffs_orientation(p)
                cnt += folder_cnt
                sz += folder_sz
                if msg:
                    log_file.write(msg+"\n")
                print("finished in ", time.time()-t)
                log_file.write(f"finished {p}\n")
            except FileNotFoundError:
                log_file.write(f"!!! ------------ No Bruty DB in {p}\n")
            # set_bruty_tiffs_orientation(r"C:\Data\NBS\debug\combines\PBC_Northeast_utm18n_MLLW_Tile3_res4_qualified", positive_ns=False)
    print("total time (min)", (time.time()-ts)/60.0, "total size MB:", sz/1000000.0, "total count:", cnt)

