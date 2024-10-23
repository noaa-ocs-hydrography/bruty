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
# PBG19     3 hours    size GB: 11.45  file count:  27247
# PBG20     3 hours    size GB: 14.98  file count:  32831
PBC18     9          size GB: 27.71  file count:  87686
PBC19    18          size GB: 87.47  file count: 186597
PBG14     6          size GB: 20.24  file count:  66713
PBG15    18          size GB: 47.09  file count: 208974
PBG16    12          size GB: 40.77  file count: 122119
PBG18
PBB16 
PBB17    30          size GB: 15.13  file count: 441098
PBB18
PBB19
PBD
PBE
PBA
"""
def set_bruty_tiffs_orientation(bruty_path, positive_ns=False, just_size=True):
    """
    Flip the tiffs in the bruty database
    """
    sz = 0
    cnt = 0
    bruty_path = pathlib.Path(bruty_path)
    db = WorldDatabase.open(bruty_path)
    msg = ""
    for tx, ty in db.db.iterate_filled_tile_indices():
        data_dir = bruty_path.joinpath(f"{tx}", f"{ty}")
        for fname in os.scandir(data_dir):
            if ".tif" in str(fname.name):
                sz += pathlib.Path(fname).stat().st_size
                cnt += 1
                if not just_size:
                    ds = gdal.Open(str(data_dir.joinpath(fname.name)), gdal.GA_Update)
                    x1, resx, dxy, y1, dyx, resy = ds.GetGeoTransform()
                    if (resy < 0 and positive_ns) or (resy > 0 and not positive_ns):
                        for lyr in range(ds.RasterCount):
                            band = ds.GetRasterBand(lyr + 1)
                            arr = band.ReadAsArray()
                            band.WriteArray(numpy.flipud(arr))
                            ds.SetGeoTransform((x1, resx, dxy, y1+resy*arr.shape[0], dyx, -resy))
                            del band
                        del ds
                    else:
                        pass
    if not just_size:
        try:
            db.create_vrt()
        except Exception as e:
            msg += f"!!! ------------ Error creating VRT {bruty_path}\n {str(e)}"
    return cnt, sz, msg


if __name__ == "__main__":
    """ arguments are production_branch (PBG) and utm zone (18n), if an underscore is put in for utm then all areas of a production branch will be processed.
     If "python flip_bruty_tiffs.py PB _" was run then all bruty databases would be processed"""
    if platform.system() == 'Windows':
        root = pathlib.Path(r"X:\bruty_databases")
    else:
        root = pathlib.Path(r"/OCS-S-HyperV17/NBS_Store2/bruty_databases/")
    ts = time.time()
    sz = 0
    cnt = 0
    if not sys.argv[1] or not sys.argv[2]:
        print("Please provide a production branch and utm zone")
        sys.exit()
    log_file = open(f"{sys.argv[1]}_{sys.argv[2]}_conversion.log", "a")
    for p in root.glob(f"{sys.argv[1]}_*{sys.argv[2]}*"):  # probably run multiple times with different PBX paths
        if p.is_dir():
            print("reversing (as needed)", p)
            # break
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

