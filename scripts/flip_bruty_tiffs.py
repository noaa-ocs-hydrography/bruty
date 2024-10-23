import platform
import time
import os
import pathlib

import numpy
from osgeo import gdal
gdal.UseExceptions()
from nbs.bruty.world_raster_database import WorldDatabase


def set_bruty_tiffs_orientation(bruty_path, positive_ns=False, just_size=True):
    """
    Flip the tiffs in the bruty database
    """
    sz = 0
    cnt = 0
    bruty_path = pathlib.Path(bruty_path)
    db = WorldDatabase.open(bruty_path)
    for tx, ty in db.db.iterate_filled_tile_indices():
        data_dir = bruty_path.joinpath(f"{tx}", f"{ty}")
        for fname in os.scandir(data_dir):
            if ".tif" in str(fname.name):
                sz += pathlib.Path(fname).stat().st_size
                cnt+=1
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
            print("Error creating VRT", bruty_path, e)
    return cnt, sz


if __name__ == "__main__":
    if platform.system() == 'Windows':
        root = pathlib.Path(r"X:\bruty_databases")
    else:
        root = pathlib.Path(r"/OCS-S-HyperV17/NBS_Store2/bruty_databases/")
    ts = time.time()
    sz = 0
    cnt = 0
    for p in root.glob("PBG_*20n*"):  # probably run multiple times with different PBX paths
        if p.is_dir():
            print("reversing (as needed)", p)
            # break
            try:
                t = time.time()
                folder_cnt, folder_sz = set_bruty_tiffs_orientation(p)
                cnt += folder_cnt
                sz += folder_sz
                print("finished in ", time.time()-t)
            except FileNotFoundError:
                print("No Bruty DB in", p)
            # set_bruty_tiffs_orientation(r"C:\Data\NBS\debug\combines\PBC_Northeast_utm18n_MLLW_Tile3_res4_qualified", positive_ns=False)
    print("total time (min)", (time.time()-ts)/60.0, "total size MB:", sz/1000000.0, "total count:", cnt)

