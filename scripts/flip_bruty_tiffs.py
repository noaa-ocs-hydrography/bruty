import os
import pathlib

import numpy
from osgeo import gdal
gdal.UseExceptions()
from nbs.bruty.world_raster_database import WorldDatabase


def set_bruty_tiffs_orientation(bruty_path, positive_ns=False):
    """
    Flip the tiffs in the bruty database
    """
    bruty_path = pathlib.Path(bruty_path)
    db = WorldDatabase.open(bruty_path)
    for tx, ty in db.db.iterate_filled_tile_indices():
        data_dir = bruty_path.joinpath(f"{tx}\\{ty}")
        for fname in os.scandir(data_dir):
            if ".tif" in str(fname.name):
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
    try:
        db.create_vrt()
    except Exception as e:
        print("Error creating VRT", bruty_path, e)


if __name__ == "__main__":
    root = pathlib.Path(r"C:\Data\NBS\debug\combines")
    for p in root.glob("*"):
        if p.is_dir():
            print("reversing (as needed)", p)
            try:
                set_bruty_tiffs_orientation(p)
            except FileNotFoundError:
                print("No Bruty DB in", p)
            # set_bruty_tiffs_orientation(r"C:\Data\NBS\debug\combines\PBC_Northeast_utm18n_MLLW_Tile3_res4_qualified", positive_ns=False)
