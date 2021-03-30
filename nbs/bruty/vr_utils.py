import pathlib

import numpy
from numba import jit
from osgeo import gdal
import rasterio

from HSTB.drivers import bag
from nbs.bruty import morton, world_raster_database

# @jit(nopython=True)

def upsample_and_interpolate_vr(vr_path, output_dir, res):
    output_path = pathlib.Path(output_dir)
    vr = bag.VRBag(vr_path, mode='r')
    epsg = rasterio.crs.CRS.from_string(vr.horizontal_crs_wkt).to_epsg()
    good_refinements = vr.get_valid_refinements()  # bool matrix of which refinements have data
    mort = morton.interleave2d_64(good_refinements.T)
    sorted_refinement_indices = good_refinements[numpy.lexsort([mort])]
    try:
        res_x = res[0]
        res_y = res[1]
    except (TypeError, IndexError):
        res_x = res_y = res
    y2 = vr.maxy
    y1 = vr.miny
    x2 = vr.maxx
    x1 = vr.minx

    source_tif = world_raster_database.SingleFile(epsg, x1, y1, x2, y2, res_x, res_y, output_dir)
    source_tif.insert_survey_vr(vr_path)
    source_tif.export(output_path.joinpath("exported_area.tif"))
    # For refinement in vr
    #   get surrounding vr refinements areas as buffer
    #   binary closing - radius dependant on refinement resolution
    #   clip back to area of center refinement
    #   Need scipy interpolate on data?
    #   merge into the upsampled tif
    # For entire tif - binary closing with size based on coarsest res of VR - this is the interpolated tif
    # Need scipy interpolate on data?


if __name__ == "__main__":
    upsample_and_interpolate_vr(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", r'G:\Data\NBS\Mississipi\test_upsample', 4)