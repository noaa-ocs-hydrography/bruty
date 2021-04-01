import pathlib

import numpy
from scipy.ndimage import binary_closing
from numba import jit
from osgeo import gdal
import rasterio

from HSTB.drivers import bag
from nbs.bruty import morton, world_raster_database
from nbs.bruty.raster_data import affine, inv_affine, affine_center

# @jit(nopython=True)

_debug = True

def ellipse_mask(r, c):
    center = numpy.array(((r-1) / 2, (c-1) / 2))
    accept_radius = 1.0
    # if the grid is even the center falls in between cells which would make the last row/column have all False
    # expanding the radius slightly (one cell in the other dimension) will create two True values in the edge row/col
    if not r % 2:
        accept_radius = max(accept_radius, (1+(r-1)**2)/((r-1)**2))
    if not c % 2:
        accept_radius = max(accept_radius, (1+(c-1)**2)/((c-1)**2))

    Y, X = numpy.ogrid[:r, :c]
    # normalize the distances since we are allowing rectangular grids
    dist2_from_center = ((Y - center[0]) / center[0]) **2 + ((X - center[1]) / center[1])**2

    mask = dist2_from_center <= accept_radius  # max(radius)**2
    return mask


def upsample_and_interpolate_vr(vr_path, output_dir, res):
    output_path = pathlib.Path(output_dir)
    vr = bag.VRBag(vr_path, mode='r')
    epsg = rasterio.crs.CRS.from_string(vr.horizontal_crs_wkt).to_epsg()
    good_refinements = numpy.argwhere(vr.get_valid_refinements())  # bool matrix of which refinements have data
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

    source_db = world_raster_database.CustomArea(epsg, x1, y1, x2, y2, res_x, res_y, output_path)
    if not _debug or not output_path.exists():
        source_db.insert_survey_vr(vr_path)
        source_db.export(output_path.joinpath("exported_source.tif"))
        source_db.export(output_path.joinpath("exported_upsample.tif"))
    source_ds = gdal.Open(output_path.joinpath("exported_source.tif"))
    upsample_ds = gdal.Open(output_path.joinpath("exported_upsample.tif"))
    output_geotransform = source_ds.GetGeotransform()
    # For refinement in vr
    # -- phase one, just iterate across and cache tiles, phase two use morton and cache (more complex)
    #   get surrounding vr refinements areas as buffer
    #   Get the output tif in that same area
    #   -- phase one just grab refinement area,
    #   -- phase two consider a second larger cache area that the calculation goes into and that larger cache writes to tif as needed
    #   binary closing - radius dependant on refinement resolution
    #   clip back to area of center refinement
    #   -- determine where the other refinements would start/end
    #   Need scipy interpolate on data?
    #   -- interpolate the area then mask using the binary closing result
    #   merge into the upsampled tif
    #   -- make sure the offsets are computed correctly and write back the interpolated binary closed area
    # For entire tif - binary closing with size based on coarsest res of VR - this is the interpolated tif
    # Need scipy interpolate on data?

    # For refinement in vr
    # -- phase one, just iterate read all 9 refinements,
    # -- phase two store the old refinements and see if we can re-use (often 6 will already be loaded), could even keep extended list fifo buffer
    for ri, rj in sorted_refinement_indices:
        refinement = vr.read_refinement(ri, rj)
        # get surrounding vr refinements as buffer
        neighbors = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                neighbors.append(vr.read_refinement(ri+i, rj+j))
        #   Get the output tif in that same area
        #   -- phase one just grab refinements area,
        #   -- phase two consider a second larger cache area that the calculation goes into and that larger cache writes to tif as needed
        x, y, pts = refinement.get_xy_pts_arrays()
        export_rows, export_cols = inv_affine(x, y, *output_geotransform)

        #   binary closing - radius dependant on refinement resolution
        # the rows and columns are the ratio of the original over the output
        # -- so 16m original and 8m output would need a 1 pixel closing
        # 16 vs 5 would need 3 pixel as max dist would be 4 pixels
        # 16 vs 4 would need 3 pixel as max dist would be 4
        # 16 vs 3 would need 5 pixel as max dist is 6 pixels
        # so (ceil(orig/new) - 1) would be the size of the ellipse mask needed
        # remember that the ellipse mask goes right and left, so covers halfway to next point but that point also has a mask so they'd meet
        cols = numpy.ceil(refinement.geotransform[1] / output_geotransform[1])
        rows = numpy.ceil(refinement.geotransform[5] / output_geotransform[5])
        if rows > 1 or cols > 1:
            ellipse = ellipse_mask(rows, cols)
            interp = binary_closing(refinement.depth!=vr.fill_value, ellipse)
        else:
            raise Exception("no interp needed")
        #   clip back to area of center refinement
        #   -- determine where the other refinements would start/end
        #   Need scipy interpolate on data?
        #   -- interpolate the area then mask using the binary closing result
        #   merge into the upsampled tif
        #   -- make sure the offsets are computed correctly and write back the interpolated binary closed area
    # For entire tif - binary closing with size based on coarsest res of VR - this is the interpolated tif
    # Need scipy interpolate on data?


if __name__ == "__main__":
    print(ellipse_mask(5,5))
    print(ellipse_mask(4,4))
    print(ellipse_mask(7,4))
    print(ellipse_mask(7,4))

    # upsample_and_interpolate_vr(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", r'G:\Data\NBS\Mississipi\test_upsample', 4)