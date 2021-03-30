import pathlib

import numpy
from numba import jit
from osgeo import gdal
import rasterio

from HSTB.drivers import bag
from nbs.bruty import morton, world_raster_database
from nbs.bruty.raster_data import affine, inv_affine, affine_center

# @jit(nopython=True)

_debug = True

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

    source_tif = world_raster_database.SingleFile(epsg, x1, y1, x2, y2, res_x, res_y, output_path)
    if not _debug or not output_path.exists():
        source_tif.insert_survey_vr(vr_path)
        source_tif.export(output_path.joinpath("exported_area.tif"))
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
        #   get surrounding vr refinements as buffer
        neighbors = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                neighbors.append(vr.read_refinement(ri+i, rj+j))
        #   Get the output tif in that same area
        #   -- phase one just grab refinements area,
        #   -- phase two consider a second larger cache area that the calculation goes into and that larger cache writes to tif as needed
        r, c = numpy.indices(refinement.depth.shape)  # make indices into array elements that can be converted to x,y coordinates
        pts = numpy.array([r, c, refinement.depth, refinement.uncertainty]).reshape(4, -1)
        pts = pts[:, pts[2] != vr.fill_value]  # remove nodata points

        x, y = affine_center(pts[0], pts[1], *refinement.geotransform)
        export_rows, export_cols = inv_affine(transformed_x, transformed_y, *affine_transform)

    #   binary closing - radius dependant on refinement resolution
    #   clip back to area of center refinement
    #   -- determine where the other refinements would start/end
    #   Need scipy interpolate on data?
    #   -- interpolate the area then mask using the binary closing result
    #   merge into the upsampled tif
    #   -- make sure the offsets are computed correctly and write back the interpolated binary closed area
    # For entire tif - binary closing with size based on coarsest res of VR - this is the interpolated tif
    # Need scipy interpolate on data?


if __name__ == "__main__":
    upsample_and_interpolate_vr(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", r'G:\Data\NBS\Mississipi\test_upsample', 4)