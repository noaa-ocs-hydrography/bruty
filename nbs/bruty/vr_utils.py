from collections import OrderedDict  # this is the default starting around python 3.7
import os
import pathlib
import timeit
import functools

import numpy
from scipy.ndimage import binary_closing, binary_dilation
from skimage.draw import polygon
from numba import jit
from osgeo import gdal
import rasterio

from HSTB.drivers import bag
from nbs.bruty import morton, world_raster_database
from nbs.bruty.raster_data import affine, inv_affine, affine_center

_debug = False
if _debug:  # turn off jit --
    """ Disabling JIT compilation
        In order to debug code, it is possible to disable JIT compilation, 
        which makes the jit decorator (and the njit decorator) act as if they perform no operation, 
        and the invocation of decorated functions calls the original Python function instead of a compiled version. 
        This can be toggled by setting the NUMBA_DISABLE_JIT environment variable to 1."""
    # this doesn't seem to take effect, perhaps needs to be before numba is loaded,
    # but can we guarantee that if another module imports it first?
    # os.environ['NUMBA_DISABLE_JIT'] = '1'

    # so override the decorator to just disable this file -- this should be effectively be a no-op
    def jit(*args, **kwargs):
        def decorator_repeat(func):
            @functools.wraps(func)
            def wrapper_repeat(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_repeat

        return decorator_repeat


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
    # don't allow divide by zero
    dist2_from_center = ((Y - center[0]) / max(center[0], 0.5)) **2 + ((X - center[1]) / max(center[1], 0.5))**2

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
    if not _debug or not output_path.joinpath("0").exists():
        source_db.insert_survey_vr(vr_path)
        source_db.export(output_path.joinpath("exported_source.tif"))
        source_db.export(output_path.joinpath("exported_upsample.tif"))
    source_ds = gdal.Open(str(output_path.joinpath("exported_source.tif")))
    source_depth = source_ds.GetRasterBand(1)
    empty_val = source_depth.GetNoDataValue()
    upsample_ds = gdal.Open(str(output_path.joinpath("exported_upsample.tif")))
    output_geotransform = source_ds.GetGeoTransform()
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
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                neighbors.append(vr.read_refinement(ri+i, rj+j))
        #   Get the output tif in that same area
        #   -- phase one just grab refinements area,
        #   -- phase two consider a second larger cache area that the calculation goes into and that larger cache writes to tif as needed
        x, y, pts = refinement.get_xy_pts_arrays()
        export_rows, export_cols = inv_affine(x, y, *output_geotransform)



        if 0:
            #   binary closing - radius dependant on refinement resolution
            # the rows and columns are the ratio of the original over the output
            # -- so 16m original and 8m output would need a 1 pixel closing
            # 16 vs 5 would need 3 pixel as max dist would be 4 pixels (3 empties in between)
            # 16 vs 4 would need 3 pixel as max dist would be 4
            # 16 vs 3.5 would need 4 pixel as max dist would be 5  (4 empties)
            # 16 vs 3 would need 5 pixel as max dist is 6 pixels  (5 empties)
            # so (ceil(orig/new)) would be the max distance
            # (ceil(orig/new)) -1) would be empty cells to fill
            # so half that and round up then double it and add one for the center
            # ceil((ceil(orig/new) -1) /2) *2 +1 would be the size of the ellipse mask needed
            # remember that the ellipse mask goes right and left, so covers halfway to next point but that point also has a mask so they'd meet
            empty_cols = numpy.ceil(numpy.abs(refinement.geotransform[1] / output_geotransform[1])) - 1
            empty_rows = numpy.ceil(numpy.abs(refinement.geotransform[5] / output_geotransform[5])) - 1
            if empty_rows > 0 or empty_cols > 0:
                # multiply by 1.415 = sqrt(2) to account for the diagonal which wouldn't close otherwise
                rows = numpy.ceil(numpy.ceil(empty_rows / 2)*1.415) * 2 + 1  # numpy.ceil(empty_rows * 1.415 / 2) * 2 + 1
                cols = numpy.ceil(numpy.ceil(empty_cols / 2)*1.415) * 2 + 1  # numpy.ceil(empty_cols * 1.415 / 2) * 2 + 1
                depths = source_depth.ReadAsArray(int(min(export_cols)), int(min(export_rows)),  # x offset, y offset
                                                  int(max(export_cols) - min(export_cols) + 1),  # x size
                                                  int(max(export_rows) - min(export_rows) + 1))  # y size
                print("adding hole for testing")
                depths[3, 2]=numpy.nan
                ellipse = ellipse_mask(rows, cols)
                if numpy.isnan(empty_val):
                    has_depth = ~numpy.isnan(depths)
                else:
                    has_depth = depths != empty_val
                interp_mask = binary_closing(has_depth, ellipse)
                print(interp_mask * 1)
            else:
                pass
            # raise Exception("no interp needed")
        #   clip back to area of center refinement
        #   -- determine where the other refinements would start/end
        #   Need scipy interpolate on data?
        #   -- interpolate the area then mask using the binary closing result
        #   merge into the upsampled tif
        #   -- make sure the offsets are computed correctly and write back the interpolated binary closed area
    # For entire tif - binary closing with size based on coarsest res of VR - this is the interpolated tif
    # Need scipy interpolate on data?

def test_array(n, el, hole=False):
    e = int(numpy.ceil(el/2))
    a = numpy.zeros((3*n + e*2+1, 3*n+e*2+1), dtype=numpy.int8)
    a[e,e] = 1; a[e,e+n] = 1; a[e,e+2*n] = 1; a[e,e+3*n] = 1
    a[e+n, e] = 1; a[e+n, e+n] = 1; a[e+n, e+2*n] = 1; a[e+n, e+3*n] = 1
    a[e + 2* n, e] = 1; a[e + 2*n, e + n] = 1*hole; a[e + 2*n, e + 2 * n] = 1; a[e + 2*n, e + 3 * n] = 1
    a[e + 3* n, e] = 1; a[e + 3*n, e + n] = 1; a[e + 3*n, e + 2 * n] = 1; a[e + 3*n, e + 3 * n] = 1
    return a

def test_close(n, e):
    a = test_array(n, e)
    s = ellipse_mask(e,e)
    print(a)
    print(s * 1)
    print(binary_dilation(a,s) * 1)
    print(binary_closing(a, s) * 1)
    s = scipy.ndimage.generate_binary_structure(2, 1)
    ns = scipy.ndimage.iterate_structure(s, int(e/2))
    print(ns*1)
    print(binary_dilation(a,ns) * 1)
    print(binary_closing(a, ns) * 1)

def test_refinement(n, hole=True):
    upsampled = numpy.zeros((3*(n+1)+1, 3*(n+1)+1), dtype=numpy.int8)
    X, Y = numpy.ogrid[:4, :4]
    x, y = numpy.meshgrid(X, Y)
    mapping = numpy.array((y, x, numpy.ones((4,4))*2), dtype=numpy.int8)  # array of row, column, depth
    # make it non-square
    mapping[0] *= n
    mapping[1] *= n+1
    if hole:
        mapping[2][1,1] = 0
        mapping[2][2,3] = 0
    set_pts = mapping.reshape(3, -1)
    upsampled[set_pts[0], set_pts[1]] = set_pts[2]
    return upsampled, mapping

def triangle_closing_old(upsampled, mapping):
    for r in range(mapping.shape[1]-1):
        for c in range(mapping.shape[2]-1):
            i, j, d = mapping[:, r, c]
            i1, j1, d1 = mapping[:, r+1, c]
            i2, j2, d2 = mapping[:, r, c+1]
            i3, j3, d3 = mapping[:, r+1, c+1]
            if d and d1 and d2 and d3:
                upsampled[i:i3+1, j:j3+1] = 1
            elif d and d1 and d2:
                rr, cc = polygon((i, i1, i2), (j, j1, j2))
                upsampled[rr, cc] = 1
            elif d and d1 and d3:
                rr, cc = polygon((i, i1, i3), (j, j1, j3))
                upsampled[rr, cc] = 1
            elif d and d2 and d3:
                rr, cc = polygon((i, i2, i3), (j, j2, j3))
                upsampled[rr, cc] = 1
            elif d1 and d2 and d3:
                rr, cc = polygon((i1, i2, i3), (j1, j2, j3))
                upsampled[rr, cc] = 1

@jit (nopython=True)
def triangle_closing_skimage(upsampled, mapping):
    polygons = []
    for r in range(mapping.shape[1]-1):
        for c in range(mapping.shape[2]-1):
            i, j, d = mapping[:, r, c]
            i1, j1, d1 = mapping[:, r+1, c]
            i2, j2, d2 = mapping[:, r, c+1]
            i3, j3, d3 = mapping[:, r+1, c+1]
            if d and d1 and d2 and d3:
                upsampled[i:i3+1, j:j3+1] = 1
            elif d and d1 and d2:
                polygons.append(((i, i1, i2), (j, j1, j2)))
            elif d and d1 and d3:
                polygons.append(((i, i1, i3), (j, j1, j3)))
            elif d and d2 and d3:
                polygons.append(((i, i2, i3), (j, j2, j3)))
            elif d1 and d2 and d3:
                polygons.append(((i1, i2, i3), (j1, j2, j3)))
    return polygons

def test_grid(out_res, ref_res, hole=True):
    # 64m supercells, resampled at out_res with 3x3 ref_res
    super_size = 64
    output_cells = int(numpy.ceil(super_size*3/out_res))+1
    upsampled = numpy.zeros((output_cells, output_cells), dtype=numpy.int8)
    refinements = [[None, None, None],[None, None, None],[None, None, None]]
    transform = [0, out_res, 0, 0, 0, out_res]
    for i in range(3):
        for j in range(3):
            try:
                resx, resy = ref_res[i][j]
            except TypeError:
                resx = resy = ref_res[i][j]
            nx = int(super_size/resx)
            ny = int(super_size / resy)
            X, Y = numpy.ogrid[:nx, :ny]
            x, y = numpy.meshgrid(X, Y)
            # move for which supergrid it is and then center the refinement data inside the supergrid
            x = x * resx + 64 * j + (super_size % resx) / 2
            y = y * resy + 64 * i + (super_size % resy) / 2
            r, c = inv_affine(x, y, *transform)
            mapping = numpy.array((r, c, y, x, numpy.ones((ny,nx))*2), dtype=float)  # array of row, column, depth
            if hole:
                try:
                    mapping[4][1,1] = 0
                    mapping[4][3,0] = 0
                except IndexError:
                    pass  # refinement probably too small for hole location
            set_pts = mapping.reshape(5, -1)
            upsampled[set_pts[0].astype(int), set_pts[1].astype(int)] = set_pts[4].astype(int)
            refinements[i][j] = mapping
    return upsampled, refinements



@jit (nopython=True)
def triangle_closing(upsampled, mapping):
    for r in range(mapping.shape[1]-1):
        for c in range(mapping.shape[2]-1):
            i, j, x, y, d = mapping[:, r, c]
            i1, j1, x1, y1, d1 = mapping[:, r+1, c]
            i2, j2, x2, y2, d2 = mapping[:, r, c+1]
            i3, j3, x3, y3, d3 = mapping[:, r+1, c+1]
            if d and d1 and d2 and d3:
                upsampled[i:i3+1, j:j3+1] = 1
            elif d and d1 and d2:  # top left
                for r_i in range(i, i1+1):
                    if i1 != i:  # avoid divide by zero - range is only one scan line
                        end_j = numpy.trunc(j2 - (j2 - j) * (r_i - i)/(i1-i))
                    else:
                        end_j = j2
                    upsampled[r_i, j:int(end_j)+1] = 1
            elif d and d1 and d3:  # bottom left
                for r_i in range(i, i1+1):
                    if i1 != i:  # avoid divide by zero - range is only one scan line
                        end_j = numpy.trunc(j + (j3 - j) * (r_i - i)/(i1-i))
                    else:
                        end_j = j3
                    upsampled[r_i, j:int(end_j)+1] = 1
            elif d and d2 and d3:  # top right
                for r_i in range(i, i3+1):
                    if i3 != i:  # avoid divide by zero - range is only one scan line
                        start_j = numpy.ceil(j + (j2 - j) * (r_i - i)/(i3-i))
                    else:
                        start_j = j
                    upsampled[r_i, int(start_j):j2+1] = 1
            elif d1 and d2 and d3:  # bottom right
                for r_i in range(i2, i3+1):
                    if i3 != i2:  # avoid divide by zero - range is only one scan line
                        start_j = numpy.ceil(j2 - (j2 - j) * (r_i - i2)/(i3-i2))
                    else:
                        start_j = j
                    upsampled[r_i, int(start_j):j2+1] = 1

@jit (nopython=True)
def draw_triangle(matrix, pts):
    # pts = numpy.array(((ai, aj), (bi, bj), (ci, cj)))
    top, middle, bottom = numpy.argsort(pts[:,0])
    if pts[top][0] != pts[middle][0]:
        slope_tm = (pts[top][1] - pts[middle][1]) / (pts[top][0] - pts[middle][0])
        slope_tb = (pts[top][1] - pts[bottom][1]) / (pts[top][0] - pts[bottom][0])
        for row in range(pts[top][0], pts[middle][0]+1):
            c1, c2 = slope_tm * (row-pts[top][0]) + pts[top][1], slope_tb * (row-pts[top][0]) + pts[top][1]
            j1 = int(numpy.ceil(min(c1,c2)))
            j2 = int(numpy.trunc(max(c1,c2)))
            matrix[row, j1:j2+1] = 4
    else:
        j1 = min(pts[top][1], pts[middle][1])
        j2 = max(pts[top][1], pts[middle][1])
        matrix[pts[top][0], j1:j2+1] = 5
    if pts[middle][0] != pts[bottom][0]:
        slope_tb = (pts[top][1] - pts[bottom][1]) / (pts[top][0] - pts[bottom][0])
        slope_mb = (pts[middle][1] - pts[bottom][1]) / (pts[middle][0] - pts[bottom][0])
        for row in range(pts[middle][0], pts[bottom][0]+1):
            c1, c2 = slope_mb * (row-pts[middle][0]) + pts[middle][1], slope_tb * (row-pts[top][0]) + pts[top][1]
            j1 = int(numpy.ceil(min(c1,c2)))
            j2 = int(numpy.trunc(max(c1,c2)))
            matrix[row, j1:j2+1] = 7
    else:
        j1 = min(pts[bottom][1], pts[middle][1])
        j2 = max(pts[bottom][1], pts[middle][1])
        matrix[pts[bottom][0], j1:j2+1] = 6

@jit (nopython=True)
def close_quad(matrix, ul, ur, ll, lr):
    # numba doesn't like indices that are tuples like -- draw_triangle(matrix, pts[(0,1,3)])
    # also doesn't like making arrays from arrays it looks like -- numpy.array([ul, ll, ur], dtype=numpy.int32)
    # but it will make an array from lists of numbers like -- numpy.array([[li1, lj1], [li2, lj2], [ri1, rj1]], dtype=numpy.int32))
    li1, lj1, ld1 = ul
    li2, lj2, ld2 = ll
    ri1, rj1, rd1 = ur
    ri2, rj2, rd2 = lr
    draw_ll, draw_ur, draw_lr, draw_ul = False, False, False, False
    if ld1 and ld2 and rd1 and rd2:
        draw_ul = True
        draw_lr = True
    elif ld1 and ld2 and rd1:
        draw_ul = True
    elif ld1 and ld2 and rd2:
        draw_ll = True
    elif rd1 and rd2 and ld1:
        draw_ur = True
    elif rd1 and rd2 and ld2:
        draw_lr = True
    if draw_ul:
        tri = numpy.array([[li1, lj1], [li2, lj2], [ri1, rj1]], dtype=numpy.int32)  # [uld, lld, urd]
        draw_triangle(matrix, tri)
    if draw_ll:
        tri = numpy.array([[li1, lj1], [li2, lj2], [ri2, rj2]], dtype=numpy.int32)  # [uld, lld, lrd]
        draw_triangle(matrix, tri)
    if draw_lr:
        tri = numpy.array([[li2, lj2], [ri2, rj2], [ri1, rj1]], dtype=numpy.int32)  # [lld, lrd, urd]
        draw_triangle(matrix, tri)
    if draw_ur:
        tri = numpy.array([[li1, lj1],[ri2, rj2], [ri1, rj1]], dtype=numpy.int32)  # [uld, lrd, urd]
        draw_triangle(matrix, tri)

@jit (nopython=True)
def close_refinements(upsampled, left_or_top, right_or_bottom, max_dist, horz=True):
    # closes the gap between two refinements - have to specify if they are horizontal or vertical
    rb_index = 0
    lt_last = -1
    rb_last = 0
    lt_index = 0
    # @todo make this compare x,y position vs max_dist instead of upsampled indices
    ri, rj, x,y, rd = right_or_bottom[:, rb_index, rb_last]
    li, lj, x,y, ld = left_or_top[:, lt_index, lt_last]
    if rj - lj < max_dist:  # make sure the gap between supercells is less than max_dist for upsampling
        more_data = True
        if horz:
            max_lt = left_or_top.shape[1] - 2
            max_rb = right_or_bottom.shape[1] - 2
        else:
            max_lt = left_or_top.shape[2] - 2
            max_rb = right_or_bottom.shape[2] - 2
        if max_lt < 0 or max_rb < 0:  # @todo not sure how we want to handle a refinement that only has one cell
            more_data = False
        while more_data:
            if horz:
                ri1, rj1, _x, _y, rd1 = right_or_bottom[:, rb_index, rb_last]
                ri2, rj2, _x, _y, rd2 = right_or_bottom[:, rb_index+1, rb_last]
                li1, lj1, _x, _y, ld1 = left_or_top[:, lt_index, lt_last]
                li2, lj2, _x, _y, ld2 = left_or_top[:, lt_index+1, lt_last]
                lt1 = li1
                lt2 = li2
                rb1 = ri1
                rb2 = ri2
            else:
                li1, lj1, _x, _y, ld1 = right_or_bottom[:, rb_last, rb_index]
                ri1, rj1, _x, _y, rd1 = right_or_bottom[:, rb_last, rb_index+1]
                li2, lj2, _x, _y, ld2 = left_or_top[:, lt_last, lt_index]
                ri2, rj2, _x, _y, rd2 = left_or_top[:, lt_last, lt_index+1]
                lt1 = lj2
                lt2 = rj2
                rb1 = lj1
                rb2 = rj1

            # @todo check the x/y distance so we don't close over a long diagonal?  Think of a 32 ro 64m res in  a 64m supergrid agains a 1m grid
            close_quad(upsampled, (ri1, rj1, rd1), (ri2, rj2, rd2), (li1, lj1, ld1), (li2, lj2, ld2))
            # @fixme -- I think we can move the indices twice,
            #   moving once duplicates a triangle
            #   -- can only break on the first loop in case there is one triangle left
            if lt1 >= rb2 and rb_index < max_rb:  # left points both above right
                rb_index += 1
            elif rb1 >= lt2 and lt_index < max_lt: # right points both above left
                lt_index += 1
            elif lt1 >= rb1 and lt2 <= rb2 and lt_index < max_lt:  # left is between right
                lt_index += 1
            elif rb1 >= lt1 and rb2 <= lt2 and rb_index < max_rb:  # right is between left
                rb_index += 1
            elif rb1 >= lt1 and lt_index < max_lt:  # left is lower
                lt_index += 1
            elif lt1 >= rb1 and rb_index < max_rb:  # right is lower
                rb_index += 1
            elif lt_index < max_lt:  # otherwise just move a row since one side is maxed out
                lt_index +=1
            elif rb_index < max_rb:
                rb_index += 1
            else:
                more_data = False
                break

# @jit (nopython=True)
def numb(upsampled, refinements):
    # @todo reduce this to only doing the upper and right boundaries.
    #    the other edges will be covered when processing the neighbors.
    #    corners need to be done since an empty refinement would not need to fill edges but the corner still needs to be processed.
    if 1:
        # fill the boundaries between all refinements in the 3x3 set.
        close_refinements(upsampled, refinements[0][0].astype(int), refinements[0][1].astype(int), 100)
        close_refinements(upsampled, refinements[0][1].astype(int), refinements[0][2].astype(int), 100)
        close_refinements(upsampled, refinements[1][0].astype(int), refinements[1][1].astype(int), 100)
        close_refinements(upsampled, refinements[1][1].astype(int), refinements[1][2].astype(int), 100)
        close_refinements(upsampled, refinements[2][0].astype(int), refinements[2][1].astype(int), 100)
        close_refinements(upsampled, refinements[2][1].astype(int), refinements[2][2].astype(int), 100)
        close_refinements(upsampled, refinements[0][0].astype(int), refinements[1][0].astype(int), 100, horz=False)
        close_refinements(upsampled, refinements[0][1].astype(int), refinements[1][1].astype(int), 100, horz=False)
        close_refinements(upsampled, refinements[0][2].astype(int), refinements[1][2].astype(int), 100, horz=False)
        close_refinements(upsampled, refinements[1][0].astype(int), refinements[2][0].astype(int), 100, horz=False)
        close_refinements(upsampled, refinements[1][1].astype(int), refinements[2][1].astype(int), 100, horz=False)
        close_refinements(upsampled, refinements[1][2].astype(int), refinements[2][2].astype(int), 100, horz=False)
    if 1:
        # fill in the corners by getting a corner point from each refinement
        # lower left of center refinement
        ur = refinements[1][1].astype(int)[(0,1,-1), 0, 0]
        ll = refinements[0][0].astype(int)[(0,1,-1), -1, -1]
        lr = refinements[0][1].astype(int)[(0,1,-1), -1, 0]
        ul = refinements[1][0].astype(int)[(0,1,-1), 0, -1]
        close_quad(upsampled, lr, ul, ur, ll)
        # upper right of center refinement
        ll = refinements[1][1].astype(int)[(0,1,-1), -1, -1]
        ul = refinements[2][1].astype(int)[(0,1,-1), 0, -1]
        lr = refinements[1][2].astype(int)[(0,1,-1), -1, 0]
        ur = refinements[2][2].astype(int)[(0,1,-1), 0, 0]
        close_quad(upsampled, lr, ul, ur, ll)
        # lower right of center refinement
        ul = refinements[1][1].astype(int)[(0,1,-1), 0, -1]
        lr = refinements[0][2].astype(int)[(0,1,-1), -1, 0]
        ll = refinements[0][1].astype(int)[(0,1,-1), -1, -1]
        ur = refinements[1][2].astype(int)[(0,1,-1), 0, 0]
        close_quad(upsampled, lr, ul, ur, ll)
        # upper left of center refinement
        lr = refinements[1][1].astype(int)[(0,1,-1), -1, 0]
        ul = refinements[2][0].astype(int)[(0,1,-1), 0, -1]
        ur = refinements[2][1].astype(int)[(0,1,-1), 0, 0]
        ll = refinements[1][0].astype(int)[(0,1,-1), -1, -1]
        close_quad(upsampled, lr, ul, ur, ll)
    if 1:
        # fill in the interior of a refinement
        triangle_closing(upsampled, refinements[0][0].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[1][0].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[2][0].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[0][1].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[1][1].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[2][1].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[0][2].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[1][2].astype(int))  # try closing middle right
        triangle_closing(upsampled, refinements[2][2].astype(int))  # try closing middle right


def numb2(upsampled, mapping):
    polgon_corners = triangle_closing_skimage(upsampled, mapping)
    for ii, jj in polgon_corners:
        rr, cc = polygon(ii, jj)
        upsampled[rr,cc] = 1


def pure(upsampled, mapping):
    triangle_closing_old(upsampled, mapping)

upsampled, mapping, refinements = None, None, None

def test_close_refinement():
    global upsampled, mapping, refinements  # just for timeit


    matr = numpy.zeros((10, 10), dtype=int)
    draw_triangle(matr, numpy.array(((2, 1), (0, 0), (0, 6))))
    draw_triangle(matr, numpy.array(((2, 1), (6, 3), (0, 6))))
    draw_triangle(matr, numpy.array(((0, 0), (6, 3), (6, 0))))

    upsampled, refinements = test_grid(4, [[1,6,10],[1,9,(8, 5)], [16, 7, 60]])
    numb(upsampled, refinements)  # compile then time
    print(timeit.timeit("numb(upsampled, refinements)",number=10000,globals=globals()))

    print(upsampled)
    upsampled, mapping = test_refinement(2)
    pure(upsampled, mapping)
    print(upsampled)

    numb2(upsampled, mapping)  # compile then time
    print(timeit.timeit("numb2(upsampled, mapping)",number=10000,globals=globals()))
    print(timeit.timeit("pure(upsampled, mapping)",number=10000, globals=globals()))
    print(upsampled)
    polgon_corners = triangle_closing(upsampled, mapping)
    for ii, jj in polgon_corners:
        rr, cc = polygon(ii, jj)
        upsampled[rr,cc] = 1
    print(upsampled)

def interpolate_raster(input_file_full_path, dst_filename, sr_cell_size=None, method='linear', use_blocks=True, nodata=numpy.nan):
    """ Interpolation scheme
    Create the POINT version of the TIFF with only data at precise points of VR BAG
    Load in blocks with enough buffer around the outside (nominally 3x3 supergrids with 1 supergrid buffer)
        run scipy.interpolate.griddata on the block (use linear as cubic causes odd peaks and valleys)
        copy the interior (3x3 supergrids without the buffer area) into the output TIFF

    Create the MIN (or MAX) version of the TIFF
    Load blocks of data and copy any NaNs from the MIN (cell based coverage) into the INTERP grid to remove erroneous interpolations,
    this essentially limits coverage to VR cells that were filled
    """
    fobj, point_filename = tempfile.mkstemp(".point.tif")
    os.close(fobj)
    fobj, min_filename = tempfile.mkstemp(".min.tif")
    os.close(fobj)
    if not DEBUGGING:
        dx, dy, cell_sz = VRBag_to_TIF(input_file_full_path, point_filename, sr_cell_size=sr_cell_size, mode=POINT, use_blocks=use_blocks,
                                       nodata=nodata)
        VRBag_to_TIF(input_file_full_path, min_filename, sr_cell_size=sr_cell_size, mode=MIN, use_blocks=use_blocks, nodata=nodata)
    else:
        dx, dy, cell_sz = 128, 128, 1.07
        point_filename = r"C:\Data\BAG\GDAL_VR\H-10771\ExampleForEven\H-10771_python.1m_point.tif"
        min_filename = r"C:\Data\BAG\GDAL_VR\H-10771\ExampleForEven\H-10771_python.1m_min.tif"

    points_ds = gdal.Open(point_filename)
    points_band = points_ds.GetRasterBand(1)
    points_no_data = points_band.GetNoDataValue()
    coverage_ds = gdal.Open(min_filename)
    coverage_band = coverage_ds.GetRasterBand(1)
    coverage_no_data = coverage_band.GetNoDataValue()
    interp_ds = points_ds.GetDriver().Create(dst_filename, points_ds.RasterXSize, points_ds.RasterYSize, bands=1, eType=points_band.DataType,
                                             options=["BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    interp_ds.SetProjection(points_ds.GetProjection())
    interp_ds.SetGeoTransform(points_ds.GetGeoTransform())
    interp_band = interp_ds.GetRasterBand(1)
    interp_band.SetNoDataValue(nodata)

    if use_blocks:
        pixels_per_supergrid = int(max(dx / cell_sz, dy / cell_sz)) + 1
        row_block_size = col_block_size = 3 * pixels_per_supergrid
        row_buffer_size = col_buffer_size = 1 * pixels_per_supergrid
        row_size = interp_band.XSize
        col_size = interp_band.YSize
        for ic in tqdm(range(0, col_size, col_block_size), mininterval=.7):
            cols = col_block_size
            if ic + col_block_size > col_size:  # read a full set of data by offsetting the column index back a bit
                ic = col_size - cols
            col_buffer_lower = col_buffer_size if ic >= col_buffer_size else ic
            col_buffer_upper = col_buffer_size if col_size - (ic + col_block_size) >= col_buffer_size else col_size - (ic + col_block_size)
            read_cols = col_buffer_lower + cols + col_buffer_upper
            for ir in tqdm(range(0, row_size, row_block_size), mininterval=.7):
                rows = row_block_size
                if ir + row_block_size > row_size:
                    ir = row_size - rows
                row_buffer_lower = row_buffer_size if ir >= row_buffer_size else ir
                row_buffer_upper = row_buffer_size if row_size - (ir + row_block_size) >= row_buffer_size else row_size - (ir + row_block_size)
                read_rows = row_buffer_lower + rows + row_buffer_upper
                points_array = points_band.ReadAsArray(ir - row_buffer_lower, ic - col_buffer_lower, read_rows, read_cols)

                # Find the points that actually have data as N,2 array shape that can index the data arrays
                if numpy.isnan(points_no_data):
                    point_indices = numpy.nonzero(~numpy.isnan(points_array))
                else:
                    point_indices = numpy.nonzero(points_array != points_no_data)
                # if there were any data points then do interpolation -- could be all empty space too which raises Exception in griddata
                if len(point_indices[0]):
                    # get the associated data values
                    point_values = points_array[point_indices]
                    # interpolate all the other points in the array
                    # (actually it's interpolating everywhere which is a waste of time where there is already data)
                    xi, yi = numpy.mgrid[row_buffer_lower:row_buffer_lower + row_block_size,
                             col_buffer_lower:col_buffer_lower + col_block_size]
                    interp_data = scipy.interpolate.griddata(numpy.transpose(point_indices), point_values,
                                                             (xi, yi), method=method)
                    # mask based on the cell coverage found using the MIN mode
                    coverage_data = coverage_band.ReadAsArray(ir, ic, row_block_size, col_block_size)
                    interp_data[coverage_data == coverage_no_data] = nodata
                    # Write the data into the TIF on disk
                    interp_band.WriteArray(interp_data, ir, ic)
        if DEBUGGING:
            points_array = points_band.ReadAsArray()
            interp_array = interp_band.ReadAsArray()
            _plot(points_array)
            _plot(interp_array)
    else:
        points_array = points_band.ReadAsArray()
        coverage_data = coverage_band.ReadAsArray()

        # Find the points that actually have data
        if numpy.isnan(points_no_data):
            point_indices = numpy.nonzero(~numpy.isnan(points_array))
        else:
            point_indices = numpy.nonzero(points_array != points_no_data)
        # get the associated data values
        point_values = points_array[point_indices]
        # interpolate all the other points in the array (actually interpolating everywhere which is a waste of time where there is already data)
        xi, yi = numpy.mgrid[0:points_array.shape[0], 0:points_array.shape[1]]
        interp_data = scipy.interpolate.griddata(numpy.transpose(point_indices), point_values,
                                                 (xi, yi), method=method)
        _plot(interp_data)
        # mask based on the cell coverage found using the MIN mode
        interp_data[coverage_data == coverage_no_data] = nodata
        _plot(interp_data)
        # Write the data into the TIF on disk
        interp_band.WriteArray(interp_data)

    # release the temporary tif files and delete them
    point_band = None
    point_ds = None
    coverage_band = None
    coverage_ds = None
    if not DEBUGGING:
        os.remove(min_filename)
        os.remove(point_filename)

def vr_to_sr_points(vr_path, output_path, output_res):
    """ Create a single res raster from vr, only fills points that where vr existed (no interpolation etc).

    Returns
    -------
    dataset

    """
    try:
        resx, resy = output_res
    except TypeError:
        resx = resy = output_res
    vr = bag.VRBag(vr_path)
    raise Need EPSG from WKT
    epsg = vr.horizontal_crs_wkt
    area_db = world_raster_database.CustomArea(vr, vr.minx, vr.miny, vr.maxx, vr.maxy, resx, resy)
    area_db.insert_survey_vr(vr_path)

def vr_raster_mask(vr, points_ds, output_path):
    """ Given a VR and it's computed SR point raster, compute the coverage mask that describes where upsample-interpolation would be valid

    Parameters
    ----------
    vr
    sr_ds
    output_path
    output_res

    Returns
    -------
    dataset

    """
    points_band = points_ds.GetRasterBand(1)
    points_no_data = points_band.GetNoDataValue()
    interp_ds = points_ds.GetDriver().CreateCopy(output_path, points_ds)
    # iterate VR refinements filling in the mask cells
    # create a cache of refinements so we aren't constantly reading from disk
    cached_refinements = OrderedDict()
    cached_refinements.popitem(last=False)  # get rid of the oldest cached item, False makes it FIFO

def vr_to_points_and_mask(path_to_vr, points_path, mask_path, output_res):
    """ Given an input VR file path, create two output rasters.
    One contains points and one contains a mask of where upsample-interpolation would be appropriate.

    Parameters
    ----------
    path_to_vr
    points_path
    mask_path
    output_res

    Returns
    -------
    tuple(dataset, dataset)
        points gdal dataset and mask gdal dataset which are stored on disk at the supplied paths.

    """
    vr = bag.VRBag(path_to_vr)
    points_ds = vr_to_sr_points(path_to_vr, points_path, output_res)
    points_ds.GetDriver()
    mask_ds = vr_raster_mask(vr, points_ds, mask_path)
    return points_ds, mask_ds


def upsample_vr(path_to_vr, output_path, output_res):
    """ Use the world_raster_database to create an output 'points' tif of the correct resolution that covers the VR extents.
    Make a copy of the tif which will hold a mask of which cells should be filled using the VR closing functions from vr_utils.
    Then run scipy.interpolate.griddata using linear (cublic makes weird peaks) on the original output 'points' tif.
    Finally use the mask tif to crop the interpolated 'points' tif back to the appropriate areas.

    Parameters
    ----------
    path_to_vr
        vr bag to resample to single res
    output_path
        output location for a resampled + filled raster dataset
    output_res
        output resolution in the projection the vr is in

    Returns
    -------

    """
    # output_path = pathlib.Path(output_path)
    mask_path = pathlib.Path(output_path).with_suffix("mask.tif")
    points_ds, mask_ds = vr_to_points_and_mask(path_to_vr, output_path, mask_path, output_res)
    interpolate_raster(points_ds, mask_ds)
    if not _debug:
        del mask_ds
        os.remove(mask_path)
    return points_ds

if __name__ == "__main__":
    # print(ellipse_mask(5,5))
    # print(ellipse_mask(4,4))
    # print(ellipse_mask(7,4))
    # print(ellipse_mask(7,4))
    test_close_refinement()
    # upsample_and_interpolate_vr(r"G:\Data\NBS\Mississipi\UTM15\NCEI\H13330_MB_VR_LWRP.bag", r'G:\Data\NBS\Mississipi\test_upsample_1.5', 1.5)