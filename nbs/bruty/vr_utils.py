from collections import OrderedDict  # this is the default starting around python 3.7
import os
import shutil
import pathlib
import timeit
import functools
import tempfile

import numpy
import scipy.interpolate
from scipy.ndimage import binary_closing, binary_dilation
from skimage.draw import polygon

try:
    from numba import jit

    has_numba = True
except ModuleNotFoundError:
    has_numba = False
from osgeo import gdal
import rasterio

from HSTB.drivers import bag
from nbs.bruty import morton, world_raster_database
from nbs.bruty.utils import tqdm, iterate_gdal_image, onerr
from nbs.bruty.raster_data import affine, inv_affine, affine_center, LayersEnum

_debug = False

if _debug or not has_numba:  # turn off jit --
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
    """ Create an elliptical mask instead of the diamond that would happen from the normal scipy functions

    Parameters
    ----------
    r
        number of rows to use
    c
        number of columns to use

    Returns
    -------
    numpy.array
        boolean array in the shape determined by r,c parameters

    """
    center = numpy.array(((r - 1) / 2, (c - 1) / 2))
    accept_radius = 1.0
    # if the grid is even the center falls in between cells which would make the last row/column have all False
    # expanding the radius slightly (one cell in the other dimension) will create two True values in the edge row/col
    if not r % 2:
        accept_radius = max(accept_radius, (1 + (r - 1) ** 2) / ((r - 1) ** 2))
    if not c % 2:
        accept_radius = max(accept_radius, (1 + (c - 1) ** 2) / ((c - 1) ** 2))

    Y, X = numpy.ogrid[:r, :c]
    # normalize the distances since we are allowing rectangular grids
    # don't allow divide by zero
    dist2_from_center = ((Y - center[0]) / max(center[0], 0.5)) ** 2 + ((X - center[1]) / max(center[1], 0.5)) ** 2

    mask = dist2_from_center <= accept_radius  # max(radius)**2
    return mask


@jit(nopython=True)
def draw_triangle(matrix, pts, fill):
    """ Fill a triangle area of an array with a given value

    Parameters
    ----------
    matrix : numpy.array
        The array to fill
    pts : array, list
        three points [(r,c,...), (r,c,...), (r,c,...)]
        where first index is rows and the second index is columns, additional dimensions would be ignored
    fill : number
        value to put into the elements that the triangle contains

    Returns
    -------
    None
        The matrix is filled in place

    """
    # @todo change algorithm to standard or Bresenham see--
    #   http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    # pts = numpy.array(((ai, aj), (bi, bj), (ci, cj)))
    top, middle, bottom = numpy.argsort(pts[:, 0])
    if pts[top][0] != pts[middle][0]:
        slope_tm = (pts[top][1] - pts[middle][1]) / (pts[top][0] - pts[middle][0])
        slope_tb = (pts[top][1] - pts[bottom][1]) / (pts[top][0] - pts[bottom][0])
        for row in range(pts[top][0], pts[middle][0] + 1):
            c1, c2 = slope_tm * (row - pts[top][0]) + pts[top][1], slope_tb * (row - pts[top][0]) + pts[top][1]
            j1 = int(numpy.ceil(min(c1, c2)))
            j2 = int(numpy.trunc(max(c1, c2)))
            matrix[row, j1:j2 + 1] = fill
    else:
        j1 = min(pts[top][1], pts[middle][1])
        j2 = max(pts[top][1], pts[middle][1])
        matrix[pts[top][0], j1:j2 + 1] = fill
    if pts[middle][0] != pts[bottom][0]:
        slope_tb = (pts[top][1] - pts[bottom][1]) / (pts[top][0] - pts[bottom][0])
        slope_mb = (pts[middle][1] - pts[bottom][1]) / (pts[middle][0] - pts[bottom][0])
        for row in range(pts[middle][0], pts[bottom][0] + 1):
            c1, c2 = slope_mb * (row - pts[middle][0]) + pts[middle][1], slope_tb * (row - pts[top][0]) + pts[top][1]
            j1 = int(numpy.ceil(min(c1, c2)))
            j2 = int(numpy.trunc(max(c1, c2)))
            matrix[row, j1:j2 + 1] = fill
    else:
        j1 = min(pts[bottom][1], pts[middle][1])
        j2 = max(pts[bottom][1], pts[middle][1])
        matrix[pts[bottom][0], j1:j2 + 1] = fill


@jit(nopython=True)
def vr_close_quad(matrix, ul, ur, lr, ll, fill):
    """ Fill the supplied matrix in place with the given value in the area covered by the points supplied.

    Pass in points in connection order, but it is not important if they actually start at upper left.
    The points specify row, col and a boolean of if the point is valid.
    If all four points are valid then the quadrangle will be filled.
    If only three are valid then a triangle will be filled.
    If two or one are valid then no action is taken.
    @TODO should we draw a line or point in the 2, 1 cases?

    Parameters
    ----------
    matrix : numpy.array
    ul : array, list
        a quadrangle point, just make sure they are in connection order
    ur : array, list
        a quadrangle point, just make sure they are in connection order
    lr : array, list
        a quadrangle point, just make sure they are in connection order
    ll : array, list
        a quadrangle point, just make sure they are in connection order
    fill : number
        value to use when filling the matrix

    Returns
    -------
    None
        matrix is updated in place

    """
    # fixme -- allow for lines to fill?  if two points are there then make line between since triangle doesn't work
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
        draw_triangle(matrix, tri, fill)
    if draw_ll:
        tri = numpy.array([[li1, lj1], [li2, lj2], [ri2, rj2]], dtype=numpy.int32)  # [uld, lld, lrd]
        draw_triangle(matrix, tri, fill)
    if draw_lr:
        tri = numpy.array([[li2, lj2], [ri2, rj2], [ri1, rj1]], dtype=numpy.int32)  # [lld, lrd, urd]
        draw_triangle(matrix, tri, fill)
    if draw_ur:
        tri = numpy.array([[li1, lj1], [ri2, rj2], [ri1, rj1]], dtype=numpy.int32)  # [uld, lrd, urd]
        draw_triangle(matrix, tri, fill)


@jit(nopython=True)
def vr_close_refinements(matrix, left_or_top_rcb, left_or_top_xyz, right_or_bottom_rcb, right_or_bottom_xyz, max_dist, fill, horz=True):
    """ Close the area between two refinements.

    Since refinements don't start on the edge of a supercell there is potentially a gap in between refinements.
    This function walks along the edge of two neighboring refinements and fills the gap that could occur between them.


    Parameters
    ----------
    matrix : numpy.array
        data to operate on
    left_or_top_rcb : numpy.array
        a rectangular array of row/column/boolean
        which defines the mapping from refinement row/columns to the output matrix row/columns
    left_or_top_xyz : numpy.array
        a rectangular array of x,y,z
        which defines the mapping from refinement row/columns x,y,z of the projected space
    right_or_bottom_rcb : numpy.array
        a rectangular array of row/column/boolean
        which defines the mapping from refinement row/columns to the output matrix row/columns
    right_or_bottom_xyz : numpy.array
        a rectangular array of x,y,z
        which defines the mapping from refinement row/columns x,y,z of the projected space
    max_dist : number
        the distance in X (horizontal neighbors) or Y (vertical neighbors) to allow filling to take place.
        If the adjacent rows/cols of the two refinements are too far away then no fill will take place
    fill
        value to place in the filled gap
    horz
        specifies if the refinements are vertical (north/south) or horizontal (east/west) neighbors

    Returns
    -------
    None
        Fille the matrix in place.

    """
    # closes the gap between two refinements - have to specify if they are horizontal or vertical
    rb_index = 0
    lt_last = -1
    rb_last = 0
    lt_index = 0

    # get the x, y at the ends of the matrices, then difference them.
    # If the difference is less than zero then the matrix is going in the opposite convention than the bag
    # so we need to flip the compare logic for walking along the gap between the two
    pos_rows = numpy.sign(right_or_bottom_rcb[0, -1, -1] - right_or_bottom_rcb[0, 0, 0])
    pos_cols = numpy.sign(right_or_bottom_rcb[1, -1, -1] - right_or_bottom_rcb[1, 0, 0])
    if pos_rows == 0:  # this implies the whole bag fits in one cell, so either a low res output or a refinement with only one point
        pos_rows = 1
    if pos_cols == 0:
        pos_cols = 1
    if max_dist >= 0:
        # make sure the gap between supercells is less than max_dist for upsampling
        rx1, ry1, rz1 = right_or_bottom_xyz[:, 0, 0]
        rx2, ry2, rz2 = right_or_bottom_xyz[:, -1, -1]
        lx1, ly1, lz1 = left_or_top_xyz[:, 0, 0]
        lx2, ly2, lz2 = left_or_top_xyz[:, -1, -1]

        # find the closest distance from one cell to the other -- this is order independent by checking all possibilities, but slower.
        dx = numpy.abs(numpy.array([(rx1 - lx1), (rx1 - lx2), (rx2 - lx1), (rx2 - lx2)])).min()
        dy = numpy.abs(numpy.array([(ry1 - ly1), (ry1 - ly2), (ry2 - ly1), (ry2 - ly2)])).min()
        grid_dist = dx if horz else dy
        close_enough = grid_dist <= max_dist
    else:
        close_enough = True

    if close_enough:  # make sure the gap between supercells is less than max_dist for upsampling
        more_data = True
        if horz:
            max_lt = left_or_top_rcb.shape[1] - 2
            max_rb = right_or_bottom_rcb.shape[1] - 2
        else:
            max_lt = left_or_top_rcb.shape[2] - 2
            max_rb = right_or_bottom_rcb.shape[2] - 2
        if max_lt < 0 or max_rb < 0:  # @todo not sure how we want to handle a refinement that only has one cell
            more_data = False
        while more_data:
            if horz:  # @fixme -- one of these should be a 0 instead of _last.  Also confirm if row column interp is right
                ri1, rj1, rd1 = right_or_bottom_rcb[:, rb_index, rb_last]
                ri2, rj2, rd2 = right_or_bottom_rcb[:, rb_index + 1, rb_last]
                li1, lj1, ld1 = left_or_top_rcb[:, lt_index, lt_last]
                li2, lj2, ld2 = left_or_top_rcb[:, lt_index + 1, lt_last]
                # compare the rows as we are walking up the vertical gap between two horizonatally aligned refinements
                lt1 = li1 * pos_rows
                lt2 = li2 * pos_rows
                rb1 = ri1 * pos_rows
                rb2 = ri2 * pos_rows
            else:
                li1, lj1, ld1 = right_or_bottom_rcb[:, rb_last, rb_index]
                ri1, rj1, rd1 = right_or_bottom_rcb[:, rb_last, rb_index + 1]
                li2, lj2, ld2 = left_or_top_rcb[:, lt_last, lt_index]
                ri2, rj2, rd2 = left_or_top_rcb[:, lt_last, lt_index + 1]
                # compare the columns as we are walking along the horizontal gap between two vertically aligned refinements
                lt1 = lj2 * pos_cols
                lt2 = rj2 * pos_cols
                rb1 = lj1 * pos_cols
                rb2 = rj1 * pos_cols

            # @todo check the x/y distance so we don't close over a long diagonal?  Think of a 32 ro 64m res in  a 64m supergrid against a 1m grid
            # make sure the quad goes in order around the outside - going diagonally around the outside will leave empty pixels
            vr_close_quad(matrix, (ri1, rj1, rd1), (ri2, rj2, rd2), (li2, lj2, ld2), (li1, lj1, ld1), fill)
            # @fixme -- I think we can move the indices twice,
            #   moving once duplicates a triangle
            #   -- can only break on the first loop in case there is one triangle left
            # compare the row (for horizontal) or col (for vertical neighbors) and see which index needs to increment
            if lt1 >= rb2 and rb_index < max_rb:  # left points both above right
                rb_index += 1
            elif rb1 >= lt2 and lt_index < max_lt:  # right points both above left
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
                lt_index += 1
            elif rb_index < max_rb:
                rb_index += 1
            else:
                more_data = False
                break


@jit(nopython=True)
def vr_triangle_closing(matrix, refinement_rcb, fill):
    """ Fill in the internal parts of a refinement retaining any holes caused by missing data but filling in spaces
    between valid data in the matrix.

    Parameters
    ----------
    matrix : numpy.array
        data to operate on
    refinement_rcb : numpy.array
        a rectangular array of row/column/boolean
        which defines the mapping from refinement row/columns to the output matrix row/columns
    fill : number
        value to fill the output matrix with

    Returns
    -------
    None
        modifies matrix in place

    """
    pos_rows = numpy.sign(refinement_rcb[0, -1, -1] - refinement_rcb[0, 0, 0])
    pos_cols = numpy.sign(refinement_rcb[1, -1, -1] - refinement_rcb[1, 0, 0])
    if pos_rows == 0:  # this implies the whole bag fits in one cell, so either a low res output or a refinement with only one point
        pos_rows = -1  # this is the default read order used below
    if pos_cols == 0:
        pos_cols = 1
    # fixme -- this won't work on refinements of size 1
    for r in range(refinement_rcb.shape[1] - 1):
        for c in range(refinement_rcb.shape[2] - 1):
            # since tiffs are normally negative DY and bags are positive DY, we'll read assuming that and adjust if the tiff is +DY or -DX
            i1, j1, d1 = refinement_rcb[:, r, c]
            i, j, d = refinement_rcb[:, r + 1, c]
            i3, j3, d3 = refinement_rcb[:, r, c + 1]
            i2, j2, d2 = refinement_rcb[:, r + 1, c + 1]
            # does numba support swap commands?
            if pos_rows == 1:  # matrix is in same Y order as bag, so flip top/bottom rows in same col since we assumed opposite
                # switch i and i1
                ti, tj, td = i, j, d
                i, j, d = i1, j1, d1
                i1, j1, d1 = ti, tj, td
                ti, tj, td = i2, j2, d2
                i2, j2, d2 = i3, j3, d3
                i3, j3, d3 = ti, tj, td
            if pos_cols == -1:  # upsample opposite col ored, flip left right in same row
                # switch i and i2, i1 with i3
                ti, tj, td = i, j, d
                i, j, d = i2, j2, d2
                i2, j2, d2 = ti, tj, td
                ti, tj, td = i1, j1, d1
                i1, j1, d1 = i3, j3, d3
                i3, j3, d3 = ti, tj, td

            if d and d1 and d2 and d3:
                matrix[i:i3 + 1, j:j3 + 1] = fill
            elif d and d1 and d2:  # top left
                for r_i in range(i, i1 + 1):
                    if i1 != i:  # avoid divide by zero - range is only one scan line
                        end_j = numpy.trunc(j2 - (j2 - j) * (r_i - i) / (i1 - i))
                    else:
                        end_j = j2
                    matrix[r_i, j:int(end_j) + 1] = fill
            elif d and d1 and d3:  # bottom left
                for r_i in range(i, i1 + 1):
                    if i1 != i:  # avoid divide by zero - range is only one scan line
                        end_j = numpy.trunc(j + (j3 - j) * (r_i - i) / (i1 - i))
                    else:
                        end_j = j3
                    matrix[r_i, j:int(end_j) + 1] = fill
            elif d and d2 and d3:  # top right
                for r_i in range(i, i3 + 1):
                    if i3 != i:  # avoid divide by zero - range is only one scan line
                        start_j = numpy.ceil(j + (j2 - j) * (r_i - i) / (i3 - i))
                    else:
                        start_j = j
                    matrix[r_i, int(start_j):j2 + 1] = fill
            elif d1 and d2 and d3:  # bottom right
                for r_i in range(i2, i3 + 1):
                    if i3 != i2:  # avoid divide by zero - range is only one scan line
                        start_j = numpy.ceil(j2 - (j2 - j) * (r_i - i2) / (i3 - i2))
                    else:
                        start_j = j
                    upsampled[r_i, int(start_j):j2 + 1] = fill


def vr_close_neighbors(matrix, refinements, refinements_xyz, refinements_res, supercell_gap_multiplier=-1,
                       edges=True, corners=True, internal=True, vrfill=2, interpfill=3):
    """ This function is used to help convert a variable res bag to a single resolution matrix.
    Given a 3x3 of refinements
    (their row/col/bool mapping to the matrix space, their xyzs in their own projection and resolutions in that projection)
    this function will fill the center refinement, fill the gaps between it and the refinement above and right of it and the top two corners.

    Parameters
    ----------
    matrix : numpy.array
        data to write the filled cells into
    refinements : list of lists of numpy arrays
        3x3 list of arrays of row,col,is_valid for each refinement where row, col are in the matrix address space
    refinements_xyz
        3x3 list of arrays of x,y,z for each refinement where x,y,z are in the projection of the VR bag
    refinements_res
        3x3 lists of tuples of res_x and res_y
    supercell_gap_multiplier
        How wide of a gap to fill between refinements.  The multiplier is times the average of the two resolutions with gaps being filled.
        Zero would never gaps and 2.0 would be the sum of the two refinements, so 4m and 8m refinements would fill a gap of 12m (2.0*6)
    edges : boolean
         determines if edge gaps should be filled
    corners : boolean
         determines if corner gaps should be filled
    internal : boolean
         determines if internal space of the center refinement should be filled.
    vrfill : int
        value to insert into matrix at positions being closed within the center refinement
    interpfill : int
        value to insert into matrix in the gaps between the center and upper and right refinements

    Returns
    -------
    None
        matrix is modified in place
    """
    # numba wants integers for row column indices, so the refinements arrays are ints.
    # numba doesn't like structured arrays(?) so passing integers in refinements and floats in refinements_xyz

    # only doing the upper and right boundaries.
    # the other edges will be covered when processing the neighbors.
    # two top corners need to be done since an empty refinement (evaluate to None) would not need to fill edges but the corner still needs to be processed.

    # fill the boundaries between all refinements in the 3x3 set.
    center_refinement = refinements[1][1]
    center_refinement_xyz = refinements_xyz[1][1]
    # find a particular supercell by a corner position
    # if _debug:
    #     supersize = 130  # tolerance
    #     px, py = 370738.52, 4758378.13  # corner position
    #     for i, (x,y,z) in enumerate((center_refinement_xyz[:, -1, -1], center_refinement_xyz[:, -1, 0],
    #                   center_refinement_xyz[:, 0, -1], center_refinement_xyz[:, 0, 0])):
    #         if x > px-supersize and x < px+supersize and y > py-supersize and y < py+supersize:
    #             pass  # add a breakpoint here

    max_dist = -1
    if edges:
        if center_refinement is not None:
            if refinements[1][2] is not None:
                if supercell_gap_multiplier >= 0:
                    max_dist = refinements_res[1][1][0] + refinements_res[1][2][0]  # x res of center and x res of right
                    max_dist *= supercell_gap_multiplier / 2  # divide by two since we summed two resolutions
                vr_close_refinements(matrix, center_refinement, center_refinement_xyz, refinements[1][2], refinements_xyz[1][2], max_dist,
                                     interpfill)
            if refinements[2][1] is not None:
                if supercell_gap_multiplier >= 0:
                    max_dist = refinements_res[1][1][1] + refinements_res[2][1][1]  # y res of center and y res of top
                    max_dist *= supercell_gap_multiplier / 2  # divide by two since we summed two resolutions
                vr_close_refinements(matrix, center_refinement, center_refinement_xyz, refinements[2][1], refinements_xyz[2][1], max_dist,
                                     interpfill, horz=False)

    if corners:
        # fill in the corners by getting a corner point from each refinement
        # upper right of center refinement
        # if the refinement is None then a TypeError will occur and just set a 'false' in the depth attribute
        # @todo don't close corner if the data is too far from the supercell edge?
        no_data = numpy.array([0, 0, 0], dtype=numpy.int32)
        try:
            ll = center_refinement[:, -1, -1]
        except TypeError:
            ll = no_data
        try:
            ul = refinements[2][1][:, 0, -1]
        except TypeError:
            ul = no_data
        try:
            lr = refinements[1][2][:, -1, 0]
        except TypeError:
            lr = no_data
        try:
            ur = refinements[2][2][:, 0, 0]
        except TypeError:
            ur = no_data
        # rows=[c[0] for c in (ll, lr, ur, ul)]; cols = [c[1] for c in (ll, lr, ur, ul)]; print(refinements_xyz[1][1][:, -1, -1][0]), print(refinements_xyz[1][1][:, -1, -1][1]); print(rows, cols); print(upsampled[min(rows): max(rows)+1, min(cols):max(cols)+1])
        vr_close_quad(matrix, ul, ur, lr, ll,
                      interpfill)  # hole_at = 370867.325,4759142.356   numpy.abs(refinements_xyz[1][1][:, -1, -1][0]-370867.325)<32 and numpy.abs(refinements_xyz[1][1][:, -1, -1][1]-4759142.356)<32
        # upper left of center refinement
        try:
            lr = center_refinement[:, -1, 0]
        except TypeError:
            lr = no_data
        try:
            ul = refinements[2][0][:, 0, -1]
        except TypeError:
            ul = no_data
        try:
            ur = refinements[2][1][:, 0, 0]
        except TypeError:
            ur = no_data
        try:
            ll = refinements[1][0][:, -1, -1]
        except TypeError:
            ll = no_data
        vr_close_quad(matrix, ul, ur, lr, ll, interpfill)

    if internal:
        # fill in the interior of a refinement
        vr_triangle_closing(matrix, center_refinement, vrfill)


def vr_to_sr_points(vr, output_path, output_res, driver='GTiff'):
    """ Create a single res raster from vr, only fills points that where vr existed (no interpolation etc).

    Parameters
    ----------
    vr : str or VRBag
        either the path to a VR bag file or an open instance of HSTB.drivers.bag.VRBag
    output_path : str or pathlib.Path
        location to store the single res gdal dataset
    output_res : float
        size of the output cells in the projection of the VRBag
    driver : str
        gdal driver name, defaults to 'GTiff'

    Returns
    -------
    gdal.dataset, world_raster_database.CustomArea, int
        Returns the single resolution dataset, the CustomArea object that was used to make a 'points only' tif and
        the number of points that were inserted into the single res dataset

    """
    try:
        resx, resy = output_res
    except TypeError:
        resx = resy = output_res
    if not isinstance(vr, bag.VRBag):
        vr = bag.VRBag(vr_path, mode='r')
    epsg = rasterio.crs.CRS.from_string(vr.horizontal_crs_wkt).to_epsg()
    db_path = pathlib.Path(output_path).parent.joinpath('custom_db')
    supercell_half_x = vr.cell_size_x / 2.0
    supercell_half_y = vr.cell_size_y / 2.0
    # adjust for the VR returning center of supercells and not the edge.
    area_db = world_raster_database.CustomArea(epsg, vr.minx - supercell_half_x, vr.miny - supercell_half_y,
                                               vr.maxx + supercell_half_x, vr.maxy + supercell_half_y, resx, resy, db_path)
    area_db.insert_survey_vr(vr)
    cnt, sr_ds = area_db.export(output_path, driver=driver, layers=[LayersEnum.ELEVATION])
    if not _debug:
        del area_db
        if os.path.exists(db_path):
            shutil.rmtree(db_path, onerror=onerr)

    return sr_ds, area_db, cnt


def vr_raster_mask(vr, points_ds, output_path, supercell_gap_multiplier=-1, block_size=1024, use_nbs_codes=True):
    """ Given a VR and it's computed SR point raster, compute the coverage mask that describes where upsample-interpolation would be valid.

    Parameters
    ----------
    vr : str or HSTB.drivers.bag.VRBag
        full path to the VR file or an open HSTB.drivers.bag.VRBag instance
    points_ds : gdal.dataset
        an open dataset of a single resolution raster which will be used to make a matching size/resolution mask dataset
    output_path
        location to store mask dataset
    supercell_gap_multiplier
        How wide of a gap to fill between refinements.  The multiplier is times the average of the two resolutions with gaps being filled.
        Zero would never gaps and 2.0 would be the sum of the two refinements, so 4m and 8m refinements would fill a gap of 12m (2.0*6)
    block_size
        specify to control memory usage, less than zero will process entire dataset in memory.
        Greater than zero specifies the size of the numpy array to read/write from/to datasets.
    use_nbs_codes
        True creates a strict mask where 3 = interpolated gaps, 2 = upsampled intra-refinement locations and 1 = original point locations.

        False give the same raster coverage result but the values (1,2) may not be correct exact due to being overwritten when processing
        neighboring refinements.
    Returns
    -------
    dataset
        an open gdal.dataset with the mask data which was saved to output_path

    """
    max_cache = 100  # test using a 100 refinement cache
    points_band = points_ds.GetRasterBand(1)
    points_no_data = points_band.GetNoDataValue()
    mask_ds = points_ds.GetDriver().CreateCopy(str(output_path), points_ds)
    output_geotransform = mask_ds.GetGeoTransform()
    # @todo make read work on blocks not just whole file
    band = mask_ds.GetRasterBand(1)
    if block_size < 0:
        block_size = max(mask_ds.RasterXSize, mask_ds.RasterYSize)

    good_refinements = numpy.argwhere(vr.get_valid_refinements())  # bool matrix of which refinements have data
    mort = morton.interleave2d_64(good_refinements.T)
    sorted_refinement_indices = good_refinements[numpy.lexsort([mort])]

    # create a cache of refinements so we aren't constantly reading from disk
    cached_refinements = OrderedDict()
    # list of lists for neighbors.  None indicates an empty refinement
    neighbors_rcb = [[None, None, None], [None, None, None], [None, None, None]]
    neighbors_xyz = [[None, None, None], [None, None, None], [None, None, None]]
    neighbors_res = [[None, None, None], [None, None, None], [None, None, None]]
    if use_nbs_codes:
        # since we want strict reason codes and the edges/corners may overwrite the internal vr upsample flag
        # then we will do all the edges and corners in a first pass then the internal refinements in a second pass
        passes = ((True, True, False), (False, False, True))
    else:
        # to speed operations for the survey outlines, we'll do all at once since we don't care why something is filled
        passes = [(True, True, True)]
    cached_row, cached_col = -1, -1  # position of the currently held cache block from the mask raster
    for corners, edges, intra_refinment in passes:
        # iterate VR refinements filling in the mask cells
        for ri, rj in tqdm(sorted_refinement_indices, mininterval=.75):
            # get surrounding vr refinements as buffer
            index_ranges = numpy.zeros((3, 3, 2, 2), dtype=numpy.int32)
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    try:
                        refinement_data_rcb, refinement_data_xyz, refinement_res, refinement_index_range = cached_refinements[(ri + i, rj + j)]
                    except KeyError:
                        # read the refinement, convert to raster row/cols and cache the result
                        refinement = vr.read_refinement(ri + i, rj + j)
                        if refinement is not None:
                            pts = refinement.get_xy_pts_matrix()
                            r, c = inv_affine(pts[0], pts[1], *output_geotransform)
                            bool_depth = numpy.ones(r.shape, dtype=numpy.int32)
                            bool_depth[pts[4] == vr.fill_value] = 0
                            refinement_data_rcb = numpy.array((r, c, bool_depth), dtype=numpy.int32)  # row col bool
                            refinement_data_xyz = numpy.array((pts[0], pts[1], pts[4]), dtype=numpy.float64)
                            refinement_res = refinement.cell_size_x, refinement.cell_size_y
                            refinement_index_range = numpy.array([[refinement_data_rcb[0].min(), refinement_data_rcb[0].max()],
                                                                  [refinement_data_rcb[1].min(), refinement_data_rcb[1].max()]])

                        else:
                            refinement_data_xyz = None
                            refinement_data_rcb = None
                            refinement_res = None
                            refinement_index_range = numpy.array([[-1, -1], [-1, -1]])
                        cached_refinements[(ri + i, rj + j)] = (refinement_data_rcb, refinement_data_xyz, refinement_res, refinement_index_range)
                    # adjust for the indices being +/- 1 to 0 thru 2 for indexing, so i+1, j+1
                    neighbors_rcb[i + 1][j + 1] = refinement_data_rcb
                    neighbors_xyz[i + 1][j + 1] = refinement_data_xyz
                    neighbors_res[i + 1][j + 1] = refinement_res
                    index_ranges[i + 1, j + 1] = refinement_index_range  # track the indices so we can tell if the cache is valid
            # Get the output tif in that same area
            # if the block size (cache) is big enough then we won't need to read from the data file more than once
            rows = index_ranges[:, :, 0]
            min_row = int(rows[rows >= 0].min())
            max_row = int(rows[rows >= 0].max())
            cols = index_ranges[:, :, 1]
            min_col = int(cols[cols >= 0].min())
            max_col = int(cols[cols >= 0].max())
            min_row_ok = min_row >= cached_row and min_row < cached_row + block_size
            max_row_ok = max_row >= cached_row and max_row < cached_row + block_size
            min_col_ok = min_col >= cached_col and min_col < cached_col + block_size
            max_col_ok = max_col >= cached_col and max_col < cached_col + block_size
            cache_ok = min_col_ok and min_row_ok and max_col_ok and max_row_ok
            if cached_row < 0 or not cache_ok:
                if cached_row >= 0:  # write the current cache to the file before moving the cache location
                    band.WriteArray(output_matrix, cached_col, cached_row)
                # reset the cache row and col to a new location and revise the block size if it's too small
                if max_row - min_row >= block_size:
                    block_size = max_row - min_row + 1
                if max_col - min_col >= block_size:
                    block_size = max_col - min_col + 1
                # set the cache position to the min row/col and then offset by half of whatever extra cells would be read based on block size
                # also set it back from the edge, no reason to read past the end of the raser
                # and make sure the cache index is >= 0
                cached_row = max(0, min(mask_ds.RasterYSize - block_size, min_row - int(((block_size - (max_row - min_row + 1)) / 2))))
                cached_col = max(0, min(mask_ds.RasterXSize - block_size, min_col - int(((block_size - (max_col - min_col + 1)) / 2))))
                # read the new data
                output_matrix = band.ReadAsArray(cached_col, cached_row, block_size, block_size)
            # adjust the neighbors rows and cols to match the output_matrix based on the cache position
            # if the block size is big enough then the cache starts at zero and we don't need to adjust the row/col of the refinements
            if cached_row > 0 or cached_col > 0:
                revised_neighbors_rcb = [[None, None, None], [None, None, None], [None, None, None]]
                adjustment = numpy.array([[[cached_row, cached_col, 0]]], dtype=numpy.int32).T
                for i in (-1, 0, 1):
                    for j in (-1, 0, 1):
                        if neighbors_rcb[i + 1][j + 1] is not None:
                            revised_neighbors_rcb[i + 1][j + 1] = neighbors_rcb[i + 1][j + 1] - adjustment
            else:
                revised_neighbors_rcb = neighbors_rcb
            vr_close_neighbors(output_matrix, revised_neighbors_rcb, neighbors_xyz, neighbors_res,
                               supercell_gap_multiplier=supercell_gap_multiplier, corners=corners, edges=edges, internal=intra_refinment)
            while len(cached_refinements) > max_cache:
                cached_refinements.popitem(last=False)  # get rid of the oldest cached item, False makes it FIFO
        if cached_row >= 0:  # write any remaining data to the file
            band.WriteArray(output_matrix, cached_col, cached_row)
    if use_nbs_codes:
        # write the point posistions as 1 since the rest are 2=upsampled 3=interpolated (gaps between refinements)
        for ic, ir, nodata, pts_bands in iterate_gdal_image(points_ds, min_block_size=block_size, max_block_size=block_size):
            points_array = pts_bands[0]
            # get the matching portion of the mask since mask and points must be same shapes
            output_matrix = band.ReadAsArray(ic, ir, points_array.shape[1], points_array.shape[0])
            # figure out where the points are
            if numpy.isnan(nodata):
                point_indices = numpy.nonzero(~numpy.isnan(points_array))
            else:
                point_indices = numpy.nonzero(points_array != nodata)
            # set point positions to mask of 1
            output_matrix[point_indices] = 1
            band.WriteArray(output_matrix, ic, ir)  # push updated data back into file
    band = None
    return mask_ds


def vr_to_points_and_mask(path_to_vr, points_path, mask_path, output_res, supercell_gap_multiplier=-1, block_size=1024, nbs_mask=True):
    """ Given an input VR file path, create two output rasters.
    One contains original point locations and one contains a mask of where upsample-interpolation would be appropriate.

    Parameters
    ----------
    path_to_vr : str
        path to the VR file
    points_path
        output location of the points raster
    mask_path
        output location of the mask raster
    output_res
        resolution in the VR projection of the resulting single resolution datasets
    supercell_gap_multiplier
        How wide of a gap to fill between refinements.  The multiplier is times the average of the two resolutions with gaps being filled.
        Zero would never gaps and 2.0 would be the sum of the two refinements, so 4m and 8m refinements would fill a gap of 12m (2.0*6)
    block_size
        specify to control memory usage, less than zero will process entire dataset in memory.
        Greater than zero specifies the size of the numpy array to read/write from/to datasets.
    nbs_mask
        True creates a strict mask where 3 = interpolated gaps, 2 = upsampled intra-refinement locations and 1 = original point locations.

        False give the same raster coverage result but the values (1,2) may not be correct exact due to being overwritten when processing
        neighboring refinements.

    Returns
    -------
    tuple(vr, dataset, dataset)
        HSTB.drivers.bag.VRBag instance, points gdal dataset and mask gdal dataset which are stored on disk at the supplied paths.

    """
    vr = bag.VRBag(path_to_vr, mode='r')
    points_ds, area_db, cnt = vr_to_sr_points(path_to_vr, points_path, output_res)
    mask_ds = vr_raster_mask(vr, points_ds, mask_path, supercell_gap_multiplier=supercell_gap_multiplier,
                             block_size=block_size, use_nbs_codes=nbs_mask)
    return vr, points_ds, mask_ds


def interpolate_raster(vr, points_ds, mask_ds, output_path, use_blocks=True, nodata=numpy.nan, method='linear'):
    """ Create a interpolated single resolution raster representing the area covered by the supplied VR bag file.
    Data inside a refinement will be 'upsampled' between any data cells that have valid data.
    Data between refinements will be 'interpolated' if the gap is less than the sum of the reolutions of the two refinements.

    Parameters
    ----------
    vr
        An open HSTB.drivers.bag.VRBag instance
    points_ds
        open gdal dataset of the points raster made from the vr
    mask_ds
        open gdal dataset of the mask raster made from the vr and points
    output_path
        path to store the resulting interpolated single resolution dataset
    use_blocks
        Reduce memory usage by reading/writing blocks from the gdal.dataset rather than loading the entire raster into memory
    nodata
        Value to write into the output file as the nodata value
    method
        scipy.interpolate method to use when processing the datasets.
    Returns
    -------
    dataset
        An open gdal.dataset object of the file saved at output_path with the interpolated data
    """
    # Interpolation scheme
    # Given the POINT version of the TIFF with only data at precise points of VR BAG (old version of function would create POINT tiff)
    # Load in blocks with enough buffer around the outside (nominally 3x3 supergrids with 1 supergrid buffer)
    #     run scipy.interpolate.griddata on the block (use linear as cubic causes odd peaks and valleys)
    #     copy the interior (3x3 supergrids without the buffer area) into the output TIFF
    #
    # Given the mask version of the TIFF  (old function used to Create a 'min')
    # Load blocks of data and copy any NaNs from the MIN (cell based coverage) into the INTERP grid to remove erroneous interpolations,
    # this essentially limits coverage to VR cells that were filled

    dx = vr.cell_size_x
    dy = vr.cell_size_y
    cell_szx = numpy.abs(points_ds.GetGeoTransform()[1])
    cell_szy = numpy.abs(points_ds.GetGeoTransform()[5])

    points_band = points_ds.GetRasterBand(1)
    points_no_data = points_band.GetNoDataValue()
    mask_band = mask_ds.GetRasterBand(1)
    mask_no_data = mask_band.GetNoDataValue()
    interp_ds = points_ds.GetDriver().CreateCopy(str(output_path), points_ds)
    # interp_ds = points_ds.GetDriver().Create(dst_filename, points_ds.RasterXSize, points_ds.RasterYSize, bands=1, eType=points_band.DataType,
    #                                          options=["BLOCKXSIZE=256", "BLOCKYSIZE=256", "TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"])
    # interp_ds.SetProjection(points_ds.GetProjection())
    # interp_ds.SetGeoTransform(points_ds.GetGeoTransform())
    interp_band = interp_ds.GetRasterBand(1)
    interp_band.SetNoDataValue(nodata)

    if use_blocks:
        # @todo move to using iterate gdal generator from bruty.utils -- would need to add buffer option to iterate_gdal
        pixels_per_supergrid = int(max(dx / cell_szx, dy / cell_szy)) + 1
        row_block_size = col_block_size = 3 * pixels_per_supergrid
        if row_block_size < 512:
            row_block_size = col_block_size = 512
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
                points_array = points_band.ReadAsArray(ic - col_buffer_lower, ir - row_buffer_lower, read_cols, read_rows)

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
                    row_i, col_i = numpy.mgrid[row_buffer_lower:row_buffer_lower + row_block_size,
                                   col_buffer_lower:col_buffer_lower + col_block_size]
                    try:
                        interp_data = scipy.interpolate.griddata(numpy.transpose(point_indices), point_values,
                                                                 (row_i, col_i), method=method)
                    except scipy.spatial.qhull.QhullError as e:
                        if len(point_indices[0]) < 3:
                            # find the data that would fall in the griddata result and just insert those couple points
                            # could do with a for loop and a couple if statements easier
                            interp_data = interp_band.ReadAsArray(ic, ir, col_block_size, row_block_size)
                            ind = numpy.transpose(point_indices)
                            row_non_neg = ind[:, 0] >= 0
                            row_not_big = ind[:, 0] < row_block_size
                            col_non_neg = ind[:, 1] >= 0
                            col_not_big = ind[:, 1] < col_block_size
                            use_indices = numpy.logical_and.reduce((row_non_neg, row_not_big, col_non_neg, col_not_big))
                            if numpy.count_nonzero(use_indices):
                                interp_data[point_indices[0][use_indices], point_indices[1][use_indices]] = point_values[use_indices]
                        else:
                            raise e

                    mask_data = mask_band.ReadAsArray(ic, ir, col_block_size, row_block_size)
                    if numpy.isnan(mask_no_data):
                        mask_array = numpy.isnan(mask_data)
                    else:
                        mask_array = mask_data == mask_no_data
                    interp_data[mask_array] = nodata
                    # Write the data into the TIF on disk
                    interp_band.WriteArray(interp_data, ic, ir)
    else:
        points_array = points_band.ReadAsArray()
        mask_data = mask_band.ReadAsArray()

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
        # mask based on the cell mask found using the MIN mode
        interp_data[mask_data == mask_no_data] = nodata
        _plot(interp_data)
        # Write the data into the TIF on disk
        interp_band.WriteArray(interp_data)

    # release the temporary tif files and delete them
    point_band = None
    point_ds = None
    mask_band = None
    mask_ds = None
    return interp_ds


def upsample_vr(path_to_vr, output_path, output_res, block_size=1024, keep_mask=False):
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
    block_size
        numpy array size to use when reading/writing to gdal datasets, less than zero will load entire dataset to memory
    keep_mask
        True will retain the mask raster file on disk when finished
        False will remove the mask raster from disk
    Returns
    -------
    dataset
        An open gdal.dataset object of the file saved at output_path with the interpolated data

    """
    # output_path = pathlib.Path(output_path)
    mask_path = pathlib.Path(output_path).with_suffix(".mask.tif")
    points_path = pathlib.Path(output_path).with_suffix(".points.tif")
    vr, points_ds, mask_ds = vr_to_points_and_mask(path_to_vr, points_path, mask_path, output_res, supercell_gap_multiplier=1.99,
                                                   block_size=block_size)
    # can we write directly into the points_ds instead of copying?
    interp_ds = interpolate_raster(vr, points_ds, mask_ds, output_path)
    if not _debug:
        if not keep_mask:
            del mask_ds
            os.remove(mask_path)
        del points_ds
        os.remove(points_path)
    return interp_ds
