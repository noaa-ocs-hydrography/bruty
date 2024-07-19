import os
import subprocess
import pathlib
import sys
import time
import re
from datetime import datetime, timedelta

import psutil
import platform
import numpy
import rasterio.crs
from osgeo import gdal, osr, ogr
import pyproj.exceptions
from pyproj import Transformer, CRS
from nbs.bruty.exceptions import BrutyFormatError, BrutyMissingScoreError, BrutyUnkownCRS, BrutyError

if os.name == 'posix':
    import fcntl
    import termios
    import struct

    char_cache = []


    def kbhit():  # Windows returns true if any characters are left in the buffer, so we are emulating that behavior
        try:
            bytes_str = fcntl.ioctl(sys.stdin.fileno(), termios.FIONREAD, struct.pack('I', 0))
        except OSError:  # linux services refuse the stdin connection
            pass  # leave the cache empty on Linux when running as a service
        else:
            bytes_num = struct.unpack('I', bytes_str)[0]
            if bytes_num > 0:
                cache_chars()
        return len(char_cache) > 0


    # When a stdin.read is called the ioctl will return zero next time, so we need to cache the available characters now
    # We do this non-blocking since the read will wait forever otherwise
    def cache_chars():
        fd = sys.stdin.fileno()
        # fetch stdin's old flags
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        # set the none-blocking flag
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        try:
            ch = sys.stdin.read(1)  # read one character at a time, read() waits for eof
            while ch:
                char_cache.append(ch)
                ch = sys.stdin.read(1)
        except:
            ch = ''
        finally:
            # resetting stdin to default flags
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
        return len(char_cache)


    # Windows gives one character at a time, so we are emulating that behavior either returning the cached characters or reading more
    def getch():
        try:
            return char_cache.pop(0)
        except IndexError:
            next_key = sys.stdin.read(1)  # this will block until a key is pressed
            cache_chars()  # also caches any other keys that are available
            return next_key

elif os.name == 'nt':
    from msvcrt import kbhit, getch
else:
    raise NotImplementedError("Unexpected operating system, not nt or posix")

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterate_stuff, *args, **kywrds):
        return iterate_stuff  # if this doesn't work, try iter(iterate_stuff)


def get_epsg_or_wkt(srs):
    if isinstance(srs, str):
        srs = osr.SpatialReference(srs)
    # epsg = rasterio.crs.CRS.from_string(vr.srs.ExportToWkt()).to_epsg()
    if srs.IsProjected():
        epsg = srs.GetAuthorityCode("PROJCS")
    elif srs.IsGeographic():
        epsg = srs.GetAuthorityCode("GEOGCS")
    else:
        raise TypeError("projection not understood - IsProjected and IsGeographic both False?")
    if epsg is None:
        epsg = srs.ExportToWkt()
        # utm_patterns = ['UTM Zone (\d+)(N|, Northern Hemisphere)', ]
        # for utm_pattern in utm_patterns:
        #     m = re.search(utm_pattern, srs.GetName(), re.IGNORECASE)
        #     if m:
        #         zone = int(m.groups()[0])
        #         if srs.GetAuthorityCode("GEOGCS") == 4326:  # WGS84
        #             epsg = 32600 + zone
        #         elif srs.GetAuthorityCode("GEOGCS") == 4269:  # NAD83
        #             epsg = 26900 + zone
        #         break
    return epsg


QUIT = "quit"
HELP = "help"


def key_to_action(key):
    if key in (b"q", 'q', b'Q', 'Q'):
        user_action = QUIT
    elif key in (b"?", '?'):
        user_action = HELP
    else:
        user_action = None
    return user_action


def user_action():
    ignore_chars = ('\n', '\r')  # characters to ignore
    action = None
    while kbhit():
        print("checking keyboard input for 'qq' or other command")
        ch = getch()
        if ch not in ignore_chars:
            action = key_to_action(ch)
            if action == QUIT:
                print("hit 'q' twice to quit")
                second_key = getch()
                if second_key in ignore_chars:  # user could hit q\n\n  (two enters) which should not quit, so ignore the first but not a second
                    second_key = getch()
                action = key_to_action(second_key)
    return action


def make_mllw_height_wkt(horz_epsg):
    wkt = make_wkt(horz_epsg, 5866)
    # 5866 with GDAL will not accept the Up axis, so have to strip the 5866 epsg authority
    down_string = 'AXIS["Depth",DOWN],AUTHORITY["EPSG","5866"]'
    if down_string not in wkt:
        raise Exception("Down not found in VertCS, did gdal change?")
    else:
        # wkt = wkt.replace('AXIS["Depth",DOWN]', 'AXIS["gravity-related height",UP]')
        # wkt = wkt.replace('AXIS["Depth",DOWN]', 'AXIS["Height",UP]')
        wkt = wkt.replace(down_string, 'AXIS["gravity-related height",UP]').replace("MLLW depth", "MLLW")

        pass
    return wkt


def make_wkt(horz_epsg, vert_epsg):
    """ Calls gdalsrsinfo with the epsgs supplied.  MLLW (5866) is the default vertical.
    down_to_up flag will change AXIS["Depth",DOWN] to AXIS["gravity-related height",UP]
    see:  https://docs.opengeospatial.org/is/18-010r7/18-010r7.html#47
    """
    # cmd = f"gdalsrsinfo EPSG:{horz_epsg}+{vert_epsg} -o WKT1 --single-line"
    # srs_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # srs_process.wait()
    # wkt_old = srs_process.stdout.read().decode().strip()
    # stderr = srs_process.stderr.read().decode().strip()

    srs = osr.SpatialReference()
    srs.SetFromUserInput(f"EPSG:{horz_epsg}+{vert_epsg}")
    # srs.ExportToWkt(["FORMAT=WKT2"])
    wkt = srs.ExportToWkt(["FORMAT=WKT1"])
    if "COMPD_CS" not in wkt:
        raise Exception("compound CRS not found")

    return wkt


def onerr(func, path, info):
    r"""This is a helper function for shutil.rmtree to take care of something happening on (at least) my local machine.
    It seems that the Windows system deletes don't remove the files immediately and the directory isn't empty when rmdir is called.
    Running the function again will work, so adding a delay to see if the system catches up.
    If a file handle is actually open then the call below will fail and the exception will be raised, just a bit slower.
        File "C:\PydroTrunk\Miniconda36\envs\Pydro38_Test\lib\shutil.py", line 617, in _rmtree_unsafe
            os.rmdir(path)
        OSError: [WinError 145] The directory is not empty: 'C:/GIT_Repos/nbs/xipe_dev/xipe2.bruty/test_data_output/tile4_utm_db_grid/tmp1hbjeurd\\3533'
    """
    print("rmdir error, pausing")
    time.sleep(2)
    try:
        func(path)
    except Exception:
        raise


def affine(r, c, x0, dxx, dyx, y0, dxy, dyy):
    """
    Returns the affine transform -- normally row, column to x,y position.
    If this is the geotransform from a gdal geotiff (for example) the coordinates are the displayed pixel corners - not the center.
    If you want the center of the pixel then use affine_center
    """
    x = x0 + c * dxx + r * dyx
    y = y0 + c * dxy + r * dyy
    return x, y


def inv_affine(x, y, x0, dxx, dyx, y0, dxy, dyy):
    if dyx == 0 and dxy == 0:
        c = numpy.array(numpy.floor((numpy.array(x) - x0) / dxx), dtype=numpy.int32)
        r = numpy.array(numpy.floor((numpy.array(y) - y0) / dyy), dtype=numpy.int32)
    else:
        # @todo support skew projection
        raise ValueError("non-North up affine transforms are not supported yet")
    return r, c


def affine_center(r, c, x0, dxx, dyx, y0, dxy, dyy):
    return affine(r + 0.5, c + 0.5, x0, dxx, dyx, y0, dxy, dyy)


def get_crs_transformer(epsg_or_wkt1, epsg_or_wkt2):
    # convert gdal/osr SpatialReference to WKT so pyproj will use it.
    if isinstance(epsg_or_wkt1, osr.SpatialReference):
        epsg_or_wkt1 = epsg_or_wkt1.ExportToWkt()
    if isinstance(epsg_or_wkt2, osr.SpatialReference):
        epsg_or_wkt2 = epsg_or_wkt2.ExportToWkt()
    # pyproj will fail on CRS("32619") so make it an integer so we call CRS(32619)
    if isinstance(epsg_or_wkt1, str):
        if epsg_or_wkt1.isdigit():
            epsg_or_wkt1 = int(epsg_or_wkt1)
    if isinstance(epsg_or_wkt2, str):
        if epsg_or_wkt2.isdigit():
            epsg_or_wkt2 = int(epsg_or_wkt2)

    if epsg_or_wkt1 is None and epsg_or_wkt2 is None:
        crs_transformer = None
    else:
        try:
            input_crs = CRS(epsg_or_wkt1)
            output_crs = CRS(epsg_or_wkt2)
        except pyproj.exceptions.CRSError as e:
            raise BrutyUnkownCRS(str(e))

        if input_crs != output_crs:
            crs_transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
        else:
            crs_transformer = None
    return crs_transformer


def merge_arrays(x, y, keys, data, *args, **kywrds):
    # create a combined two dimensional array that can then be indexed and reduced as needed,
    #   if the incoming data was multidimensional this will flatten it too.
    #   So an incoming dataset is say 6 fields of 4 rows and 5 columns will be turned into 10 x 20 -- (6+4 fields x 4 rows * 5 cols)
    #   wouldn't be able to remove nans and potentially do some other operations otherwise(?).
    pts = numpy.array((x, y, *keys, *data)).reshape(2 + len(keys) + len(data), -1)
    merge_array(pts, *args, **kywrds)


def merge_array(pts, output_data, output_sort_key_values,
                crs_transform=None, affine_transform=None, start_col=0, start_row=0, block_cols=None, block_rows=None,
                reverse_sort=None, key_bounds=None
                ):
    """  Merge a new dataset (array) into an existing dataset (array).
    The source data is compared to the existing data to determine if it should supercede the existing data.
    For example, this could be hydro-health score comparison for higher quality data or depths for shoal biasing.

    The incoming dataset uses a geotransform to go from source to destination coordinates
    then an affine transform to go from destination x,y to array row, column

    Basically pass in x,y,(sort_key1, sort_key2, ...), data_to_for_output_array, output_array, output_sort_key_result
    if affine_transform is None then pass in row, column instead of x,y

    note: modifies the output array in place

    Parameters
    ----------
    pts
        x,y, sort_key1, sort_key2, ..., data_for_output_array
        The number of sort keys is defined by the length of the supplied output_sort_key_values array.
    output_data
        array of length that matches the 'data_for_output_array' without desired rows/cols for the data to be inserted to.
        Must match the number of arrays passed in after the sort_keys
    output_sort_key_values
        array of length must match the number of sort keys passed in.
        Must match the row/column size of the output_data
    crs_transform
        object with a .transform(x,y) method returning x2, y2
    affine_transform
        If supplied, converts from x,y to row,col
    start_col
        column offset value for the output_array.
    start_row
        row offset value for the output_array.
    block_cols
        maximum column to fill with data
    block_rows
        maximum row to fill with data
    reverse_sort
        If supplied, an iterable of booleans specifying if the corresponding sort_key is reversed.
        ex: if sort keys were (z,x,y) and the smallest z was desired then (True, False, False) would be the reverse_sort value
    key_bounds
        Ranges of acceptable values for each sort passed in as (min, max).  Values outside this range will be discarded.
        None can be supplied if no min or max is needed.  Either min or max could also be None.

        ex: given a (z,lat,lon) sort, if elevations were desired between 0m and 40m and only from 30deg to 40deg latitude you
        would specify ((0, 40), (30, 40), None).

        ex: given a (z,lat,lon) sort, if elevations were desired below 0m and above 40deg latitude you
        would specify ((None, 0), (40, None), None).

        !!Note - exclusive bands do not work!!  passing in (40, 0) to try and get above 40 or below 0 will return nothing.
        @todo add the ability to have a callback or specify the if the range is and/or so excludes would work.

    Returns
    -------
    None
    """

    pts = pts[:, ~numpy.isnan(pts[2])]  # remove empty cells (no score = empty)
    if len(pts[0]) > 0:
        if block_rows is None:
            row_index = 1 if len(output_data.shape) > 2 else 0
            block_rows = output_data.shape[row_index] - start_row
        if block_cols is None:
            col_index = 2 if len(output_data.shape) > 2 else 1
            block_cols = output_data.shape[col_index] - start_col

        # 6) Sort on score in case multiple points go into a position that the right value is retained
        #   sort based on score then on depth so the shoalest top score is kept
        if reverse_sort is None:
            sort_multiplier = [1] * len(output_sort_key_values)
        else:
            sort_multiplier = [-1 if flag else 1 for flag in reverse_sort]

        # sort the points, the following puts all the keys in backwards (how lexsort wants) and flips signs as needed
        sorted_ind = numpy.lexsort([sort_multiplier[num_key] * pts[num_key + 2] for num_key in range(len(output_sort_key_values) - 1, -1, -1)])
        # sorted_ind = numpy.lexsort((sort_z_multiplier * pts[3], sort_score_multiplier * pts[2]))
        sorted_pts = pts[:, sorted_ind]

        # 7) Use affine geotransform convert x,y into the i,j for the exported area
        if crs_transform:
            transformed_x, transformed_y = crs_transform.transform(sorted_pts[0], sorted_pts[1])
        else:
            transformed_x, transformed_y = sorted_pts[0], sorted_pts[1]

        if affine_transform is not None:
            export_rows, export_cols = inv_affine(transformed_x, transformed_y, *affine_transform)
        else:
            export_rows, export_cols = transformed_x.astype(numpy.int32), transformed_y.astype(numpy.int32)
        export_rows -= start_row  # adjust to the sub area in memory
        export_cols -= start_col

        # clip to the edges of the export area since our db tiles can cover the earth [0:block_rows-1, 0:block_cols]
        row_out_of_bounds = numpy.logical_or(export_rows < 0, export_rows >= block_rows)
        col_out_of_bounds = numpy.logical_or(export_cols < 0, export_cols >= block_cols)
        out_of_bounds = numpy.logical_or(row_out_of_bounds, col_out_of_bounds)
        if out_of_bounds.any():
            sorted_pts = sorted_pts[:, ~out_of_bounds]
            export_rows = export_rows[~out_of_bounds]
            export_cols = export_cols[~out_of_bounds]

        # 8) Write the data into the export (single) tif.
        # replace x,y with row, col for the points
        # @todo write unit test to confirm that the sort is working in case numpy changes behavior.
        #   currently assumes the last value is stored in the array if more than one have the same ri, rj indices.
        replace_cells = numpy.isnan(output_sort_key_values[0, export_rows, export_cols])
        previous_all_equal = numpy.full(replace_cells.shape,
                                        True)  # tracks if the sort keys are all equal in which case we have to check the next key
        key_in_bounds = numpy.full(replace_cells.shape, True)
        for key_num, key in enumerate(output_sort_key_values):
            if reverse_sort is None or not reverse_sort[key_num]:
                comp_func = numpy.greater
            else:
                comp_func = numpy.less
            replace_cells = numpy.logical_or(replace_cells,
                                             numpy.logical_and(previous_all_equal,
                                                               comp_func(sorted_pts[2 + key_num], key[export_rows, export_cols])))
            previous_all_equal = numpy.logical_and(previous_all_equal,
                                                   numpy.equal(sorted_pts[2 + key_num], key[export_rows, export_cols]))
            # check that the key values are within the desired ranges
            if key_bounds is not None:
                if key_bounds[key_num] is not None:
                    key_min, key_max = key_bounds[key_num]
                    if key_min is not None:
                        key_in_bounds = numpy.logical_and(key_in_bounds, sorted_pts[2 + key_num] >= key_min)
                    if key_max is not None:
                        key_in_bounds = numpy.logical_and(key_in_bounds, sorted_pts[2 + key_num] <= key_max)
        replace_cells = numpy.logical_and(replace_cells, key_in_bounds)

        replacements = sorted_pts[2 + len(output_sort_key_values):, replace_cells]
        ri = export_rows[replace_cells]
        rj = export_cols[replace_cells]
        output_data[:, ri, rj] = replacements
        for key_num, key in enumerate(output_sort_key_values):
            key[ri, rj] = sorted_pts[2 + key_num, replace_cells]


def remake_tif(fname):
    """Tiffs that are updated don't compress properly, so move them to a temporary name, copy the data and delete the temporary name"""
    os.rename(fname, fname + ".old.tif")
    ds = gdal.Open(fname + ".old.tif")
    data = ds.ReadAsArray()
    driver = gdal.GetDriverByName('GTiff')
    if 0:
        new_ds = driver.Create(fname, xsize=ds.RasterXSize, ysize=ds.RasterYSize, bands=ds.RasterCount, eType=gdal.GDT_Float32,
                               options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])
        new_ds.SetProjection(ds.GetProjection())
        new_ds.SetGeoTransform(ds.GetGeoTransform())
        for n in range(ds.RasterCount):
            band = new_ds.GetRasterBand(n + 1)
            band.WriteArray(data[n])
    else:
        new_ds = driver.CreateCopy(fname, ds, options=['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"])

    new_ds = None
    ds = None
    os.remove(fname + ".old.tif")


# for (dirpath, dirnames, filenames) in os.walk(r'C:\data\nbs\test_data_output\test_pbc_19_db'):
#     for fname in filenames:
#         if fname.endswith(".tif"):
#             remake_tif(os.path.join(dirpath, fname))


def soundings_from_image(fname, res):
    ds = gdal.Open(str(fname))
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    affine_params = ds.GetGeoTransform()
    epsg = rasterio.crs.CRS.from_string(ds.GetProjection()).to_epsg()
    xform = ds.GetGeoTransform()  # x0, dxx, dyx, y0, dxy, dyy
    d_val = ds.GetRasterBand(1)
    col_size = d_val.XSize
    row_size = d_val.YSize
    del d_val
    x1, y1 = affine(0, 0, *xform)
    x2, y2 = affine(row_size, col_size, *xform)
    try:  # allow res to be tuple or single value
        res_x, res_y = res
    except TypeError:
        res_x = res_y = res

    # move the minimum to an origin based on the resolution so future exports would match
    # ex: res = 50 would make origin be 0 or 50 or 100 but not contain 77.3 etc.
    if x1 < x2:
        x1 -= x1 % res_x
    else:
        x2 -= x2 % res_x

    if y1 < y2:
        y1 -= y1 % res_y
    else:
        y2 -= y2 % res_y
    min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
    # create an x,y,z array of nans in the necessary shape
    output_array = numpy.full([3, shape_y, shape_x], numpy.nan, dtype=numpy.float64)
    output_sort_values = numpy.full([3, shape_y, shape_x], numpy.nan, dtype=numpy.float64)
    output_xform = [min_x, res_x, 0, max_y, 0, -res_y]
    layers = [ds.GetRasterBand(b + 1).GetDescription().lower() for b in range(ds.RasterCount)]
    band_num = None
    for name in ('elevation', 'depth'):
        if name in layers:
            band_num = layers.index(name) + 1
            break
    if band_num is None:
        band_num = 1
    for ic, ir, nodata, (depths,) in iterate_gdal_image(ds, (band_num,)):
        depths[depths == nodata] = numpy.nan
        r, c = numpy.indices(depths.shape)
        x, y = affine_center(r + ir, c + ic, *xform)
        # first key is depth second is latitude (just to have a tiebreaker so results stay consistent)
        # reusing the depth (output_array[2]) and y output_array[1] in the sortkey arrays
        merge_arrays(x, y, (depths, x, y), (x, y, depths), output_array, output_sort_values, affine_transform=output_xform)
    return srs, output_array


def iterate_gdal_image(dataset, band_nums=(1,), min_block_size=512, max_block_size=1024,
                       start_col=0, end_col=None, start_row=0, end_row=None, leave_progress_bar=True):
    """ Iterate a gdal dataset using blocks to reduce memory usage.
    The last blocks at the edge of the dataset will have a smaller size than the others.
    Reads down all the rows first then moves to the next group of columns.
    The function will use the dataset's GetBlockSize() if it falls between the min and max block size arguments

    Ex: a 10x10 image read in blocks of 4x4 would return  two 4x4 arrays followed by a 2x4 array.  Then 4x4, 4x4, 2x4.  Then 4x2, 4x2, 2x2.

    Parameters
    ----------
    dataset
        gdal dataset to read from
    band_nums
        list of band number integers to read arrays from
    min_block_size
        minimum size to allow block reads to use
    max_block_size
        maximum size to allow block reads to use

    Returns
    -------
    ic, ir, nodata, data
        column index, row index, no data value, list of arrays from the dataset

    """
    bands = [dataset.GetRasterBand(num) for num in band_nums]
    block_sizes = bands[0].GetBlockSize()
    row_block_size = min(max(block_sizes[1], min_block_size), max_block_size)
    col_block_size = min(max(block_sizes[0], min_block_size), max_block_size)
    col_size = bands[0].XSize
    if end_col is not None and end_col >= 0 and end_col <= col_size:
        col_size = end_col
    row_size = bands[0].YSize
    if end_row is not None and end_row >= 0 and end_row <= row_size:
        row_size = end_row
    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0
    nodata = bands[0].GetNoDataValue()
    # read the data array in blocks
    for ic in tqdm(range(start_col, col_size, col_block_size), desc='column block', mininterval=.7, leave=leave_progress_bar):
        if ic + col_block_size < col_size:
            cols = col_block_size
        else:
            cols = col_size - ic
        for ir in tqdm(range(start_row, row_size, row_block_size), desc='row block', mininterval=.7, leave=False):
            if ir + row_block_size < row_size:
                rows = row_block_size
            else:
                rows = row_size - ir
            yield ic, ir, nodata, [band.ReadAsArray(ic, ir, cols, rows) for band in bands]


def iterate_gdal_buffered_image(dataset, col_buffer_size, row_buffer_size, band_nums=(1,), min_block_size=512, max_block_size=1024,
                                start_col=0, end_col=None, start_row=0, end_row=None, leave_progress_bar=True):
    """ Iterate a gdal dataset using blocks to reduce memory usage.
    The last blocks at the edge of the dataset will have a smaller size than the others.
    Reads down all the rows first then moves to the next group of columns.
    The function will use the dataset's GetBlockSize() if it falls between the min and max block size arguments

    Unlike the iterate_gdal_image() this will keep the data size consistent,
    i.e. when it reaches an edge of the dataset it will move the row/col read positions back to where they can read enough data.
    This is to allow mathematical operations to have a minimum amount of data available.
    Without this the 'data' portion could be a 1x1 which may not be desired.
    It also means that you may receive the same cells in multiple reads.

    Ex: a 10x10 image read in blocks of 4x4 with a buffer size of 1 would return
      5x5, 6x5, 5x6, 6x6, 6x6, 5x6, 6x5, 6x5, 5x5
    # (one based) iteration numbers the data should come back in
    table = numpy.array([[ 1, 1, 1,  14,  14,  47,  47,  47,  47,  7],
                         [ 1, 1, 1,  14,  14,  47,  47,  47,  47,  7],
                         [ 1, 1, 1,  14,  14,  47,  47,  47,  47,  7],
                         [12,12,124,1245,1245,4578,4578,4578,4578,78],
                         [12,12,124,1245,1245,4578,4578,4578,4578,78],
                         [23,23,236,2356,2356,5689,5689,5689,5689,89],
                         [23,23,236,2356,2356,5689,5689,5689,5689,89],
                         [23,23,236,2356,2356,5689,5689,5689,5689,89],
                         [23,23, 23,2356,2356,5689,5689,5689,5689,89],
                         [ 3, 3, 3,  36,  36,  69,  69,  69,  69,  9],
                         ])

    Parameters
    ----------
    dataset
        gdal dataset to read from
    row_buffer_size
        number of rows to additionally read on either side of the data array
    col_buffer_size
        number of columns to additionally read on either side of the data array
    band_nums
        list of band number integers to read arrays from
    min_block_size
        minimum size to allow block reads to use
    max_block_size
        maximum size to allow block reads to use

    Returns
    -------
    (ic, ir, cols, rows, col_buffer_lower, row_buffer_lower, nodata, data)
        ic = column index of the data, not including the buffer
        ir = row index of the data, not including the buffer
        cols = size of the data, not including buffers
        rows = size of the data, not including buffers
        col_buffer_lower = The amount of column buffer preceding the data
        row_buffer_lower = The amount of row buffer preceding the data
        nodata = no data value
        data = list of arrays from the dataset

    """
    # cast to ints since gdal will throw exception if float gets through (even if it's 1.0)
    col_buffer_size, row_buffer_size = int(col_buffer_size), int(row_buffer_size)
    min_block_size, max_block_size = int(min_block_size), int(max_block_size)

    bands = [dataset.GetRasterBand(num) for num in band_nums]
    block_sizes = bands[0].GetBlockSize()
    col_size = bands[0].XSize
    if end_col is not None and end_col >= 0 and end_col <= col_size:
        col_size = end_col
    row_size = bands[0].YSize
    if end_row is not None and end_row >= 0 and end_row <= row_size:
        row_size = end_row
    if start_row < 0:
        start_row = 0
    if start_col < 0:
        start_col = 0
    row_block_size = min(max(block_sizes[1], min_block_size), max_block_size, row_size)  # don't let the block size be bigger than the data
    col_block_size = min(max(block_sizes[0], min_block_size), max_block_size, col_size)  # don't let the block size be bigger than the data
    nodata = bands[0].GetNoDataValue()
    for ic in tqdm(range(start_col, col_size, col_block_size), desc='column block', mininterval=.7, leave=leave_progress_bar):
        cols = col_block_size
        if ic + col_block_size > col_size:  # read a full set of data by offsetting the column index back a bit
            ic = col_size - cols
        col_buffer_lower = col_buffer_size if ic >= col_buffer_size else ic
        col_buffer_upper = col_buffer_size if col_size - (ic + col_block_size) >= col_buffer_size else col_size - (ic + col_block_size)
        read_cols = col_buffer_lower + cols + col_buffer_upper
        for ir in tqdm(range(start_row, row_size, row_block_size), desc='row block', mininterval=.7, leave=False):
            rows = row_block_size
            if ir + row_block_size > row_size:
                ir = row_size - rows
            row_buffer_lower = row_buffer_size if ir >= row_buffer_size else ir
            row_buffer_upper = row_buffer_size if row_size - (ir + row_block_size) >= row_buffer_size else row_size - (ir + row_block_size)
            read_rows = row_buffer_lower + rows + row_buffer_upper

            yield (ic, ir, cols, rows, col_buffer_lower, row_buffer_lower, nodata,
                   [band.ReadAsArray(ic - col_buffer_lower, ir - row_buffer_lower, read_cols, read_rows) for band in bands])


class BufferedImageOps:
    def __init__(self, dataset_or_path):
        if isinstance(dataset_or_path, (str, pathlib.Path)):
            self.ds = gdal.Open(str(dataset_or_path), gdal.GA_Update)  # pathlib needs str() conversion
        else:
            self.ds = dataset_or_path
        # the data block location, not including the buffers
        self.ir = 0
        self.ic = 0
        # buffer sizes before the data
        self.row_buffer_lower = 0
        self.col_buffer_lower = 0
        # how much data is inside the arrays
        self.data_rows = 0
        self.data_cols = 0
        # size of what came back from the ReadAsArray calls
        self.read_shape = None

    def iterate_gdal(self, col_buffer_size, row_buffer_size, band_nums=(1,), min_block_size=512, max_block_size=1024,
                     start_col=0, end_col=None, start_row=0, end_row=None):
        """ See iterate_gdal_buffered_image() where dataset will be automatically supplied based on the __init__() for this instance """
        self.band_nums = band_nums
        for block in iterate_gdal_buffered_image(self.ds, row_buffer_size, col_buffer_size,
                                                 band_nums=band_nums, min_block_size=min_block_size, max_block_size=max_block_size,
                                                 start_col=start_col, end_col=end_col, start_row=start_row, end_row=end_row):
            self.ic, self.ir, self.data_cols, self.data_rows, self.col_buffer_lower, self.row_buffer_lower, nodata, data = block
            self.read_shape = data[0].shape
            yield block

    def _get_band_object(self, band=None):
        band_num = None
        if band is None:
            band_num = self.band_nums[0]
        elif isinstance(band, int):
            band_num = band
        # either didn't supply band or used an integer, so get a raster band from the dataset
        if band_num:
            band = self.ds.GetRasterBand(band_num)
        return band

    def read(self, band, buffered=True):
        band = self._get_band_object(band)
        if not buffered:
            data = band.ReadAsArray(self.ic, self.ir, self.data_cols, self.data_rows)
        else:
            data = band.ReadAsArray(self.ic - self.col_buffer_lower, self.ir - self.row_buffer_lower,
                                    self.read_shape[1], self.read_shape[0])  # numpy shape is row, col while gdal wants col, row
        return data

    def trim_buffer(self, array, buff_width):
        try:
            row_size, col_size = buff_width
        except TypeError:
            row_size = col_size = buff_width
        low_row = max(self.row_buffer_lower - row_size, 0)
        high_row = min(self.row_buffer_lower + self.data_rows + row_size, array.shape[0])
        low_col = max(self.col_buffer_lower - col_size, 0)
        high_col = min(self.col_buffer_lower + self.data_cols + col_size, array.shape[1])
        remaining_row_buffer_lower = row_size if low_row > 0 else self.row_buffer_lower
        remaining_col_buffer_lower = col_size if low_col > 0 else self.col_buffer_lower
        return remaining_row_buffer_lower, remaining_col_buffer_lower, array[low_row:high_row, low_col: high_col]

    def trim_array(self, array):
        """ Return the portion of the array that is the non-buffer area
        """
        # array[self.row_buffer_lower:self.row_buffer_lower + self.data_rows, self.col_buffer_lower: self.col_buffer_lower + self.data_cols]
        _low_buff, _low_col, data = self.trim_buffer(array, 0)
        return data

    def write_array(self, array, band=None):
        """ Write an array back into a gdal dataset in the band specified.  Band can be from a different dataset if the indexing matches.
        I.e. you could read this dataset using the iterate to get an array then write it back to another dataset, perhaps made using CreateCopy

        Parameters
        ----------
        array
            array of either the shape of the non-buffered data block or the entire read data from the last iterate_gdal_image call
        band
            None will use the first band number from the __init__ construction.
            If an integer is provided it is interpreted as the value to call GetRasterBand() with.
            Lastly, a Band object can be supplied which could be from this dataset or a different one
            - useful when transforming data between one file and another

        Returns
        -------
        None

        """
        band = self._get_band_object(band)
        # figure out if the supplied array includes the buffers or not
        if array.shape[0] == self.data_rows and array.shape[1] == self.data_cols:
            ic = self.ic
            ir = self.ir
        elif array.shape == self.read_shape:
            ic = self.ic - self.col_buffer_lower
            ir = self.ir - self.row_buffer_lower
        else:
            raise ValueError("The array did not match the buffered array size that was read or the non-buffered size")
        band.WriteArray(array, ic, ir)


def save_soundings_from_image(inputname, outputname, res, flip_depth=True):
    # import pickle
    # tmpname="c:\\temp\\sounding.numpy"
    # if os.path.exists(tmpname):
    #     fil = open(tmpname, "rb")
    #     wkt = pickle.load(fil)
    #     srs = osr.SpatialReference(wkt=wkt)
    #     sounding_array = pickle.load(fil)
    # else:

    srs, sounding_array = soundings_from_image(inputname, res)
    # make a geopackage of the x,y,z values held in the sounding matrix
    sounding_array = sounding_array.reshape(sounding_array.shape[0], -1)
    sounding_array = sounding_array[:, ~numpy.isnan(sounding_array[2])]
    if flip_depth:
        sounding_array[2] *= -1

    # fil = open(tmpname, "wb")
    # pickle.dump(srs.ExportToWkt(),fil)
    # pickle.dump(sounding_array, fil)

    dst_ds = ogr.GetDriverByName('Memory').CreateDataSource(outputname)
    lyr = dst_ds.CreateLayer('SOUNDG', srs, ogr.wkbPoint)

    # match the geopackage format from Caris
    for field in (
            # ogr.FieldDefn('SCAMIN', ogr.OFTInteger64),
            # ogr.FieldDefn('SCAMAX', ogr.OFTInteger64),
            # ogr.FieldDefn('EXPSOU', ogr.OFTString),
            # ogr.FieldDefn('NOBJNM', ogr.OFTString),
            # ogr.FieldDefn('OBJNAM', ogr.OFTString),
            # ogr.FieldDefn('QUASOU', ogr.OFTString),
            ogr.FieldDefn('SOUACC', ogr.OFTReal),
            # ogr.FieldDefn('STATUS', ogr.OFTString),
            # ogr.FieldDefn('TECSOU', ogr.OFTString),
            # ogr.FieldDefn('VERDAT', ogr.OFTString),
            ogr.FieldDefn('SORDAT', ogr.OFTString),
            ogr.FieldDefn('SORIND', ogr.OFTString),
            # ogr.FieldDefn('remrks', ogr.OFTString),
            # ogr.FieldDefn('descrp', ogr.OFTString),
            # ogr.FieldDefn('recomd', ogr.OFTString),
            # ogr.FieldDefn('sftype', ogr.OFTString),
            # ogr.FieldDefn('obstim', ogr.OFTString),
            # ogr.FieldDefn('INFORM', ogr.OFTString),
            # ogr.FieldDefn('NINFOM', ogr.OFTString),
            # ogr.FieldDefn('NTXTDS', ogr.OFTString),
            # ogr.FieldDefn('TXTDSC', ogr.OFTString),
            # ogr.FieldDefn('userid', ogr.OFTString),
            # ogr.FieldDefn('prmsec', ogr.OFTString),
            # ogr.FieldDefn('prkyid', ogr.OFTString),
            # ogr.FieldDefn('asgnmt', ogr.OFTString),
            # ogr.FieldDefn('invreq', ogr.OFTString),
            # ogr.FieldDefn('acqsts', ogr.OFTString),
            # ogr.FieldDefn('images', ogr.OFTString),
            # ogr.FieldDefn('keywrd', ogr.OFTString),
            # ogr.FieldDefn('obsdpt', ogr.OFTReal),
            # ogr.FieldDefn('tidadj', ogr.OFTReal),
            # ogr.FieldDefn('tidfil', ogr.OFTString),
            # ogr.FieldDefn('cnthgt', ogr.OFTReal),
            # ogr.FieldDefn('updtim', ogr.OFTString),
            # ogr.FieldDefn('dbkyid', ogr.OFTString),
            # ogr.FieldDefn('hsdrec', ogr.OFTString),
            # ogr.FieldDefn('onotes', ogr.OFTString)
    ):
        # if field.GetType()==ogr.OFTString:
        #     field.SetWidth(254)
        if 0 != lyr.CreateField(field):
            raise RuntimeError("Creating field failed.", field.GetName())

    sounding_array = sounding_array.astype(numpy.float).T
    for x, y, z in sounding_array:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), float(y), float(z))
        # Create a feature, using the attributes/fields that are required for this layer
        feat = ogr.Feature(feature_def=lyr.GetLayerDefn())
        feat.SetGeometry(point)
        lyr.CreateFeature(feat)
        # Clean up
        feat.Destroy()
    ogr.GetDriverByName('GPKG').CopyDataSource(dst_ds, dst_ds.GetName())


def calc_area_array_params(x1, y1, x2, y2, res_x, res_y, align_x=None, align_y=None):
    """ Compute a coherent min and max position and shape given a resolution.
    Basically we may know the desired corners but they won't fit perfectly based on the resolution.
    So we will compute what the adjusted corners would be and number of cells needed based on the resolution provided.

    ex: (0, 10, 5, 0, 4, 3) would return (0, 0, 8, 12, 2, 4)
    The minimum is held constant (0,0) the other corner would be moved from (5, 10) to (8, 12) because the resolution was (4,3)
    and there would be 2 columns (xsize) and 4 rows (ysize)

    Parameters
    ----------
    x1
        an X corner coordinate
    y1
        an Y corner coordinate
    x2
        an X corner coordinate
    y2
        an Y corner coordinate
    res_x
        pixel size in x direction
    res_y
        pixel size in y direction
    align_x
        if supplied the min_x will be shifted to align to an integer cell offset from the align_x, if None then no effect
    align_y
        if supplied the min_y will be shifted to align to an integer cell offset from the align_y, if None then no effect
    Returns
    -------
    min_x, min_y, max_x, max_y, cols (shape_x), rows (shape_y)

    """
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    if align_x:
        min_x -= (min_x - align_x) % res_x
    if align_y:
        min_y -= (min_y - align_y) % res_y
    shape_x = int(numpy.ceil((max_x - min_x) / res_x))
    shape_y = int(numpy.ceil((max_y - min_y) / res_y))
    max_x = shape_x * res_x + min_x
    max_y = shape_y * res_y + min_y
    return min_x, min_y, max_x, max_y, shape_x, shape_y


def compute_delta_coord(x, y, dx, dy, crs_transform, inv_crs_transform):
    """ Given a refernce point, desired delta and geotransform+inverse, compute what the desired delta in the target SRS is in the source SRS.
    For example, 4 meters in Mississippi would be (epsg 4326 is WGS84 and 26915 is UTMzone 15N):
    compute_delta_coord(-93, 20, 4, 4, get_geotransform(4326, 26915), get_geotransform(26915, 4326))
    Parameters
    ----------
    x
        X coordinate to work from in the source reference system
    y
        Y coordinate to work from in the source reference system
    dx
        The X distance desired in the target reference system
    dy
        The Y distance desired in the target reference system
    crs_transform
        a transformation (something with a .transform method) going from source to target
    inv_crs_transform
        a transformation (something with a .transform method) going from target to source

    Returns
    -------
    dx, dy
        The delta in x and y in the source reference system for the given delta in the target system
    """
    target_x, target_y = crs_transform.transform(x, y)
    x2, y2 = inv_crs_transform.transform(target_x + dx, target_y + dy)
    sdx = numpy.abs(x2 - x)
    sdy = numpy.abs(y2 - y)
    return sdx, sdy


def compute_delta_coord_epsg(x, y, dx, dy, source_epsg, target_epsg):
    crs_transform = get_crs_transformer(source_epsg, target_epsg)
    inv_crs_transform = get_crs_transformer(target_epsg, source_epsg)
    return compute_delta_coord(x, y, dx, dy, crs_transform, inv_crs_transform)


def make_gdal_dataset_size(fname, bands, min_x, max_y, res_x, res_y, shape_x, shape_y, epsg,
                           driver="GTiff", options=(), nodata=numpy.nan, etype=gdal.GDT_Float32):
    """ Makes a north up gdal dataset with nodata = numpy.nan and LZW compression.
    Specifying a positive res_y will be input as a negative value into the gdal file,
    since tif/gdal likes max_y and a negative Y pixel size.
    i.e. the geotransform in gdal will be stored as [min_x, res_x, 0, max_y, 0, -res_y]

    Parameters
    ----------
    fname
        filename to create
    bands
        number of bands to create or a list of names for bands in the file
    min_x
        minimum X coordinate
    max_y
        maximum Y coordinate (because tiff images like to specify max Y and a negative res_y)
    res_x
        pixel size in x direction
    res_y
        pixel size in y direction
    shape_x
        number of pixels in X direction (columns)
    shape_y
        number of pixels in Y directions (rows)
    epsg
        epsg of the target coordinate system
    driver
        gdal driver name of the output file (defaults to geotiff)
    options
        gdal driver options.  Generally bag metadata or tiff compression settings etc
    nodata
        no_data value - defaults to nan but for bag will be set to 1000000
    etype
        data type to store in the tif, usually a float but could be gdal.GDT_Int32 etc
    Returns
    -------
    gdal.dataset

    """
    gdal_driver = gdal.GetDriverByName(driver)
    if driver.lower() == 'bag':  # this is required in the spec
        nodata = 1000000.0
    if not options:
        if driver.lower() == "gtiff":
            options = ['COMPRESS=LZW', "TILED=YES", "BIGTIFF=YES"]
        if driver.lower() == 'bag':
            options = list(options)
            options.insert(0,
                           'TEMPLATE=G:\\Pydro_new_svn_1\\Pydro21\\NOAA\\site-packages\\Python38\\git_repos\\hstb_resources\\HSTB\\resources\\gdal_bag_template.xml')

    try:
        num_bands = int(bands)
    except:
        num_bands = len(bands)
    dataset = gdal_driver.Create(str(fname), xsize=shape_x, ysize=shape_y, bands=num_bands, eType=etype,
                                 options=options)

    # Set location
    gt = [min_x, res_x, 0, max_y, 0, -res_y]  # north up
    dataset.SetGeoTransform(gt)

    if epsg is not None:
        if isinstance(epsg, (float, int)):
            # Get raster projection
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            dest_wkt = srs.ExportToWkt()
        elif isinstance(epsg, str):
            dest_wkt = epsg
        # Set projection
        dataset.SetProjection(dest_wkt)
    for b in range(dataset.RasterCount):
        band = dataset.GetRasterBand(b + 1)
        band.SetNoDataValue(nodata)
        if driver == 'GTiff':
            break
    del band
    return dataset


def make_gdal_dataset_area(fname, bands, x1, y1, x2, y2, res_x, res_y, epsg,
                           driver="GTiff", options=(), nodata=numpy.nan,
                           align_x=None, align_y=None):
    """ Makes a north up gdal dataset with nodata = numpy.nan and LZW compression.
    Specifying a positive res_y will be input as a negative value into the gdal file,
    since tif/gdal likes max_y and a negative Y pixel size.
    i.e. the geotransform in gdal will be stored as [min_x, res_x, 0, max_y, 0, -res_y]

    Note: the minimum x,y will be honored, so if there is a mismatch in the coordinates and resolution
    then the minimum values for the coordinates will be used with the necessary cell count and the maximum value
    will be adjusted.

    Ex: x1,y1 = 0,0  x2,y2 = 5,6  resx, resy = 4,4  will result in a 2x2 matrix where

    minx, miny = 0,0  and  res_x, res_y = 4,4 are retained while modifying  max_x, max_y = 8,8

    Calls make_gdal_dataset_size
    Parameters
    ----------
    fname
        filename to create
    bands
        number of bands to create or list of names of bands in the file
    x1
        an X corner coordinate
    y1
        an Y corner coordinate
    x2
        an X corner coordinate
    y2
        an Y corner coordinate
    res_x
        pixel size in x direction
    res_y
        pixel size in y direction
    epsg
        epsg of the target coordinate system
    driver
        gdal driver name of the output file (defaults to geotiff)
    options
        gdal driver options.  Generally bag metadata or tiff compression settings etc
    nodata
        no_data value - defaults to nan but for bag will be set to 1000000
    Returns
    -------
    gdal.dataset

    """
    min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y, align_x=align_x, align_y=align_y)
    dataset = make_gdal_dataset_size(fname, bands, min_x, max_y, res_x, res_y, shape_x, shape_y, epsg, driver, options, nodata)
    return dataset


def add_uncertainty_layer(infile, outfile, depth_mult=0.01, uncert_offset=0.5, depth_band=1, driver='GTIFF'):
    ds = gdal.Open(infile)
    tmp_ds = gdal.GetDriverByName('MEM').CreateCopy('', ds, 0)
    band = ds.GetRasterBand(1)
    tmp_ds.AddBand(band.DataType)
    depth = band.ReadAsArray()
    uncert = depth * depth_mult + uncert_offset
    uncert[depth == band.GetNoDataValue()] = band.GetNoDataValue()
    tmp_ds.GetRasterBand(ds.RasterCount + 1).WriteArray(uncert)

    new_ds = gdal.GetDriverByName(driver).CreateCopy(outfile, tmp_ds, 0, options=['COMPRESS=LZW'])
    new_ds.FlushCache()

    # this is meaningless since it does out of scope -- maybe garbage collects sooner?
    del new_ds
    del tmp_ds
    del ds


def transform_rect(x1, y1, x2, y2, transform_func):
    # convert a rectangle to the minimum fully enclosing rectangle in transformed coordinates
    # @todo if a transform is curved, then bisect to find the maximum which may not be the corners
    tx1, ty1 = transform_func(x1, y1)
    tx2, ty2 = transform_func(x2, y2)
    tx3, ty3 = transform_func(x1, y2)
    tx4, ty4 = transform_func(x2, y1)
    xs = (tx1, tx2, tx3, tx4)
    ys = (ty1, ty2, ty3, ty4)
    return min(xs), min(ys), max(xs), max(ys)


def find_overrides_in_log(log_path):
    data = open(log_path).readlines()
    overrides = {}
    for line in data:
        m = re.search("override (\d+|\w+)\swith\s(\d+)\sin\s(.*)", line)
        if m:
            if m.groups()[0] != m.groups()[1]:
                overrides[m.groups()[2]] = line[m.start():-1]
    for line in overrides.values():
        print(line)


def remove_file(pth: (str, pathlib.Path), allow_permission_fail: bool = False, limit: int = 2, nth: int = 1, tdelay: float = 2,
                silent: bool = False, raise_on_fail=False):
    """ Try to remove a file and just print a warning if it doesn't work.
    Will retry 'nth' times every 'tdelay' seconds up to 'limit' times.
    Will not raise an error on FileNotFound but will for PermissionError unless allow_permission_fail is set to True.

    Parameters
    ----------
    pth
        path to the file to remove
    allow_permission_fail
        True = continue trying to remove the file if a permission error is encountered, False = raise exception
    limit
        number of attempts to make
    nth
        attempt number this is
    tdelay
        time in seconds to wait between attempts
    silent
        False = print a message when the file isn't removed due to not being found or,
        depending on allow_permission_fail, file being in use/not having permissions
    raise_on_fail
        If allow_permission_fail was True then raises an exception (rather than printing a message)
        if os.remove is still failing after specified number of attempts.
        Note: FileNotFound will pass the first time so this only applies to any additional exceptions being caught
        (permission is currently the only one).
    Returns
    -------

    """

    if allow_permission_fail:
        ok_except = (FileNotFoundError, PermissionError)
    else:
        ok_except = (FileNotFoundError,)
    success = False
    try:
        os.remove(pth)
        success = True
    except ok_except as ex:
        if isinstance(ex, FileNotFoundError):
            success = True
        else:
            if nth > limit:
                if not silent:
                    if raise_on_fail:
                        raise ex
                    else:
                        print(f"File not found or permission error {type(ex)}, {pth}")
            else:
                time.sleep(tdelay)
                success = remove_file(pth, allow_permission_fail, nth=nth + 1, silent=silent)
    return success


def num_active_processes(cmds, ignore_pids=(), ordered=True, excludes=()):
    # cmd.exe has all the python commands in it too, so is getting double counted and stays permanently if the cmd window doesn't close after errors
    # so use the ordered flag to be able to distinguish between "cmd.exe /K python"  and just "python" processes
    count = 0
    for p in psutil.pids():
        try:
            if p not in ignore_pids:
                cmdline = psutil.Process(p).cmdline()
                if ordered:
                    try:
                        has_cmds = [cmdline[n] == cmd for n, cmd in enumerate(cmds)]
                    except IndexError:
                        has_cmds = [False]
                else:
                    has_cmds = [cmd in cmdline for cmd in cmds]
                is_excluding = [cmd not in cmdline for cmd in excludes]
                has_cmds.extend(is_excluding)
                if all(has_cmds):
                    count += 1
        except:  # permission errors and things end up here
            pass
    return count


def wait_for_processes(process_list, max_processes, ignore_pids, ordered=True):
    while num_active_processes(process_list, ignore_pids, ordered=ordered) >= max_processes:
        print(".", end="")
        time.sleep(30)


class ProcessTracker:
    """ Do not supply cmds or excludes with strings in them"""

    def __init__(self, cmds, excludes=(), ignore_pids=()):
        """ Do not supply cmds or excludes with strings in them"""
        self.cmds = cmds
        self.excludes = excludes
        self.ignore_pids = ignore_pids
        self.last_started = datetime.now()
        self.runs = 0
        self.last_pid = self.find()

    def find(self):
        if platform.system() == 'Windows':
            separator = "&"
        else:
            separator = ";"
        ret = None
        for p in psutil.pids():
            try:
                if p not in self.ignore_pids:
                    cmdline_raw = psutil.Process(p).cmdline()
                    # the cmdline comes back as a list of arguments, basically the commandline split by spaces
                    # but some commands were separated by a && or ; (depending on platform) so we want to further split the strings
                    # without this we were not finding 'python' correctly as it was appended to the end of the PYTHONPATH line.
                    # Could have also done cmd=separator.join(cmds) then cmds=cmd.split(separator)
                    cmdline = []
                    for cmd in cmdline_raw:
                        sub_cmds = cmd.split(separator)
                        if platform.system() == "Windows":  # windows breaks command line arguments into separate list items
                            cmdline.extend(sub_cmds)
                        else:  # linux leaving the arguments as a combined string so split on spaces - may fail for search items with strings
                            # FIXME - allow for quoted arguments to not get split so spaces in parameters would work
                            for sub_cmd in sub_cmds:
                                cmdline.extend(sub_cmd.split(" "))
                    has_cmds = [cmd in cmdline for cmd in self.cmds]
                    is_excluding = [cmd not in cmdline for cmd in self.excludes]
                    has_cmds.extend(is_excluding)
                    if all(has_cmds):
                        ret = p
                        break
            except:  # permission errors and things end up here
                pass
        return ret

    def is_zombie(self):
        ret = False
        if self.last_pid is None:
            self.last_pid = self.find()
        if self.last_pid is not None:
            proc = psutil.Process(self.last_pid)
            ret = proc.status() == psutil.STATUS_ZOMBIE
        return ret

    def is_running(self, timeout=300):
        """
        Returns
        -------
        """
        if self.last_pid is None:
            self.last_pid = self.find()
        if self.last_pid is not None:
            try:
                proc = psutil.Process(self.last_pid)
                running = proc.is_running() and not proc.status() == psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:
                running = False
        else:
            # see if it has been long enough (five minutes by default)
            if datetime.now() - self.last_started > timedelta(0, timeout):
                running = False
            else:
                running = True  # we didn't find the process but it hasn't been long enough to give up
        return running


class ConsoleProcessTracker:
    def __init__(self, cmds, excludes=(), ignore_pids=(), console_str=None):
        if console_str is None:
            if platform.system() == "Windows":
                console_str = "cmd.exe"
            else:
                console_str = 'sh'
        self.console = ProcessTracker(list(cmds) + [console_str], excludes, ignore_pids)
        if platform.system() == "Windows":
            self.app = ProcessTracker(cmds, list(excludes) + [console_str])
        else:  # linux is not showing a second process id like Windows is, so just track the sh process
            self.app = self.console

    def is_running(self):
        console = self.console.is_running()
        app = self.app.is_running()
        return console and app


def popen_kwargs(new_console=True, activate=True, minimize=False):
    kwargs = {}
    kwargs['creationflags'] = subprocess.CREATE_NEW_CONSOLE if new_console else 0

    # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-showwindow
    SW_SHOWMINNOACTIVE = 7
    SW_MINIMIZE = 6
    SW_SHOWNOACTIVATE = 4
    SW_SHOWNA = 8
    if minimize or activate:  # runs minimized without taking focus from the current app
        info = subprocess.STARTUPINFO()
        info.dwFlags = subprocess.STARTF_USESHOWWINDOW
        if minimize and activate:
            info.wShowWindow = SW_MINIMIZE
        elif minimize:
            info.wShowWindow = SW_SHOWMINNOACTIVE
        else:  # activate
            info.wShowWindow = SW_SHOWNOACTIVATE
    else:
        info = None
    kwargs['startupinfo'] = info
    return kwargs

