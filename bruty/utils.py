import numpy
import rasterio.crs
from osgeo import gdal, osr, ogr
from pyproj import Transformer, CRS

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterate_stuff, *args, **kywrds):
        return iterate_stuff  # if this doesn't work, try iter(iterate_stuff)


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
        c = numpy.array(numpy.floor((x - x0) / dxx), dtype=numpy.int32)
        r = numpy.array(numpy.floor((y - y0) / dyy), dtype=numpy.int32)
    else:
        # @todo support skew projection
        raise ValueError("non-North up affine transforms are not supported yet")
    return r, c


def affine_center(r, c, x0, dxx, dyx, y0, dxy, dyy):
    return affine(r + 0.5, c + 0.5, x0, dxx, dyx, y0, dxy, dyy)


def get_geotransformer(epsg1, epsg2):
    if epsg1 != epsg2:
        input_crs = CRS.from_epsg(epsg1)
        output_crs = CRS.from_epsg(epsg2)
        georef_transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)
    else:
        georef_transformer = None
    return georef_transformer


def merge_arrays(x, y, keys, data, *args, **kywrds):
    # create a combined two dimensional array that can then be indexed and reduced as needed,
    #   if the incoming data was multidimensional this will flatten it too.
    #   So an incoming dataset is say 6 fields of 4 rows and 5 columns will be turned into 10 x 20 -- (6+4 fields x 4 rows * 5 cols)
    #   wouldn't be able to remove nans and potentially do some other operations otherwise(?).
    pts = numpy.array((x, y, *keys, *data)).reshape(2 + len(keys) + len(data), -1)
    merge_array(pts, *args, **kywrds)


def merge_array(pts, output_data, output_sort_key_values,
                affine_transform=None, start_col=0, start_row=0, block_cols=None, block_rows=None,
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
        @todo add the ability to have a callback or specify the range is and/or so excludes would work.

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
        if affine_transform is not None:
            export_rows, export_cols = inv_affine(sorted_pts[0], sorted_pts[1], *affine_transform)
        else:
            export_rows, export_cols = sorted_pts[0].astype(numpy.int32), sorted_pts[1].astype(numpy.int32)
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
        merge_arrays(x, y, (depths, x, y), (x, y, depths), output_array, output_sort_values, output_xform)
    return srs, output_array


def iterate_gdal_image(dataset, band_nums=(1,), min_block_size=512, max_block_size=1024):
    bands = [dataset.GetRasterBand(num) for num in band_nums]
    block_sizes = bands[0].GetBlockSize()
    row_block_size = min(max(block_sizes[1], min_block_size), max_block_size)
    col_block_size = min(max(block_sizes[0], min_block_size), max_block_size)
    col_size = bands[0].XSize
    row_size = bands[0].YSize
    nodata = bands[0].GetNoDataValue()
    # read the data array in blocks
    for ic in tqdm(range(0, col_size, col_block_size), mininterval=.7):
        if ic + col_block_size < col_size:
            cols = col_block_size
        else:
            cols = col_size - ic
        for ir in tqdm(range(0, row_size, row_block_size), mininterval=.7):
            if ir + row_block_size < row_size:
                rows = row_block_size
            else:
                rows = row_size - ir
            yield ic, ir, nodata, [band.ReadAsArray(ic, ir, cols, rows) for band in bands]


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

def calc_area_array_params(x1, y1, x2, y2, res_x, res_y):
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

    Returns
    -------
    min_x, min_y, max_x, max_y, cols (shape_x), rows (shape_y)

    """
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1, x2)
    max_y = max(y1, y2)
    shape_x = int(numpy.ceil((max_x - min_x) / res_x))
    shape_y = int(numpy.ceil((max_y - min_y) / res_y))
    max_x = shape_x * res_x + min_x
    max_y = shape_y * res_y + min_y
    return min_x, min_y, max_x, max_y, shape_x, shape_y


def make_gdal_dataset_area(fname, bands, x1, y1, x2, y2, res_x, res_y, epsg, driver="GTiff"):
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
        list of names of bands in the file
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

    Returns
    -------
    gdal.dataset

    """
    min_x, min_y, max_x, max_y, shape_x, shape_y = calc_area_array_params(x1, y1, x2, y2, res_x, res_y)
    dataset = make_gdal_dataset_size(fname, bands, min_x, max_y, res_x, res_y, shape_x, shape_y, epsg, driver)
    return dataset


