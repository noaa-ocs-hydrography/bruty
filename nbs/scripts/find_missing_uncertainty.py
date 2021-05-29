import sys

from osgeo import gdal
import numpy

def find_missing_uncert(fname):
    ds = gdal.Open(fname)
    print("load depths")
    depth = ds.GetRasterBand(1).ReadAsArray()
    print("load uncertainties")
    uncert = ds.GetRasterBand(2).ReadAsArray()
    no_data = ds.GetRasterBand(2).GetNoDataValue()
    print("find mismatches")
    if numpy.isnan(no_data):
        bad_indices = numpy.where(numpy.logical_and(~numpy.isnan(depth), numpy.isnan(uncert)))
    else:
        bad_indices = numpy.where(numpy.logical_and(depth != no_data, uncert == no_data))
    if len(bad_indices[0]) > 0:
        print("compute positions")
        geotransform = ds.GetGeoTransform()
        bad_x = bad_indices[1] * geotransform[1] + geotransform[0]
        bad_y = bad_indices[0] * geotransform[5] + geotransform[3]
        print("load contributors")
        contrib = ds.GetRasterBand(3).ReadAsArray()
        bad_c = contrib[bad_indices]
        return bad_indices, numpy.vstack((bad_x, bad_y)).T, bad_c
    else:
        return bad_indices, numpy.zeros([2,0]), bad_indices[0]

if __name__ == "__main__":
    file_path = sys.argv[1]
    if file_path:
        ind, bad_p, bad_c = find_missing_uncert(file_path)
        with numpy.printoptions(precision=9, suppress=True):
            for i in range(len(bad_c)):
                print(bad_p[i, 0], bad_p[i, 1], int(bad_c[i]), sep=", ")