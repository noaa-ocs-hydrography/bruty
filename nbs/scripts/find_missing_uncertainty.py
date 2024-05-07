import sys

from osgeo import gdal
import numpy

def find_missing_uncert(fname):
    """ Find pixels where uncertainty is missing but depth is present
    Parameters
    ----------
    fname
        Path to the 3 band tif to check.  Band 1 is depth, band 2 is uncertainty, band 3 is contributor

    Returns
    -------
    bad_indices, bad_positions, bad_contributors
        bad_indices are the indices of the pixels where uncertainty is missing but depth is present
        bad_positions are the x, y positions of the bad pixels
        bad_contributors are the contributors of the bad pixels
    """
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
        print("No mismatches found")
        return bad_indices, numpy.zeros([2, 0]), bad_indices[0]


if __name__ == "__main__":
    r""" Use this script on an exported TIF file from Bruty or Xipe to find missing uncertainty values.
     
    Usage: python find_missing_uncertainty.py <path_to_file> 
    e.g. python find_missing_uncertainty.py w:\buty_tile_exports\PBC19\file.tif"""
    file_path = sys.argv[1]
    if file_path:
        ind, bad_p, bad_c = find_missing_uncert(file_path)
        with numpy.printoptions(precision=9, suppress=True):
            for i in numpy.unique(bad_c):
                pts = bad_p[numpy.where(bad_c == i)]
                print(f"Total bad points for contributor {int(i)}:", len(pts))
                print(f"    example point position:  {pts[0, 0]}, {pts[0, 1]} ")