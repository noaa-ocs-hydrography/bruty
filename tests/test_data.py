import pathlib
import os
import shutil

import numpy

from nbs.bruty.utils import onerr

data_dir = pathlib.Path(__file__).parent.joinpath("test_data_output")

def make_clean_dir(name):
    use_dir = data_dir.joinpath(name)
    if os.path.exists(use_dir):
        shutil.rmtree(use_dir, onerror=onerr)
    return use_dir

os.makedirs(data_dir, exist_ok=True)
arr = numpy.zeros((5, 2, 1))
nan = numpy.nan
arr0 = arr+0
arr0[:, 0] = nan  # insert a nan in the first row of the first dataset to represent empty
arr1 = arr+1
arr1[:, 1] = nan  # insert a nan in the second row of the first dataset to represent empty
arr2 = arr1.copy()
arr2[0, 0] += 1  # Leave some data the same and change other

# these are the final arrays (after supercession/combine process to test making deltas and putting them into the history
master_data = {
    "test rasters": (arr0, arr1, arr2, arr+3, arr+4, arr+5),
    # "full overwrite": ([[0, 0], [0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2], [2, 2]]),
    # "update stripes": ([[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [2, 2], [0, 0]]),
    # "stripes with nan": ([[0, 0], [nan, nan], [nan, nan], [0, 0]], [[0, 0], [1, 1], [nan, nan], [0, 0]], [[0, 0], [1, 1], [2, 2], [0, 0]]),
}

r, c = numpy.indices([5,5])
"""
(array([[0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4]]),
 array([[0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4]]))
    """

z = numpy.array(r + c * 10)  # numpy.array(i+j*10)
"""
array([[ 0, 10, 20, 30, 40],
       [ 1, 11, 21, 31, 41],
       [ 2, 12, 22, 32, 42],
       [ 3, 13, 23, 33, 43],
       [ 4, 14, 24, 34, 44]])
    """
uncertainty = numpy.full(r.shape, 1.5)
score = numpy.full(r.shape, 2)
flags = numpy.full(r.shape, 1)

SW_5x5 = [r, c, z, uncertainty, score, flags]
SE_5x5 = [r, c + 5, z + 100, uncertainty + 1, score + 1, flags]
NW_5x5 = [r + 5, c, z + 200, uncertainty + 2, score + 2, flags]
MID_5x5 = [r + 2, c + 2, z * 0 + 999, uncertainty + 0.75, score + 0.5, flags]

""" This should have made an array like this (remember that increasing Y does down the array with increasing row)
array([[  0,  10,  20,  30,  40, 100, 110, 120, 130, 140],
       [  1,  11,  21,  31,  41, 101, 111, 121, 131, 141],
       [  2,  12,  22,  32,  42, 102, 112, 122, 132, 142],
       [  3,  13,  23,  33,  43, 103, 113, 123, 133, 143],
       [  4,  14,  24,  34,  44, 104, 114, 124, 134, 144],
       [200, 210, 220, 230, 240, nan, nan, nan, nan, nan],
       [201, 211, 221, 231, 241, nan, nan, nan, nan, nan],
       [202, 212, 222, 232, 242, nan, nan, nan, nan, nan],
       [203, 213, 223, 233, 243, nan, nan, nan, nan, nan],
       [204, 214, 224, 234, 244, nan, nan, nan, nan, nan]],
       ])

So, displayed in Arc it would look like: 
array([[204., 214., 224., 234., 244.,  nan,  nan,  nan,  nan,  nan],
       [203., 213., 223., 233., 243.,  nan,  nan,  nan,  nan,  nan],
       [202., 212., 222., 232., 242.,  nan,  nan,  nan,  nan,  nan],
       [201., 211., 221., 231., 241., 999., 999.,  nan,  nan,  nan],
       [200., 210., 220., 230., 240., 999., 999.,  nan,  nan,  nan],
       [  4.,  14., 999., 999., 999., 104., 114., 124., 134., 144.],
       [  3.,  13., 999., 999., 999., 103., 113., 123., 133., 143.],
       [  2.,  12., 999., 999., 999., 102., 112., 122., 132., 142.],
       [  1.,  11.,  21.,  31.,  41., 101., 111., 121., 131., 141.],
       [  0.,  10.,  20.,  30.,  40., 100., 110., 120., 130., 140.]],

Then put an array in the center with all 999. 

array([[999, 999, 999, 999, 999],
       [999, 999, 999, 999, 999],
       [999, 999, 999, 999, 999],
       [999, 999, 999, 999, 999],
       [999, 999, 999, 999, 999]])

Based on the scores it should overwrite some values fill some empty space       

array([[  0,  10,  20,  30,  40, 100, 110, 120, 130, 140],
       [  1,  11,  21,  31,  41, 101, 111, 121, 131, 141],
       [  2,  12, 999, 999, 999, 102, 112, 122, 132, 142],
       [  3,  13, 999, 999, 999, 103, 113, 123, 133, 143],
       [  4,  14, 999, 999, 999, 104, 114, 124, 134, 144],
       [200, 210, 220, 230, 240, 999, 999, nan, nan, nan],
       [201, 211, 221, 231, 241, 999, 999, nan, nan, nan],
       [202, 212, 222, 232, 242, nan, nan, nan, nan, nan],
       [203, 213, 223, 233, 243, nan, nan, nan, nan, nan],
       [204, 214, 224, 234, 244, nan, nan, nan, nan, nan]],
      dtype=float32)
"""
