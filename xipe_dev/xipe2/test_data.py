import pathlib
import os

import numpy

data_dir = pathlib.Path(__file__).parent.joinpath("test_data_output")
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
