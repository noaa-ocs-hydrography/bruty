import pathlib
import os

import numpy

data_dir = pathlib.Path(__file__).parent.joinpath("test_data_output")
os.makedirs(data_dir, exist_ok=True)
arr = numpy.zeros((4, 2, 1))
nan = numpy.nan

# these are the final arrays (after supercession/combine process to test making deltas and putting them into the history
master_data = {
    "more overwrite": (arr+0, arr+1, arr+2, arr+3, arr+4, arr+5),
    "full overwrite": ([[0, 0], [0, 0], [0, 0], [0, 0]], [[1, 1], [1, 1], [1, 1], [1, 1]], [[2, 2], [2, 2], [2, 2], [2, 2]]),
    "update stripes": ([[0, 0], [0, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [0, 0], [0, 0]], [[0, 0], [1, 0], [2, 2], [0, 0]]),
    "stripes with nan": ([[0, 0], [nan, nan], [nan, nan], [0, 0]], [[0, 0], [1, 1], [nan, nan], [0, 0]], [[0, 0], [1, 1], [2, 2], [0, 0]]),
}
