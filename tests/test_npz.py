import numpy
import os

import pytest

from nbs_utils.points_utils import to_npz, mmap_from_npz

def test_npz_mmap():
    # create .npz file
    fname = 'test.npz'
    x = numpy.arange(255, dtype=numpy.uint8)
    wkt = "Test WKT[{<()>}]"
    # save npz file
    to_npz(fname, wkt, x)

    # load as zip file
    wkt2, x2 = mmap_from_npz(fname)
    # memmap should prevent removal
    with pytest.raises(PermissionError) as e_info:
        os.remove(fname)
    # compare
    assert numpy.allclose(x, x2)

    del wkt, x2
    os.remove(fname)
