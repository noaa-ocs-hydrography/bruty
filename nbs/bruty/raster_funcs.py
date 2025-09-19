# From xipe_dev originally
import os
from typing import Any, Union

import rasterio

from nbs.configs import get_logger

LOGGER = get_logger('nbs.rastr')


def parse_filename(filename: str) -> (str, Union[str, int]):
    if ':' in os.path.splitdrive(filename)[-1]:
        filename, layer = filename.rsplit(':', 1)
        try:
            layer = int(layer)
        except ValueError:
            pass
    else:
        layer = None

    return filename, layer


def raster_band_name_index(filename: str, band_name: str) -> int:
    filename = parse_filename(filename)[0]
    with rasterio.open(filename) as raster:
        band_names = [band_name.capitalize() if band_name is not None else None for band_name in raster.descriptions]

    try:
        return band_names.index(band_name.capitalize()) + 1
    except ValueError:
        raise KeyError(f'no band "{band_name}" found in "{filename}"')
