from xipe2.abstract import VABC

class RasterDelta(VABC):
    def __init__(self, data_raster):
        self.data = data_raster
        self._version = 1
    def get_array(self):
        return self.data.get_array()
    def set_array(self, array):
        self.data.set_array(data_raster)

    @staticmethod
    def from_rasters(raster_one, raster_two):
        pass
