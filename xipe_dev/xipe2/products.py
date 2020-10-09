from xipe_dev.xipe2.abstract import VABC, abstractmethod

class ProductArea(VABC):
    def __init__(self, name, date, geom, crs):
        pass

class ProductParameters(VABC):
    def __init__(self, src_interp_generalized, soundings, contours, closing_dist, resolution, min_score):
        pass

class Product(VABC):
    def __init__(self, requestor, processor, area, parameters, hash):

