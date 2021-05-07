import numpy

from nbs.bruty.abstract import VABC

class TilingScheme(VABC):
    """
    A generic coordinate to tile index class.
    Given an area and zoom level it will compute tile indices for a given coordinate (or array of coordinates).
    Also it will return the boundaries of a tile given the index.
    """

    def __init__(self, min_x=-180, min_y=-90.0, max_x=180.0, max_y=90.0, zoom=13, epsg=None):
        self._version = 1
        self._loaded_from_version = None
        self.min_y = min(min_y, max_y)
        self.max_y = max(min_y, max_y)
        self.min_x = min(min_x, max_x)
        self.max_x = max(min_x, max_x)
        self.zoom = zoom
        self.epsg = epsg

    def for_json(self):
        json_dict = {'class': self.__class__.__name__,
                     'module': self.__class__.__module__,
                     'version': self._version,
                     'min_x': self.min_x,
                     'max_x': self.max_x,
                     'min_y': self.min_y,
                     'max_y': self.max_y,
                     'zoom': self.zoom,
                     'epsg': self.epsg,
                     }
        return json_dict

    @staticmethod
    def create_from_json(json_dict):
        cls = eval(json_dict['class'])
        # bypasses the init function as we will use 'from_json' to initialize - like what pickle does
        obj = cls.__new__(cls)
        obj.from_json(json_dict)
        return obj

    def from_json(self, json_dict):
        self.min_x = json_dict['min_x']
        self.max_x = json_dict['max_x']
        self.min_y = json_dict['min_y']
        self.max_y = json_dict['max_y']
        self.zoom = json_dict['zoom']
        self.epsg = json_dict['epsg']
        self._version = json_dict['version']
        # do version updates here
        self._loaded_from_version = json_dict['version']

    def height(self):
        return self.max_y - self.min_y

    def width(self):
        return self.max_x - self.min_x

    def xy_to_tile_index(self, x, y, zoom=None):
        # y index is bigger than x because there is a smaller range (-90 to 90) which makes smaller tiles in the y direction
        if zoom is None:
            zoom = self.zoom
        tx = numpy.array(numpy.floor(((x - self.min_x) / (self.width() / self.num_tiles(zoom)))), dtype=numpy.int32)
        ty = numpy.array(numpy.floor(((y - self.min_y) / (self.height() / self.num_tiles(zoom)))), dtype=numpy.int32)
        return tx, ty

    def tile_index_to_xy(self, tx, ty, zoom=None):
        if zoom is None:
            zoom = self.zoom
        tile_width = self.width() / self.num_tiles(zoom)
        lx = (tx * tile_width) + self.min_x
        # ux = ((tx + 1) * tile_width) + self.min_x
        ux = lx + tile_width

        tile_height = self.height() / self.num_tiles(zoom)
        ly = (ty * tile_height) + self.min_y
        # uy = ((ty + 1) * tile_height) + self.min_y
        uy = ly + tile_height
        return lx, ly, ux, uy

    def num_tiles(self, zoom=None):
        if zoom is None:
            zoom = self.zoom
        return 2**zoom

class ExactTilingScheme(TilingScheme):
    def __init__(self, res_x, res_y,  min_x=-180, min_y=-90.0, max_x=180.0, max_y=90.0, zoom=13, epsg=None):
        super().__init__(min_x, min_y, max_x, max_y, zoom=zoom, epsg=epsg)
        self.res_x = res_x
        self.res_y = res_y
        self.edges_x, self.edges_y = self.calc_edges()
        self.max_x = self.edges_x[-1]
        self.max_y = self.edges_y[-1]

    def for_json(self):
        json_dict = super().for_json()
        json_dict['res_x'] = self.res_x
        json_dict['res_y'] = self.res_y
        json_dict['edges_x'] = self.edges_x.tolist()
        json_dict['edges_y'] = self.edges_y.tolist()
        return json_dict

    def from_json(self, json_dict):
        # restore the basic elements
        super().from_json(json_dict)
        # do version updates here if needed
        self.res_x = json_dict['res_x']
        self.res_y = json_dict['res_y']
        self.edges_x = numpy.array(json_dict['edges_x'])
        self.edges_y = numpy.array(json_dict['edges_y'])

    def xy_to_tile_index(self, x, y, zoom=None):
        # y index is bigger than x because there is a smaller range (-90 to 90) which makes smaller tiles in the y direction
        if zoom is None:
            zoom = self.zoom
        # get the tiling scheme value - if we were to ignore the strict resolution edges
        tx, ty = super().xy_to_tile_index(x, y, zoom)
        # since we round each tile DOWN to the cell edge that would be lower, check if we should use the next greater tile instead
        tx[self.edges_x[tx+1] <= x] += 1  # tx+1 is the uppder edge of the current tile == lower edge of the next tile
        ty[self.edges_y[ty+1] <= y] += 1
        return tx, ty

    def tile_index_to_xy(self, tx, ty):
        return self.edges_x[tx], self.edges_y[ty], self.edges_x[tx+1], self.edges_y[ty+1]

    def calc_edges(self, zoom=None):
        if zoom is None:
            zoom = self.zoom
        num_tiles = self.num_tiles(zoom)
        rough_xs, rough_ys, _ux, _uy = super().tile_index_to_xy(numpy.arange(num_tiles+1), numpy.arange(num_tiles+1))
        # round off the edges to the closest lower cell
        xs = rough_xs - numpy.mod(rough_xs - self.min_x, self.res_x)
        ys = rough_ys - numpy.mod(rough_ys - self.min_y, self.res_y)
        xs[-1] += self.res_x  # round the last tile up to make sure the max value is included
        ys[-1] += self.res_y
        return xs, ys

class LatLonTiles(TilingScheme):
    def __init__(self, min_x=-180, min_y=-90.0, max_x=180.0, max_y=90.0, zoom=13, epsg=None):
        """ Latitude Longitude grid, defaults the the whole world.
        Parameters
        ----------
        min_x
        max_x
        min_y
        max_y
        zoom
        """
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, zoom=zoom, epsg=epsg)

class ExactLatLonTiles(ExactTilingScheme):
    def __init__(self, res_x, resy_y, min_x=-180, min_y=-90.0, max_x=180.0, max_y=90.0, zoom=13, epsg=None):
        """ Latitude Longitude grid, defaults the the whole world.
        Parameters
        ----------
        min_x
        max_x
        min_y
        max_y
        zoom
        """
        super().__init__(res_x, res_y, min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, zoom=zoom, epsg=epsg)

class TMSTilesMercator(TilingScheme):
    """ Use the global spherical mercator projection coordinates (EPSG:900913) to match the Google Tile scheme,
    visualized at https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/
    """
    def __init__(self, zoom=13):
        circumference = 2 * numpy.pi * 6378137 / 2.0
        super().__init__(min_x=-circumference, min_y=-circumference, max_x=circumference, max_y=circumference, zoom=zoom, epsg=900913)


class GoogleTilesLatLon(LatLonTiles):
    """ Use the Lat Lon tiles like TMS but the Y indexing is reversed,
    visualized at https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/
    """
    def _flip_y(self, ty, zoom):
        return (self.num_tiles(zoom) - 1) - ty

    def xy_to_tile_index(self, x, y, zoom=None):
        tx, rev_ty = super().xy_to_tile_index(x, y, zoom)
        ty = self._flip_y(rev_ty, zoom)
        return tx, ty

    def tile_index_to_xy(self, tx, ty, zoom=None):
        return super().tile_index_to_xy(tx, self._flip_y(ty, zoom), zoom)

class GoogleTilesMercator(GoogleTilesLatLon):
    """ Use the global spherical mercator projection coordinates (EPSG:900913) to match the Google Tile scheme,
    visualized at https://www.maptiler.com/google-maps-coordinates-tile-bounds-projection/
    """
    def __init__(self, zoom=13):
        circumference = 2 * numpy.pi * 6378137 / 2.0
        super().__init__(min_x=-circumference, min_y=-circumference, max_x=circumference, max_y=circumference, zoom=zoom, epsg=900913)

    # def _flip_y(self, ty, zoom):
    #     return (self.num_tiles(zoom) - 1) - ty
    #
    # def xy_to_tile_index(self, x, y, zoom=None):
    #     tx, ty = super().xy_to_tile_index(x, self._flip_y(y, zoom), zoom)
    #     return tx, ty
    #
    # def tile_index_to_xy(self, tx, ty, zoom=None):
    #     return super().tile_index_to_xy(tx, self._flip_y(ty, zoom), zoom)

class UTMTiles(TilingScheme):
    def __init__(self, zoom=13, epsg=None):
        super().__init__(min_x=-1000000, min_y=-1000000, max_x=2000000, max_y=10000000, zoom=zoom, epsg=epsg)

class ExactUTMTiles(ExactTilingScheme):
    def __init__(self, res_x, res_y, zoom=13, epsg=None):
        super().__init__(res_x, res_y, min_x=-1000000, min_y=-1000000, max_x=2000000, max_y=10000000, zoom=zoom, epsg=epsg)


def test():
    """
    g = tile_calculations.TilingScheme(zoom=2); f=g.xy_to_tile; print(f(-100, 40)); print(f(100, -40)); print(f(182, -91))
    (0, 2)
    (3, 1)
    (4, -1)
    g.tile_to_xy(0,0, 2)
    Out[70]: (-180.0, -90.0, -90.0, -45.0)
    g.tile_to_xy(0,1, 2)
    Out[71]: (-180.0, -90.0, -45.0, 0.0)
    g.tile_to_xy(1,1, 2)
    Out[72]: (-90.0, 0.0, -45.0, 0.0)
    g.tile_to_xy(2,1, 2)
    Out[73]: (0.0, 90.0, -45.0, 0.0)
    g.tile_to_xy(numpy.array([0,1,2]),1, 2)
    Out[74]: (array([-180.,  -90.,    0.]), array([-90.,   0.,  90.]), -45.0, 0.0)
    g.tile_to_xy(numpy.array([0,1,2]),numpy.array([1,0]), 2)
    Out[75]:
    (array([-180.,  -90.,    0.]),
     array([-90.,   0.,  90.]),
     array([-45., -90.]),
     array([  0., -45.]))
    """
    pass
