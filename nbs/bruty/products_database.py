from nbs.bruty.abstract import VABC, abstractmethod

class ProductDatabase(VABC):
    def __init__(self, world_data):
        pass
    def export(self, area, crs, parameters):
        pass
    def export_ocm(self):
        pass
    def export_mcd(self):
        pass
    def manifest_email(self):
        pass
    def export_soundings(self):
        pass
    def export_contours(self):
        pass


