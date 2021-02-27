from bruty.abstract import VABC, abstractmethod

class Tile(VABC):
    """ Class to encapulate a raster that has a history.
    It should be able to get the current raster data and also at a previoud point in time.
    Initial thought is to have two sets of history, one based on insertion date and one based on survey date.
    Think of it as a VCS like GIT or SVN.
    Each transaction will be held as a double linked list of deltas between it and the next more recent version of the surface.

    The current raster is correct for both insertion and survey date histories.
    When a survey is inserted the most recent insertion is changed and the current raster updated.
    The survey date history is harder,
    the location of the survey in the linked list must be determined and then the entire more recent history must be changed.
    This is similar to if a survey needs to be revised (due to data quality/reprocessing) and the insertion deltas revised.
    Think of it like a rebase in GIT.
    """
    def __init__(self, id, raster_storage, history_storage):
        self.id = id
        self._version = 1
        self.raster_storage = storage
        self.history_storage = history_storage

    def extract(self, area=None):
        return extract_at_commit(area)
    def extract_at_date(self, area, date):
        pass
    def extract_at_commit(self, area=None, commit=None):
        pass
    def reproject(self):
        pass
    def get_history(self):
        pass
    @property
    def location(self):
        pass
    @property
    def resolutions(self):
        pass
    @property
    def writeable(self):
        return False

class WritableTile(Tile):
    def insert_raster(self, raster, survey_date):
        # create a new set of files, if needed, so the transaction can be rolled back
        pass

    def save(self):
        self.storage.save_tile(self.id, self)

    def cleanup(self):
        # delete old delta files if there is no longer a need for rollback
        pass

    @property
    def writeable(self):
        return True
