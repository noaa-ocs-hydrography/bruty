from nbs.bruty.abstract import VABC, abstractmethod

class SurveyHistory(VABC):
    def __init__(self, survey_id):
        pass

    @property
    def tiles_affected(self):
        pass
    def deltas_created(self):
        pass
    def undo(self):
        pass
    def redo(self):
        pass