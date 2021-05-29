from nbs_utils.points_utils import NBSFormatError

class BrutyError(Exception):
    pass


class BrutyFormatError(BrutyError, NBSFormatError):
    pass


class BrutyMissingScoreError(BrutyError):
    pass

class BrutyUnkownCRS(BrutyError):
    pass

class UserCancelled(BrutyError):
    pass
