class BrutyError(Exception):
    pass


class BrutyFormatError(BrutyError):
    pass


class BrutyMissingScoreError(BrutyError):
    pass

class BrutyUnkownCRS(BrutyError):
    pass

