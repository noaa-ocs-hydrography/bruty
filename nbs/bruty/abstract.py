from abc import ABC, abstractmethod

class VABC(ABC):
    """ A base class that supplies an id and version attribute
    """
    @property
    def id(self):
        return self._id

    @property
    def __version__(self) -> int:
        return self._version
