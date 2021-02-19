from abc import ABC, abstractmethod

class VABC(ABC):
    @property
    def id(self):
        return self._id

    @property
    def __version__(self) -> int:
        return self._version
