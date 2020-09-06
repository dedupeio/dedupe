from abc import ABC, abstractmethod


class Index(ABC):
    @abstractmethod
    def __init__(self):  # pragma: no cover
        pass

    @abstractmethod
    def index(self, doc):  # pragma: no cover
        pass

    @abstractmethod
    def unindex(self, doc):  # pragma: no cover
        pass

    @abstractmethod  # pragma: no cover
    def search(self, doc, threshold=0):
        pass

    @abstractmethod
    def initSearch(self):  # pragma: no cover
        pass
