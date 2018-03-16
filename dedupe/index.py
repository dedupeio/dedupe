from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod


class Index(with_metaclass(ABCMeta)):
    @abstractmethod
    def __init__():  # pragma: no cover
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
    def initSearch():  # pragma: no cover
        pass
