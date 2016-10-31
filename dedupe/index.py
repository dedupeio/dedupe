from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod

class Index(with_metaclass(ABCMeta)):
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def index(self, doc):
        pass

    @abstractmethod
    def unindex(self, doc):
        pass

    @abstractmethod
    def search(self, doc, threshold=0):
        pass

    @abstractmethod
    def initSearch():
        pass
    
