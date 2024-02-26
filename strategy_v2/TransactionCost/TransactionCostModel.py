from abc import ABC, abstractmethod
from pandas import DataFrame
from pandas.core.api import Series as Series

class TransactionCostModel(ABC):

    @abstractmethod
    def compute():
        pass

