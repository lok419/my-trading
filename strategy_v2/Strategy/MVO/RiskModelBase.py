from abc import abstractmethod
from utils.logging import get_logger
from datetime import datetime
from pandas.core.frame import DataFrame
import numpy as np

class RiskModel(object):
    def __init__(self):                        
        self.logger = get_logger(self.__class__.__name__)
    
    def __str__(self) -> str:
        return self.__class__.__name__

    def preprocess_data(self, data:dict[DataFrame]):
        self.data = data.copy()
        self.instruments = self.data['px']['Close'].columns                         

    @abstractmethod
    def expected_variance(self, pos_date: datetime) -> np.ndarray:
        pass
