from utils.logging import get_logger
from datetime import datetime
from strategy_v2.Strategy.MVO import RiskModel
from pandas.tseries.offsets import BDay
import numpy as np

class ZeroCov(RiskModel):
    def __init__(self):                        
        self.logger = get_logger(self.__class__.__name__)            

    def expected_variance(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected stock covariance matrix
        '''        
        ret = self.data['px']['Return']        
        n = ret.shape[1]
        return np.zeros((n,n))        
