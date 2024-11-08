from utils.logging import get_logger
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel
from pandas.tseries.offsets import BDay
import numpy as np

class ZeroAlpha(AlphaModel):
    def __init__(self):                        
        self.logger = get_logger(self.__class__.__name__)            

    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
        '''
        ret = self.data['px']['Return']        
        n = ret.shape[1]
        return np.zeros(n)