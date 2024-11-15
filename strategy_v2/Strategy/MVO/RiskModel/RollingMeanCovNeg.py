from utils.logging import get_logger
from datetime import datetime
from strategy_v2.Strategy.MVO import RiskModel
from pandas.tseries.offsets import BDay
import numpy as np

class RollingMeanCovNeg(RiskModel):
    def __init__(self, lookback=30):                        
        self.logger = get_logger(self.__class__.__name__)    
        self.lookback = lookback

    def expected_variance(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected stock covariance matrix, only filters for negative returns
            as whole point of risk management is to avoid covariance when every stocks are going down
        '''        
        ret = self.data['px']['Return']        
        lookback_end = pos_date - BDay(1)  

        # return within lookback periods - MAKE SURE NOT LOOK AHEAD BIAS HERE
        ret = ret[:lookback_end]
        ret = ret.tail(self.lookback)                
        assert max(ret.index) < pos_date, 'Optimization has lookahead bias'
        ret = ret.values
        ret = np.minimum(ret, -1)
        ret_cov = np.cov(ret.T)         
        return ret_cov   
