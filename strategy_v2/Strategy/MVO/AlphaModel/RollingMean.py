from utils.logging import get_logger
from datetime import datetime
from strategy_v2.Strategy.MVO.AlphaModelBase import AlphaModel
import numpy as np

class RollingMean(AlphaModel):
    def __init__(self, lookback=30):                        
        self.logger = get_logger(self.__class__.__name__)    
        self.lookback = lookback

    def __str__(self):
        return f'{super().__str__()}({self.lookback})'

    def preprocess_data(self, data):
        super().preprocess_data(data)
        '''
            Compute rolling returns as T's trading signals in advance
        '''

        ret = self.data['px']['Return']    

        # T's position is based on T-1 return and before
        df_ret = ret.shift(1)
        df_ret = df_ret.rolling(self.lookback).mean()
        self.df_ret = df_ret        
        print("test")

    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
        '''        
        expected_ret = np.array(self.df_ret.loc[:pos_date].iloc[-1])

        return expected_ret
