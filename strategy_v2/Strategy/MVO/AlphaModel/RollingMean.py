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
        df_ret = self.data['px']['Return']                    
        df_ret = df_ret.rolling(self.lookback).mean()
        self.df_ret = df_ret                

    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
        '''        
        # select row with date just before pos_date
        expected_ret = self.df_ret.loc[self.df_ret.index[self.df_ret.index < pos_date][-1]]
        assert expected_ret.name < pos_date, 'Optimization has lookahead bias'
        expected_ret = np.array(expected_ret)

        return expected_ret
