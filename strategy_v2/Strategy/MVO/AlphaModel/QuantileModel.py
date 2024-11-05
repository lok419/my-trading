import random
from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel
from utils.data_helper import add_bday


class QuantileModel(AlphaModel):      
    '''
        1. We want to model the next 10days return (rebalance frequency)

        2. Bootstrap the data (n=10) into a new distribution, takes quantile q, and we argue that q% chance the return samller than that

        3. Adjust q for the aggressiveness of the alpha model

        4. This assume daily return has no serial correlation, but likely incorrect
    '''
    def __init__(self, lookback=60, q=0.5, k=10, n=1000):   
        '''
            Args:
                Lookback: number of days to lookback
                k: number of data to bootstrap into one sample
                n: number of data in bootstrap samples
                q: quantile
        '''  
        assert q > 0 and q < 1, "q should be between 0 and 1"
        self.lookback = lookback
        self.q = q
        self.k = k
        self.n = n

    def __str__(self) -> str:
        return f"Quantile{self.q}"
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:

        lookback_end = add_bday(pos_date, -1)
        expected_ret = []

        df = self.data['px']            
        df = df[:lookback_end]   
        df = df.tail(self.lookback)              
        assert max(df.index) < pos_date, 'Optimization has lookahead bias'             

        for i in self.instruments:            
            rets = list(np.log(1+df['Return'][i].values))
            vols = list(df['Volume'][i].values)
            rets_bs = [np.mean(random.choices(rets, weights=vols, k=self.k)) for _ in range(self.n)]
            ret = np.quantile(rets_bs, self.q)
            ret = np.exp(ret)-1            
            expected_ret.append(ret)
            
        return np.array(expected_ret)
    