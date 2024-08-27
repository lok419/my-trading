import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from pandas.tseries.offsets import BDay
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cvxpy as cp
from utils.data import get_yahoo_data_formatted
from utils.data_helper import add_bday
from .StrategyBase import StrategyBase

class MeanVarianceOpt(StrategyBase):      

    def __init__(self, 
                 confidence:float=1, 
                 opt_freq:int =1, 
                 gamma:float=0.01,
                 hhi:float=0.2,
                 lookback:int=60
                ):        
        if confidence > 2 or confidence < -2:
            raise('confidence has to between -2 and 2')
                
        self.confidence = confidence
        self.opt_freq = opt_freq
        self.gamma = gamma
        self.hhi = hhi
        self.lookback = lookback

        super().__init__()

    def __str__(self) -> str:
        return f'MVO - SMA{self.confidence}'
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
        '''
        ret = self.data['px']['Return']        
        lookback_end = pos_date - BDay(1)  
        
        # return within lookback periods - MAKE SURE NOT LOOK AHEAD BIAS HERE
        ret = ret[:lookback_end]
        ret = ret.tail(self.lookback)               

        assert max(ret.index) < pos_date, 'Optimization has lookahead bias'
        ret = ret.values
        expected_ret = ret.mean(axis=0)
        return expected_ret

    def expected_variance(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected stock covariance matrix
        '''
        ret = self.data['px']['Return']        
        lookback_end = pos_date - BDay(1)  

        # return within lookback periods - MAKE SURE NOT LOOK AHEAD BIAS HERE
        ret = ret[:lookback_end]
        ret = ret.tail(self.lookback)        
        assert max(ret.index) < pos_date, 'Optimization has lookahead bias'
        ret = ret.values
        ret_cov = np.cov(ret.T)         
        return ret_cov        

    def load_data(self):
        '''
            In case the data wasn't passed
        '''        
        if not hasattr(self, 'data') or 'px' not in self.data:
            data_start = add_bday(self.start_date, -self.lookback)   
            data_end = self.end_date
            self.data['px'] = get_yahoo_data_formatted(self.instruments, data_start, data_end)

    def preprocess_data(self):
        '''
            Add return columns to data
        '''
        ret = self.data['px']['Close'].pct_change().fillna(0)        
        ret.columns = pd.MultiIndex.from_tuples([('Return', c) for c in ret.columns])
        self.data['px'] = pd.concat([self.data['px'], ret], axis=1)        

    def generate_position(self):                         
        '''
            Mean Variance Optimization
            Maximize 
                ret - gamma * risk 
            Subject to constraints:
                1. sum of weight = 1
                2. HHI <= x
                3. all weights smaller than 1 (no leverage)
                4. all weights greater than 0 (no short)
        '''
        close = self.data['px']['Close']
        self.position = pd.DataFrame(index=close.index, columns=close.columns).fillna(1/len(self.instruments))

        # all historical dates        
        date = self.start_date
        dates_arr = []     
        self.opt_ret = []
        self.opt_risk = []

        while date <= self.end_date:            
            dates_arr.append(date)
            date += BDay(self.opt_freq)

        last_weight = np.ones(len(self.instruments)) / len(self.instruments)        

        with tqdm(total=len(dates_arr)) as pbar:

            # generates position as of SOD of "date", so you only access to data prior to "date"
            for date in dates_arr:        

                expected_ret = self.expected_return(date)
                expected_ret_cov = self.expected_variance(date)                                

                w = cp.Variable(len(self.instruments))                    
                gamma_par = cp.Parameter(nonneg=True)

                # optimal risk-return trade-off                
                gamma_par.value = self.gamma            

                # we use year as basis
                ret = 252 * expected_ret.T @ w
                risk = 252 * cp.quad_form(w, expected_ret_cov)                    

                constraints = [                    
                    cp.sum(w) == 1,               
                    cp.sum_squares(w) <= self.hhi,                    
                    w <= 1,
                    w >= 0,
                ]
                prob = cp.Problem(cp.Maximize(ret - gamma_par * risk), constraints)                            
                prob.solve()    

                weight = w.value

                if weight is None:
                    weight = last_weight
                    self.logger.error('{} - Failed to find optimal weights: use weight from last periods'.format(date.strftime('%Y-%m-%d')))
                else:
                    weight = np.round(weight, 3)
                    last_weight = weight

                self.opt_ret.append(ret.value)
                self.opt_risk.append(risk.value)

                # Assign the weight
                next = date + BDay(self.opt_freq-1)
                self.position.loc[date:next] = weight                         
                pbar.update(1)

        self.position = self.position.loc[self.start_date: self.end_date]
        self.position *= self.confidence

    def get_position(self) -> DataFrame:
        return self.position