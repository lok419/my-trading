import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from pandas.tseries.offsets import BDay
import numpy as np
from tqdm import tqdm
from datetime import datetime
import cvxpy as cp
from strategy_v2.Strategy.MVO import AlphaModel, RiskModel
from strategy_v2.Strategy.StrategyBase import StrategyBase
from strategy_v2.Strategy.MVO.AlphaModel import ZeroAlpha
from strategy_v2.Strategy.MVO.RiskModel import ZeroCov
from utils.data import get_yahoo_data_formatted
from utils.data_helper import add_bday, is_ny_trading_hours, get_today

class MeanVarianceOpt(StrategyBase):      

    def __init__(self, 
                 alpha_model:AlphaModel=ZeroAlpha(),
                 risk_model:RiskModel=ZeroCov(),
                 confidence:float=1, 
                 opt_freq:int=1, 
                 gamma:float=0.01,
                 hhi:float=0.2,
                 leverage:float=-1  
                ):                
        '''
            alpha_model: alpha signals
            risk_model:  covariance matrix of the instruments
            confidence:  confidence of the mean-variance optimization. the optimized results are scaled by confidence
            opt_freq:    optimization frequency
            gamma:       risk aversion
            hhi:         Herfindahl-Hirschman Index
            leverage:    Leverage, determine the optimization constraint
        '''

        self.alpha_model = alpha_model
        self.risk_model = risk_model        
        self.confidence = confidence
        self.opt_freq = opt_freq
        self.gamma = gamma
        self.hhi = hhi
        self.leverage = leverage
        if self.hhi > 0:
            assert self.leverage == 1, 'leverage must be 1 if hhi > 0'

        # store the historcal signals
        self.alphas = []
        self.risks = []

        super().__init__()

    def __str__(self) -> str:
        return f'MVO - {self.alpha_model}|{self.confidence}'   

    def load_data(self, lookback=100):
        '''
            In case the data wasn't passed
        '''        
        if not hasattr(self, 'data') or 'px' not in self.data:
            data_start = add_bday(self.start_date, -lookback)   
            data_end = self.end_date
            self.data['px'] = get_yahoo_data_formatted(self.instruments, data_start, data_end)

    def preprocess_data(self):
        '''
            Add return columns to data
        '''
        ret = self.data['px']['Close'].pct_change().fillna(0)        
        ret.columns = pd.MultiIndex.from_tuples([('Return', c) for c in ret.columns])
        self.data['px'] = pd.concat([self.data['px'], ret], axis=1)               
        self.alpha_model.preprocess_data(self.data) 
        self.risk_model.preprocess_data(self.data)

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

            # generates position as of SOD of "date", so you can only access to data prior to "date"
            for date in dates_arr:   
                                
                # If T is today, include latest data T
                # Normally when we backtest at T, we should use data prior to T to avoid lookahead bias
                # However, if T is today and we want to generate a position for T, it is okay to include T data 
                model_date = date
                if date == get_today():
                    model_date = date + BDay(1)
                    self.logger.info(f"{date:%Y-%m-%d} is today, shift position date to include latest market data at {datetime.now():%I:%m %p}")

                expected_ret = self.alpha_model.expected_return(model_date)
                expected_ret_cov = self.risk_model.expected_variance(model_date)

                self.alphas.append(expected_ret)
                self.risks.append(expected_ret_cov)

                w = cp.Variable(len(self.instruments))                    
                gamma_par = cp.Parameter(nonneg=True)

                # optimal risk-return trade-off                
                gamma_par.value = self.gamma            

                # we use year as basis
                ret = 252 * expected_ret.T @ w
                risk = 252 * cp.quad_form(w, expected_ret_cov)

                constraints = []                
                constraints.append(w <= 1)
                constraints.append(w >= 0)

                if self.hhi > 0:
                    constraints.append(cp.sum_squares(w) <= self.hhi)

                if self.leverage > 0:
                    constraints.append(cp.sum(w) == self.leverage)
                else:
                    constraints.append(cp.sum(w) <= 1)
                
                prob = cp.Problem(cp.Maximize(ret - gamma_par * risk), constraints)                            
                prob.solve(solver='ECOS')    

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
        self.alphas = pd.DataFrame(self.alphas, columns=close.columns, index=dates_arr)

    def get_position(self) -> DataFrame:
        return self.position