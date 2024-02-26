from pandas.core.api import DataFrame as DataFrame, Series as Series
from strategy_v2.TradingSubSystem import TradingSubSystemBase
from strategy_v2.Portfolio import PortfolioBase
from pandas.tseries.offsets import BDay
from tqdm import tqdm
from utils.data import get_risk_free_rate
from plotly.subplots import make_subplots
from utils.performance import performance_summary_table
import plotly.graph_objects as go
import cvxpy as cp
import numpy as np
import pandas as pd

class PortfolioLinearOpt(PortfolioBase):
    '''
        Linear Optimized Portfolio        
        Linear Optimization is done such that it optimize the Portfolio Sharp while maintain the volatility target
    ''' 

    def __init__(self,                 
                 opt_freq=20,
                 lookback_period=180,
                 *args,
                 **kwargs,                              
        ):
        '''
            opt_freq:           Frequency of the optimization
            lookback_period:    Lookback period when compute the expected returns and variance, how long does it look back.
        '''
        super().__init__(offset=lookback_period+5, *args, **kwargs)        
        assert lookback_period>=2, 'Lookback Period must be >= 2'
        self.opt_freq = opt_freq        
        self.lookback_period = lookback_period

    def __str__(self):
        return "Linear Optimized Portfolio" + (f' ({self.name})' if len(self.name) else '')
    
    def optimize(self, gamma:float=30, hhi:float=0.2):    
        '''
            Linear Optimization

            To get the historical returns, we use "self.ret_raw" which has historical returns with extended lookback periods before backtest start date  
            So even the first day we can still compute LVO from the whole lookback periods.
        '''
        # run backtest_subsystems before optimization
        if len(self.position) == 0 and len(self.ret) == 0:
            self.backtest_subsystems()

        # initilize the weights by equal weights
        self.port_w = pd.DataFrame(index=self.ret.index, columns=self.ret.columns).fillna(1/len(self.systems))

        # all historical dates
        dates = self.ret.index           
        date = dates[0]

        dates_arr = []     
        self.opt_ret = []
        self.opt_risk = []

        while date <= self.end_date:            
            dates_arr.append(date)
            date += BDay(self.opt_freq)

        last_weight = np.ones(len(self.systems)) / len(self.systems)

        with tqdm(total=len(dates_arr)) as pbar:
            for date in dates_arr:        

                # return within lookback periods - MAKE SURE NOT LOOK AHEAD BIAS HERE
                lookback_end = date - BDay(1)
                ret_mat = self.ret_raw[:lookback_end]                
                ret_mat = ret_mat.tail(self.lookback_period)
                assert max(ret_mat.index) < date, 'Optimization has lookahead bias'
                ret_mat = ret_mat.values
            
                # expected return by SMA (alternative could be EMA)
                expected_ret = ret_mat.mean(axis=0)

                # returns covariance
                ret_cov = np.cov(ret_mat.T)                

                w = cp.Variable(len(self.systems))                    
                gamma_par = cp.Parameter(nonneg=True)

                # optimal risk-return trade-off                
                gamma_par.value = gamma            

                # we use year as basis
                ret = 252 * expected_ret.T @ w
                risk = 252 * cp.quad_form(w, ret_cov)                    

                constraints = [                    
                    cp.sum(w) == 1,               
                    cp.sum_squares(w) <= hhi,                    
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
                self.port_w.loc[date:next] = weight                            
                pbar.update(1)

    def calibrate_gamma(self, base_start=1, base_end=2.5, num=10):
        '''
            Function to find the best optimal risk-return trade-off
        '''

        gamma_vals = np.logspace(base_start, base_end, num=num)
        self.calib_table = []

        for gamma in gamma_vals:               
            self.optimize(gamma=gamma)
            self.backtest()
            ret = pd.DataFrame(data=self.port_ret, columns=[str(gamma)])
            table = performance_summary_table(ret)
            self.calib_table.append(table)            

        gamma = [t.columns[0] for t in self.calib_table]
        ret = np.array([t.loc['Annualized Return'][0] for t in self.calib_table])
        vol = np.array([t.loc['Annualized Volatility'][0] for t in self.calib_table])
        sr = np.array([t.loc['Annualized Sharpe Ratio'][0] for t in self.calib_table])

        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            width=500, height=500, 
            title='Gamma Calibration',
            hovermode='x',
        )

        gamma = ['Gamma = {:.2f}'.format(float(g)) for g in gamma]
        sr = ['Sharpe Ratio = {:.2f}'.format(float(g)) for g in sr]
        text = zip(gamma, sr)
        text = ['<br>'.join(t) for t in text]

        fig.add_trace(go.Scatter(x=ret*100, y=vol*100,  hovertext=text), row=1, col=1)
        fig.update_traces(textposition='top center')
        fig['layout']['xaxis']['title']= 'Annualized Return (%)'
        fig['layout']['yaxis']['title']= 'Annualized Vol (%)'
        fig.show()

    def calibrate_hhi(self):
        '''
            Function to find portfolio concentration
        '''
        vals = np.linspace(0.1, 1, num=10)
        self.calib_table = []

        for val in vals:               
            self.optimize(hhi=val)
            self.backtest()
            ret = pd.DataFrame(data=self.port_ret, columns=[str(val)])
            table = performance_summary_table(ret)
            self.calib_table.append(table)            

        hhi = [t.columns[0] for t in self.calib_table]
        ret = np.array([t.loc['Annualized Return'][0] for t in self.calib_table])
        vol = np.array([t.loc['Annualized Volatility'][0] for t in self.calib_table])
        sr = np.array([t.loc['Annualized Sharpe Ratio'][0] for t in self.calib_table])

        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(
            width=500, height=500, 
            title='HHI Constraint Calibration',
            hovermode='x',
        )

        hhi = ['hhi = {:.2f}'.format(float(g)) for g in hhi]
        sr = ['Sharpe Ratio = {:.2f}'.format(float(g)) for g in sr]
        text = zip(hhi, sr)
        text = ['<br>'.join(t) for t in text]

        fig.add_trace(go.Scatter(x=ret*100, y=vol*100,  hovertext=text), row=1, col=1)
        fig.update_traces(textposition='top center')
        fig['layout']['xaxis']['title']= 'Annualized Return (%)'
        fig['layout']['yaxis']['title']= 'Annualized Vol (%)'
        fig.show()












            

            

            

            