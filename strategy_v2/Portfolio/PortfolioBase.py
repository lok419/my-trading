from datetime import datetime
import math
from strategy_v2.TradingSubSystem import TradingSubSystemBase
from strategy_v2.TransactionCost import TransactionCostModel
from strategy_v2.Portfolio import Performance, PortfolioModel, RebalancerIter
from abc import abstractmethod
from pandas.tseries.offsets import BDay
from utils.logging import get_logger

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class PortfolioBase(PortfolioModel, Performance):
    '''
        Base class of Portfolios
        A Portfolio should take backtest from subsystems and create an optimized portfolios
        Each of the output from subsystem is treated as a single instrument. 

        Some general definition in position or weights:
        - Position      @ T => position being held from T-1 close to T close. Of course, the position @ T should be computed using T-1 data, but not T
                               e.g. this portfolio runs and generates position at T open using T-1 close price

        - Position Shs  @ T => Same as above. But we need to use the instrument price from T-1 and capital from T-1 to derive it
        - Capital       @ T => portfolio capital at T close. So we should include the return @ T
    '''    
    def __init__(self,
                 name:str='',
                 systems: list[TradingSubSystemBase]=[],
                 capital:float=10000,
                 offset:float=1,
                 rebalance_freq:float=20,
                 rebalance_iter:RebalancerIter=None,                 
                 rebalance_inertia:float=0.1,
                 tc_model:TransactionCostModel=None,
        ):

        '''
            systems:            list of tradingsubsystem
            captial:            capital allcoated to the whole portfolio
            offset:             extended periods before backtest start date to compute
            rebalance_freq:     rebalance frequence (this is probably used in backtest)
            rebalance_iter:     rebalance iterator where you can specify the actual rebalance periods (e.g. every 2 Friday)
            rebalance_inertia:  rebalance inertia, specifty the minimim size to rebalance
            tc_model            define the transaction mdoel
        '''
        self.name = name
        self.systems = systems
        self.capital = capital
        self.offset = offset
        self.rebalance_freq = rebalance_freq
        self.rebalance_iter = rebalance_iter
        self.rebalance_inertia = rebalance_inertia

        # backtest generated position from underlying sub-systems
        self.position = pd.DataFrame()

        # backtest return from underlying sub-systems
        self.ret = pd.DataFrame()
        self.ret_raw = pd.DataFrame()

        # Portfolio Weight by sub systems (this is what your child class should optimize)
        # By definition, port_w @ T is the weight held from T-1 close to T close. So this should be optimized using all data before T but not on T to avoid look ahead bias
        self.port_w = pd.DataFrame()

        # portfolio position and return after optimization
        self.port_position = pd.DataFrame()   
        self.port_position_shs = pd.DataFrame() 
        self.port_position_dp = pd.DataFrame()
        self.port_capital = pd.DataFrame()
        self.port_ret = pd.DataFrame()        

        # portfolio position adn return after rebalance
        self.port_position_rebal = pd.DataFrame()   
        self.port_position_shs_rebal = pd.DataFrame() 
        self.port_position_dp_rebal = pd.DataFrame()
        self.port_capital_rebal = pd.DataFrame()
        self.port_ret_rebal = pd.DataFrame()    

        # strike position means the position and capital will be based on last rebalance rather affected by MTM change
        self.port_position_rebal_strike = pd.DataFrame()
        self.port_position_dp_rebal_strike = pd.DataFrame()

        # close price for each instrument so that we can compute the position in shares
        self.close_px = pd.DataFrame()
        self.close_px_raw = pd.DataFrame()    

        # Transaction Cost Model
        self.tc_model = tc_model
        self.port_tc = pd.Series()
        self.port_tc_rebal = pd.Series()
        self.logger = get_logger(self.__str__())   

    def __str__(self):        
        return self.__class__.__name__ + (f' ({self.name})' if len(self.name) else '')

    def set_start_date(self, start_date: datetime):
        self.start_date = start_date
        return self
    
    def set_end_date(self, end_date: datetime):
        self.end_date = end_date
        return self
    
    def backtest_subsystems(self):
        '''
            Run the backtests for all subsystems individually.
            Each subsystems as if they trade 100% of capital
            Also the function cominbes all px data from sub-systems            
        '''       

        for i, system in enumerate(self.systems):        
            system.set_start_date(self.start_date - BDay(self.offset))
            system.set_end_date(self.end_date)

            # generate all positions per strategy
            system.backtest()

            # combine positions from strategy and volatility targeting
            system.optimize()

            pos = system.get_position()
            pos.columns = pd.MultiIndex.from_tuples([(str(system), c) for c in pos.columns])
            ret = system.get_return()            
            
            if i == 0:
                self.position = pos
                self.ret = pd.DataFrame(ret, columns=[str(system)])
            else:
                self.position = pd.merge(self.position, pos, how='outer', on=['Date'], validate='1:1')
                self.ret[str(system)] = ret

            # Combine all price data to compute the shares data                            
            close_px = system.get_data()['px']['Close']
            if len(self.close_px) == 0:
                self.close_px = close_px
            else:
                for col in close_px.columns:
                    if col not in self.close_px:
                        self.close_px[col] = close_px[col]

        # fill all null to zeros
        self.position = self.position.fillna(0)
        self.ret = self.ret.fillna(0)
        self.ret_raw = self.ret.copy()

        # forward fill in case any missing price
        self.close_px = self.close_px.ffill()

        # restruct position and ret based on start date and end date      
        self.close_px_raw = self.close_px.copy()
        self.close_px = self.close_px.loc[self.start_date: self.end_date]  
        self.position = self.position.loc[self.start_date: self.end_date]
        self.ret = self.ret.loc[self.start_date: self.end_date]

    def backtest(self):
        '''
            Generate position and return based on optimized weights among different sub-system
            In backtest, position are the controlled variables (fixed) and position_shs depends on position
            
            self.position           = pre-optimized weights per stock among different sub-system (multi-indexed)
            self.ret                = backtest return among different subsystem
            self.port_w             = optimzied weights among different sub-system
            self.port_position      = optimzied weights among differnt stocks, deriving from self.port_w
            self.port_position_shs  = optimzied shares among different stocks, deriving from self.port_w
        '''        

        # FIXME: need to make sure the self.port_position are in same orders as self.close_px, now this is implied by the orders of sub-systems
        for i, system in enumerate(self.systems):
            pos = self.position[str(system)].mul(self.port_w[str(system)], axis=0)                                            
            if i == 0:
                self.port_position = pos
            else:
                for col in pos.columns:
                    if col in self.port_position:
                        self.port_position[col] += pos[col]                        
                    else:
                        self.port_position = pd.merge(self.port_position, pos, how='outer', on=['Date'], validate='1:1')
                                         
        '''
            Three ways to calculate theoretical portfolio returns
                1. use sub system returns times the optimized weights
                2. use actual position weight times instrument percentage returns
                3. use actual position shares times instrument dollar return
            All above should yield the same daily returns, just in case, we want to assert (1) == (2) == (3)
        '''        

        # (1) sub system return * optimized weights
        self.port_ret = (self.ret * self.port_w).sum(axis=1)        

        # (2) actual position * instrument return %
        close_ret = self.close_px_raw.pct_change().fillna(0)
        close_ret = close_ret.loc[self.start_date:self.end_date]
        port_ret2 = (self.port_position * close_ret).sum(axis=1)

        assert(np.abs(port_ret2 - self.port_ret) >= 1e-7).sum() == 0, 'portfolio returns has discrepancy'

        # fill all null to zeros
        self.port_position = self.port_position.fillna(0)        
        self.port_ret = self.port_ret.fillna(0)
        self.port_capital = self.capital * (1+self.port_ret).cumprod()        

        # find position in capital and shares                
        # By definition, position @ T means the position held from T-1 close to T close, 
        # So in order to calculate the number of shares, we need to use T-1 close price and capital
        capital_lag = self.port_capital.shift(1).fillna(self.capital)
        close_px_lag = self.close_px_raw.shift(1).loc[self.start_date:self.end_date]

        self.port_position_shs = self.port_position.multiply(capital_lag, axis=0) / close_px_lag
        self.port_position_dp = self.port_position_shs * self.close_px

        # (3) actual position in shares * instrument return $    
        close_ret_dp = self.close_px_raw.diff().loc[self.start_date:self.end_date]
        ret_cum_dp = (self.port_position_shs * close_ret_dp).sum(axis=1).cumsum()

        # calculate pct change of portfolio dollar price, this should match with (1) and (2)
        # pct_change() is missing the first day return, so need to assign it manually        
        port_ret3 = (self.capital + ret_cum_dp).pct_change().fillna(0)
        port_ret3.loc[self.start_date] = ret_cum_dp.iloc[0] / self.capital

        assert (np.abs(port_ret3 - self.port_ret) >= 1e-7).sum() == 0, 'portfolio returns has discrepancy'

        # add transaction cost
        if self.tc_model is not None:
            port_tc = self.tc_model.compute(self.port_position_shs, self.close_px)
            port_rc_cum = port_tc.cumsum()

            self.port_capital = (self.capital + ret_cum_dp - port_rc_cum)      

            port_ret4 = self.port_capital.pct_change().fillna(0)            
            port_ret4.loc[self.start_date] = (ret_cum_dp - port_tc).iloc[0] / self.capital

            self.port_ret = port_ret4                        
            self.port_tc = port_tc

    @abstractmethod
    def optimize(self):
        pass
    
    def get_weights(self) -> pd.DataFrame:
        '''
            Return the backtest Weights of differnet sub-system
        '''
        return self.port_w        
    
    def get_position(self) -> pd.DataFrame:
        '''
            Return the final backtest positions of portfolio in weights after optimization            
        '''
        return self.port_position_rebal
    
    def get_position_shs(self) -> pd.DataFrame:
        '''
            Return the final backtest positions of portfolios in shares after optimization            
        '''
        return self.port_position_shs_rebal
    
    def get_return(self) -> pd.Series:
        '''
            Return the final backtest returns of portfolios after optimization
        '''
        return self.port_ret_rebal
    
    def rebalance(self):
        '''
            Regenerate the position data based on rebalance rule and inertia
            We keep the position shares freezed between rebalance days
            Therefore, we start with position shares and derive back the weights and capital (*****this is the opposite of what we did in backtest()********)

            In rebalance, the position_shs are the controlled variables (fixed), where the position depends on position_shs
        '''        
        self.port_position_shs_rebal = self.port_position_shs.copy()        
        self.port_position_rebal_strike = self.port_position.copy()
        dates = self.port_position_shs_rebal.index
        date = dates[0]

        # use rebalance iterator
        if self.rebalance_iter is not None:
            self.rebalance_iter.set_base_date(date)

        while date <= self.end_date:
            next_date = self.rebalance_iter.get_next() if self.rebalance_iter is not None else date + BDay(self.rebalance_freq)
            freeze_start = date + BDay(1)
            freeze_end = next_date - BDay(1)
            self.port_position_shs_rebal.loc[freeze_start: freeze_end] = np.nan
            self.port_position_rebal_strike.loc[freeze_start: freeze_end] = np.nan
            date = next_date

        # this is the rebalanced position in dollar price (using T-1 Close) (excl. MTM)
        # recalls "Shares = Weights * T-1 Captial / T-1 Close", if we want to get the portfolio value, need to use T-1 Close           
        close_px_lag = self.close_px_raw.shift(1).loc[self.start_date:self.end_date]
        self.port_position_dp_rebal_strike = self.port_position_shs_rebal * close_px_lag
        self.port_position_dp_rebal_strike = self.port_position_dp_rebal_strike.ffill()   
    
        # this is the rebalanced weights (excl. MTM)
        self.port_position_rebal_strike = self.port_position_rebal_strike.ffill()

        # this is the rebalanced position in shares
        # round the shares to integer
        self.port_position_shs_rebal = self.port_position_shs_rebal.ffill()           
        self.port_position_shs_rebal = self.port_position_shs_rebal.round()

        # Portfolio value
        self.port_position_dp_rebal = self.port_position_shs_rebal * self.close_px

        # Regenerate the returns using rebalanced shares position
        close_ret_dp = self.close_px_raw.diff().loc[self.start_date:self.end_date]
        ret_cum_dp = (self.port_position_shs_rebal * close_ret_dp).sum(axis=1).cumsum()

        self.port_capital_rebal = self.capital + ret_cum_dp        
        self.port_ret_rebal = (self.port_capital_rebal).pct_change().fillna(0)
        self.port_ret_rebal.loc[self.start_date] = ret_cum_dp.iloc[0] / self.capital

        # add transaction cost
        if self.tc_model is not None:
            port_tc = self.tc_model.compute(self.port_position_shs_rebal, self.close_px)
            port_rc_cum = port_tc.cumsum()

            self.port_capital_rebal = (self.capital + ret_cum_dp - port_rc_cum)
            port_ret3 = self.port_capital_rebal.pct_change().fillna(0)            
            port_ret3.loc[self.start_date] = (ret_cum_dp - port_tc).iloc[0] / self.capital

            self.port_ret_rebal = port_ret3
            self.port_tc_rebal = port_tc
            
    def get_position_for_trade(
            self, 
            target_capital:float= float('nan')
        ) -> pd.DataFrame:
        '''
            Get the actual position for trade based on target capital and output is used by Executor. 
            Target captial should also be an input from Executor.

            Reason of we need to use this function rather than get_position_shs() is:
                get_position_shs() is based on backtest returns, and the shares are based on backtest capital, not real target portfolio captial
                e.g.
                    We we define capital as $100, the backtest cum return as of latest is 150% => capital are now $150. 
                    Therefore, the shares are now based on $150 instead of $100.            

            Scale logic:
                1. We need to know the backtest leverage when we rebalanced
                2. We need to know the backtest portfolio value for when we rebalanced
                3. Scale the position in shares by 
                    target capital / (backtest portfolio value / backtest leverage)
                    portfolio value / leverage implies the captial
        '''

        if math.isnan(target_capital):
            self.logger.critical(f'Portfolio target capital is not specified, use initial backtest capital of ${self.capital:,.0f}')
            target_capital = self.capital

        self.logger.info(f'Generate trade position based on target capital of ${target_capital:,.0f}')

        position_shs = self.get_position_shs()        
        lev = self.port_position_rebal_strike.sum(axis=1)
        pos_dp = self.port_position_dp_rebal_strike.sum(axis=1)
        scale_ts = target_capital / (pos_dp / lev)

        position_shs_trade = position_shs.multiply(scale_ts, axis=0)
        position_shs_trade = position_shs_trade.round()

        return position_shs_trade




        


        
        

        

