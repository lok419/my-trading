import pandas as pd
from datetime import datetime
from functools import cache
from abc import abstractmethod
from strategy_v2.Strategy import StrategyBase
from strategy_v2.TradingSubSystem import DataLoader, TradingSubSystemModel
from utils.performance import performance_summary_plot
from utils.logging import get_logger

class TradingSubSystemBase(TradingSubSystemModel, DataLoader):
    '''
        Base Class of a Trading System which implements all the base init function and data function
        Datatype definiton for convenience:
        - Position = pd.DataFrame
        - returns = pd.Series
    '''
    def __init__(self,                
                instruments: list=[str], 
                vol_target: float=0.2, 
                max_leverage=2,
                strategy: list[StrategyBase]=[],
                offset=0,                                 
                ):
        
        '''            
            instruments:    List of instruments to be included in the trading systems
            vol_target:     User defined volatility target
            offset:         Extended periods before start date when we get the data, but not include in backtest periods        
            instruments:    List of instruments in the trading sub-systems
            strategy:       List of strategy to include in the trading sub-systems            
        '''        
        self.offset = offset     
        self.instruments = instruments
        self.vol_target = vol_target
        self.max_leverage = max_leverage
        self.strategy = strategy        
        self.logger = get_logger('{} [{}]'.format(self.__class__.__name__, ','.join(instruments)))

        # data object which will pass to all underlying strategy
        self.data = {}

        # backtest generated position from underlying strategy
        self.position = pd.DataFrame()

        # backtest return from underlying strategy
        self.ret = pd.DataFrame()        

    def __str__(self) -> str:
        strategy_names = ','.join([str(s) for s in self.strategy])
        instrument_names = ','.join(self.instruments)
        return "{} - {} ({})".format(self.__class__.__name__.replace('TradingSubSystem', ''), instrument_names, strategy_names)   

    def set_start_date(self, start_date: datetime):
        '''
            start_date: start_date of the backtest periods            
        '''
        self.start_date = start_date
        return self

    def set_end_date(self, end_date: datetime):
        '''            
            end_date: end_date of the backtest periods
        '''
        self.end_date = end_date
        return self
    
    def backtest(self):
        """
            Backtest all your strategies in the subsystem
        """
        self.load_data()
        for i, s in enumerate(self.strategy):            
            self.logger.info('Generating position for strategy {} between {} and {}......'.format(str(s), self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d')))

            # set properties to strategy
            s.set_data(self.data)            
            s.set_start_date(self.start_date)
            s.set_end_date(self.end_date)                        
            s.preprocess_data()
            s.generate_position()

            pos = s.get_position().copy()
            pos.columns = pd.MultiIndex.from_tuples([(str(s), c) for c in pos.columns])
            ret = self.generate_backtest_return(s.get_position())

            if i == 0:
                self.position = pos
                self.ret = pd.DataFrame(ret, columns=[str(s)])
            else:
                self.position = pd.merge(self.position, pos, how='outer', on=['Date'], validate='1:1')
                self.ret[str(s)] = ret    

        # position columns here are in strategy - symbol but NOT symbol only
        self.position = self.position.fillna(0)
        self.ret = self.ret.fillna(0) 

    def generate_backtest_return(self, 
                                 position: pd.DataFrame
        ) -> pd.Series:
        """
            Assume Daily Rebalance so we can directly calculate the return using C2C Pct change.
        """                  
        close_px = self.data['px']['Close'].loc[position.index]                
        close_ret = close_px / close_px.shift(1) - 1
        close_ret = close_ret.fillna(0)        
        ret = position * close_ret
        ret = ret.sum(axis=1)
        return ret
    
    def performance(self, 
                    benchmark: list[str]=['^SPX'],
                    show_all_rets: bool=True
                    ):
        """
            Generate backtest summary given positions from multiple strategies
            benchmark:       Benchmarks to compare in performance plots
            show_all_rets:   True if you want to show all sub-system return. Otherwise, only show the portfolio returns
        """         

        all_rets = self.ret.copy()
        all_rets['Combined'] = self.combined_ret
        all_rets['Combined and Volatility Targeted'] = self.scaled_combined_ret
        if not show_all_rets:
            all_rets = all_rets[['Combined and Volatility Targeted']]

        # performance summary
        performance_summary_plot(all_rets, benchmark=benchmark)

    def get_data(self) -> dict:
        return self.data

    @cache    
    def load_data(self):
        """
            Load all required data for underlying strategy based on start_date and end_date
        """        
        self.load_price_data_yahoo()        
    
    @abstractmethod
    def optimize(self):
        pass

    @abstractmethod    
    def get_position(self) -> pd.DataFrame:
        '''
            Return the final position after all combine weights and scaling
            This function is called by Portfolio
        '''
        pass

    @abstractmethod
    def get_return(self) -> pd.Series:
        '''
            Return the final returns of the system after all combined weights and scaling
            This function is called by Portfolio
        '''
        pass