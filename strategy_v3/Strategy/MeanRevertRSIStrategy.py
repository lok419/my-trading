from datetime import datetime
from pandas.core.frame import DataFrame
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Strategy import STATUS, StrategyBase
from strategy_v3.Strategy.Performance import StrategyPerformance
import numpy as np
import re
import pandas as pd

from utils.ta import rsi


class MeanRevertRSIStrategy(StrategyBase, StrategyPerformance):    

    def __init__(self, 
                 instrument: str, 
                 interval: str,   
                 refresh_interval: int = 60,    
                 sma_windows: int = 5,            
                 lma_windows: int = 200,
                 rsi_windows: int = 2,
                 rsi_threshold: int = 10,
                 position_size: float = 50,                 
                 status: str = STATUS.RUN,
                 start_date: str = None,
                 verbose: bool = True,   
                 comment: str = '',
        ):
        '''
            instrument:             The instrument to trade
            interval:               time interval to trade            
            refresh_interval:       frequency of function execute() is called (in mintues)                 
            status:                 user can set status to control the strategy behavior
            start_date:             indicate the start time of the strategy so that we can extract the whole performance history of a strategy. The time is based on HongKong Time
            verbose:                True to print the log message
            comment:                Comment to save for the daily pnl or logging. This is used to tag what did you do to the model so we can keep track the actions
        '''
        super().__init__(
            instrument=instrument,
            interval=interval,   
            refresh_interval=refresh_interval,                     
            status=status,   
            start_date=start_date,         
            verbose=verbose,
            comment=comment,
        )    
        self.sma_windows = sma_windows
        self.lma_windows = lma_windows
        self.rsi_windows = rsi_windows
        self.rsi_threshold = rsi_threshold
        self.position_size = position_size  
        self.trd_id = 0

    def __str__(self):
        return 'mrr_{}'.format(self.strategy_id)
    
    def set_strategy_id(self, 
                        strategy_id: str, 
                        reload: bool = False, 
                        force_reload_all: bool = False):
        '''
            Set strategy id. this is used to identify the orders from same strategy instance
            strategy_id:        The strategy name
            reload:             True to reload all previous property (e.g. grid_id, grid_type) by querying orders before last closing periods
            force_reload_all:   True to reload all previous property (e.g. grid_id, grid_type) by quering LTD orders
        '''
        super().set_strategy_id(strategy_id)

        if reload and not self.is_backtest() and self.mr_id == 0:            
            if force_reload_all:
                df_orders = self.get_all_orders(start_date=self.start_date)
            else:
                df_orders = self.get_all_orders()

            if len(df_orders) > 0:                
                trd_id = df_orders.iloc[-1]['trd_id']                    
                self.trd_id = trd_id     

    def execute(self, data):
        '''
            Execute function which is called repeatedly for each tick
        '''
        date = data['Date']        
        current_px = data['Open'] if self.is_backtest() else data['Close']
        low = data['Low']
        high = data['High']

        current_position = self.get_current_position()
        signals = self.signals(data)

        if signals == 1 and current_position == 0:     
            self.place_order(current_px, date)

        self.executor.fill_orders(date, low, high)        

        if signals == -1 and current_position > 0:
            self.cancel_all_orders()
            stop_px = data['Close'] if self.is_backtest() else None
            self.close_out_positions('cls', stop_px, date)             

    def signals(self, data):
        rsi = data['rsi']
        px_sma = data['px_sma']     
        px_lma = data['px_lma']   
        current_px = data['current_px']
            
        entry = current_px > px_lma and rsi < self.rsi_threshold
        exit = current_px > px_sma or current_px < px_lma
        
        if entry:
            return 1
        elif exit:
            return -1
        else:
            return 0

    def run(self):
        '''
            Actual function to execute the strategy repeatedly
        '''        
        super().run(lookback='48 hours ago')

    def place_order(self, 
                    px: float,                    
                    date: datetime):
        '''
            Place the optimal bid-ask spread orders        
            px:    Current Price
            date:  date of order (only used for backtest)
        '''
        self.trd_id += 1
        quantity = self.position_size / px
        order_id = f'{self.__str__()}_id{self.trd_id}_trd'

        self.executor.place_order(
                instrument=self.instrument,
                side='BUY',
                order_type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=px,      
                stopPrice=None,           
                date=date,          # only used for backtest      
                order_id=order_id
        )
         
    def load_data(self, lookback:str|datetime, lookback_end:str|datetime=None) -> DataFrame:        
        '''
            Load all hisotorical price data

            In backtest scenarios, assume we made all decisions based at the beginning of the interval,
            so we use open price as current price

            In real scenarios, close price is always the latest price, so currnet price is close price
        '''
        df = super().load_data(lookback, lookback_end) 
        df['sma_windows'] = self.sma_windows
        df['lma_windows'] = self.lma_windows
        df['rsi_windows'] = self.rsi_windows
        df['rsi_threshold'] = self.rsi_threshold
        df['current_px'] = df['Open'] if self.is_backtest() else df['Close']

        df['return'] = df['current_px'].pct_change().shift(1)
        df['rsi'] = rsi(df['current_px'], window=self.rsi_windows)
        df['px_sma'] = df['current_px'].rolling(self.sma_windows).mean()
        df['px_lma'] = df['current_px'].rolling(self.lma_windows).mean()
        
        df['cond_rsi'] = df['rsi'] < df['rsi_threshold']
        df['cont_lma'] = df['current_px'] > df['px_lma']
                
        self.df = df 
    
    def get_all_orders(self,                        
                       trade_details: bool = False,     
                       limit: int = 1000,  
                       start_date: datetime = None,
                       end_date: datetime = None,    
                       offset: int = 0,                         
                       ) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)            
            trade_details:  True if we want to add trade details (e.g. fill price, commission etc....)
            limit:          number of orders per query
            start_date:     query start date of the orders
            end_date:       query end date of the orders
            offset:         lookback period to get the orders if start_date is not given
        '''
        
        df_orders = super().get_all_orders(trade_details=trade_details, limit=limit, start_date=start_date, end_date=end_date, offset=offset)                   
        df_orders['trd_id'] = df_orders['clientOrderId'].apply(lambda x: int(re.search(r"(?<=id)\d+(?=_)", x)[0]))        
        df_orders['trd_tt'] = df_orders['clientOrderId'].apply(lambda x: x.split('_')[-1]).apply(lambda x: re.sub(r'[0-9]', '', x))

        return df_orders     
    
    def close_out_positions(self,                                                         
                            type:str = 'cls',
                            price: float = None,
                            date:datetime = None,
                            offset: int = 0,                            
                            ):
        '''
            Close out all outstanding positions based on Grid orders
            type:   reason of the close out. Either stoploss or close (end of strategy)
            price:  only for backtest, MARKET ORDER does not need price
            date:   only used for backtest
            offset: lookback period to derive the outstanding positions to close         
        '''
        order_id = order_id = f'{self.__str__()}_id{self.trd_id}_{type}'
        super().close_out_positions(type=type, price=price, order_id=order_id, date=date, offset=offset)

    # def log_data(self, data: dict = dict()):
    #     '''
    #         Log the strategy data for each execute()
    #     '''
    #     try:
    #         log = {}
    #         # strategy parameters            
    #         log['date'] = self.get_current_time()
    #         log['strategy_id'] = self.strategy_id
    #         log['interval'] = self.interval
    #         log['refresh_interval'] = self.refresh_interval
    #         log['ma_windows'] = self.ma_windows
    #         log['rsi_windows'] = self.rsi_windows
    #         log['rsi_threshold'] = self.rsi_threshold            
    #         log['position_size'] = self.position_size            
    #         log['status'] = self.status.name   

    #         # execute data
    #         log.update(data)
    #         log = pd.DataFrame([log])

    #         log_df = self.get_log_data()
    #         if log_df is not None:
    #             log_df = pd.concat([log_df, log])
    #         else:
    #             log_df = log

    #         log_df.to_hdf(self.log_path, key='log', format='table')

    #     except Exception as e:
    #         self.logger.error(e)