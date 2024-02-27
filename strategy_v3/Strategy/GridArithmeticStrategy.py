from strategy_v3.Executor import ExecutorBinance, ExecutorModel, ExecutorBacktest
from strategy_v3.DataLoader import DataLoaderModel
from strategy_v3.Strategy import StrategyPerformance
from datetime import datetime, timezone
from utils.stats import time_series_half_life, time_series_hurst_exponent
from utils.logging import get_logger
from pandas.core.frame import DataFrame
from enum import Enum
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import random
import math
import re

class Status(Enum):
    IDLE = 1
    ACTIVE = 2
    NEUTRAL = 3

class TS_PROP(Enum):
    RANDOM = 0
    MEAN_REVERT = 1
    MOMENTUM = 2

class GridArithmeticStrategy(StrategyPerformance):

    def __init__(self, 
                 instrument:str, 
                 interval:str,
                 grid_size:int = 5,
                 vol_lookback:float = 30,
                 vol_grid_scale: float = 1,
                 vol_stoploss_scale: float = 7,
                 position_size: float = 100,
                 hurst_exp_mr_threshold: float = 0.5,
                 hurst_exp_mo_threshold: float = float('inf'),
                 verbose: bool = True
        ):
        '''
            instrument:             The instrument to trade
            interval:               time interval to trade
            grid_size:              Size of the Grid, this refers to one directional, i.e. if grids size = 5, that means 5 buy ordes + 5 sell orders
            vol_lookback:           Lookback period to determine the historical volatility
            vol_grid_size:          Grid spacing in terms of historical volatility
            vol_stoploss_scale:     Stoploss distance from center of the grids in terms of historical volatility
            position_size:          Position for each orders in terms of $USD
            hurst_exp_threshold:    Maxmium hurst exponent ratio to put a grid trade
            verbose:                True to print the log message
        '''
        self.instrument = instrument
        self.interval = interval
        self.grid_size = grid_size        
        self.vol_lookback = vol_lookback
        self.vol_grid_scale = vol_grid_scale
        self.vol_stoploss_scale = vol_stoploss_scale
        self.position_size = position_size
        self.hurst_exp_mr_threshold = hurst_exp_mr_threshold
        self.hurst_exp_mo_threshold = hurst_exp_mo_threshold

        self.grid_id = 0        
        self.stoploss = (float('-inf'), float('inf'))
        self.executor = None
        self.data_loader = None

        # strategy_id is used to identify the strategy from list of orders
        self.strategy_id = str(round(random.random() * 1e6))
        self.logger = get_logger(self.__str__())

        self.qty_decimal = 5
        self.price_decimal = 2

        # 5m -> 5mins for round function
        self.interval_round = self.interval + 'in' if self.interval.endswith('m') else self.interval
        if not verbose:
            self.logger.setLevel('CRITICAL')

        self.execute_start_time = pd.to_datetime(datetime.now(tz=ZoneInfo("HongKong")))

    def __str__(self):
        return 'grid_{}'.format(self.strategy_id)

    def set_executor(self, executor:ExecutorModel):
        '''
            Define a executor object which executes all trading operation
            We need to make sure both executor and data_loader has the same timezone
                - For Binance, it is based on UTC time (i.e. GMT+0)                        
        '''
        self.executor = executor
        self.executor.set_logger(self.logger)

    def set_data_loder(self, data_loader:DataLoaderModel):
        '''
            Defines a dataload object which sources all OHCL data
            We need to make sure both executor and data_loader has the same timezone
                - For Binance, it is based on UTC time (i.e. GMT+0)
        '''
        self.data_loader = data_loader

    def set_price_decimal(self, price_decimal:int):
        self.price_decimal = price_decimal

    def set_qty_decimal(self, qty_decimal:int):
        self.qty_decimal = qty_decimal

    def set_strategy_id(self, id:str):
        self.strategy_id = id
        self.logger.name = self.__str__()    

    def is_backtest(self):
        return type(self.executor) is ExecutorBacktest
    
    def get_current_time(self) -> datetime:
        return pd.to_datetime(datetime.now(timezone.utc)).replace(tzinfo=None)

    def load_data(self, lookback):
        df = self.data_loader.load_price_data(self.instrument, self.interval, lookback)        

        # need at least 100 data points to determine the hurst exponent ratio
        assert len(df) > min(100, self.vol_lookback)

        # preprocess all data. These data are using close price, so we need to shift by 1
        #df['Vol'] = df['Close'].diff().rolling(self.vol_lookback).std().shift(1)

        '''
            for backtest, we cannot use close price for the same interval as this implies lookahead bias
            for actual trading, we can use the close price which represents the latest price within the same interval
        '''
        
        shift = 1 if self.is_backtest() else 0
        df['Vol'] = df['Close'].rolling(self.vol_lookback).std().shift(shift)
        df["half_life"] = df['Close'].rolling(100).apply(lambda x: time_series_half_life(x)).shift(shift)
        df["hurst_exponent"] = df["Close"].rolling(100).apply(lambda x: time_series_hurst_exponent(x)).shift(shift)
        df['hurst_exponent_avg'] = df["hurst_exponent"].rolling(self.vol_lookback).mean().shift(shift)
        df['Close_t5'] = df['Close'].shift(5)
        df['Close_t10'] = df['Close'].shift(10)
        df['Close_t20'] = df['Close'].shift(20)
        df['Close_sma'] = df['Close'].rolling(self.vol_lookback).mean().shift(shift)        

        # compute rolling metrics based on time-series half-life
        vol_hl = []
        close_hl = []

        close = df['Close'].values        
        half_life = df['half_life'].values

        for i in range(len(close)):            
            if not math.isnan(half_life[i]):
                # we want to calculate the average of one complete mean-reversion cycle, so we lookback half-life * 2
                lookback = round(half_life[i]) * 2
                close_ = close[max(i-lookback+1, 0):i+1]
                close_mean = np.mean(close_)
                std = np.std(close_)
            else:
                close_mean = std = math.nan
                
            close_hl.append(close_mean)       
            vol_hl.append(std)

        df['Close_sma_hl'] = close_hl     
        df['Close_sma_hl'] = df['Close_sma_hl'].shift(shift)
        df['Vol_hl'] = vol_hl
        df['Vol_hl'] = df['Vol_hl'].shift(shift)

        # remove all null data
        df = df[~df['Vol'].isnull()]
        df = df[~df['half_life'].isnull()]
        df = df[~df['hurst_exponent'].isnull()]
        self.df = df    

    def execute(self, data):
        '''
            This function is used to execute for ONE TIME INTERVAL
            For real trading, for each refreshed time interval, we call this function once only.
                - we use close price on grid orders
                
            For backtesting, for each historical time interval, we call this function every time interval            
                - we use open price on grid orders
                - high and low price to see if fill orders and triggers stop loss

            data: structure contains all required data for strategy
        '''
        date = data['Date']
        hurst_exponent = data['hurst_exponent']
        open, close, high, low = data['Open'], data['Close'], data['High'], data['Low']            
        vol = data['Vol']        

        if vol is None or math.isnan(vol):
            self.logger.error('historical volatility is null. exit.'.format(date.strftime('%Y-%m-%d %H:%M:%S')))
            return 

        if hurst_exponent is None or math.isnan(hurst_exponent):
            self.logger.error('hurst exponent is null. exit.'.format(date.strftime('%Y-%m-%d %H:%M:%S')))
            return        
        
        status = self.get_status()
        ts_prop = self.get_ts_prop(data)
        self.logger.info('state: {}, ts_prop: {}, hurst_exponent: {:.2f}.'.format(status.name, ts_prop.name, hurst_exponent))

        # At beginning of periods, use open price to create grid orders                
        if status == Status.IDLE and ts_prop != TS_PROP.RANDOM:            
            center_px = self.derive_grid_center_px(data, ts_prop)                  
            if center_px is not None:
                self.place_grid_order(center_px, vol, ts_prop, date)

        '''
            Only used for backtest - fill the orders using high and low            
        '''        
        self.executor.fill_orders(date, low, high)        
        status = self.get_status()            

        # after filling, if position is netural, cancel all orders
        if status == Status.NEUTRAL:
            self.logger.info('status: {}.'.format(status.name))      
            self.cancel_all_orders()            

        # finally, check the close price and determine if we need stop loss            
        if status == Status.ACTIVE and (close < self.stoploss[0] or close > self.stoploss[1]):                         
            stop_px = self.stoploss[0] if close < self.stoploss[0] else self.stoploss[1]
            stop_px = round(stop_px, self.price_decimal)
            self.logger.info('stop loss are triggered at {}.'.format(stop_px))

            # for real trading, we don't need a actual price for stoploss
            stop_px = stop_px if self.is_backtest() else None
            
            self.cancel_all_orders()
            self.close_out_positions('stoploss', stop_px, date)

    def get_status(self) -> Status:
        if self.is_idle():
            return Status.IDLE
        elif self.is_position_neutral():
            return Status.NEUTRAL
        else:
            return Status.ACTIVE
        
    def get_ts_prop(self, data: dict) -> TS_PROP:
        hurst_exponent = data['hurst_exponent']
        if hurst_exponent < self.hurst_exp_mr_threshold:
            return TS_PROP.MEAN_REVERT        
        elif hurst_exponent > self.hurst_exp_mo_threshold:
            return TS_PROP.MOMENTUM        
        else:
            return TS_PROP.RANDOM

    def is_idle(self) -> bool:
        '''
            Check if the current status idle => no pending grid orders

            Real trading only: We also need to make sure the filter out orders before strategy start-time
        '''
        all_orders = self.get_all_orders() 

        if not self.is_backtest():
            all_orders = all_orders[all_orders['time'] > self.execute_start_time]
        
        pending = all_orders.copy()
        pending = pending[pending['status'] != 'FILLED']
        pending = pending[pending['status'] != 'CANCELED']        
        is_idle = len(pending) == 0
        return is_idle

    def is_position_neutral(self) -> bool:
        '''
            Check if the filled orders are delta neutral given there is pending orders within last set of grid orders.
                1. There is pending orders
                2. There is filled orders
                3. All filled orders qty add up to zero (delta neutral)

            If yes, we should consider to cancel all pending orders and restart the grid orders

            Real trading only: We also need to make sure the filter out orders before strategy start-time
        '''
        all_orders = self.get_all_orders() 

        if not self.is_backtest():              
            all_orders = all_orders[all_orders['time'] > self.execute_start_time]

        last_grid = all_orders[all_orders['clientOrderId'].str.contains(f'_gridid{self.grid_id}_')]        

        pending = last_grid.copy()        
        pending = pending[pending['status'] != 'FILLED']
        pending = pending[pending['status'] != 'CANCELED']

        filled = last_grid.copy()    
        filled = filled[filled['status'] == 'FILLED']        

        has_pending = len(pending) > 0
        has_filled = len(filled) > 0
        filled_net_qty = filled['NetExecutedQty'].sum()
        filled_net_qty = round(filled_net_qty, self.qty_decimal)

        return has_filled and has_pending and abs(filled_net_qty) == 0
    
    def close_out_positions(self,                                                         
                            type:str = 'close',
                            price: float = None,
                            date:datetime = None,
                            ):
        '''
            Close out all outstanding positions based on Grid orders
            type:  reason of the close out. Either stoploss or close (end of strategy)
            price: only for backtest, MARKET ORDER does not need price
            date:  only used for backtest
        '''
        date = date if date is not None else self.get_current_time()
        all_orders = self.get_all_orders()            

        filled = all_orders[all_orders['status'] == 'FILLED']        
        filled_net_qty = filled['NetExecutedQty'].sum()   
        filled_net_qty = round(filled_net_qty, self.qty_decimal)

        if abs(filled_net_qty) > 0:
            self.logger.info('closing out net position of {} {}....'.format(filled_net_qty, self.instrument))

            if price is not None:
                price = round(price, self.price_decimal)

            order_id = f'{self.__str__()}_gridid{self.grid_id}_{type}'
            side = 'BUY' if filled_net_qty < 0 else 'SELL'
            quantity = abs(filled_net_qty)
            self.executor.place_order(
                instrument=self.instrument,
                side=side,
                order_type='MARKET',
                timeInForce=None,
                quantity=quantity,                
                date=date, 
                order_id=order_id,
                price=price,
            )
        else:
            self.logger.info('nothing to close out because of no oustanding positions....'.format(date.strftime('%Y-%m-%d %H:%M:%S')))

    def derive_grid_center_px(
            self,
            data: DataFrame,   
            ts_prop: TS_PROP,         
        ) -> float:
        '''
            Calculate the grid center price
                - for mean reverting, we use current price as grid center price
                - for momentum, we use a current price +- x * vol * vol_grid_scale to follow the price momentum.
                  meanwhile, there is also a extra momentum filter to make sure there is short-term momentum
                
            data:       the input data to execute()
            ts_prop     price-series property
        '''
        current_px = data['Open'] if self.is_backtest() else data['Close']
        current_vol = data['Vol']

        if ts_prop == TS_PROP.MEAN_REVERT:
            center_px = current_px

        elif ts_prop == TS_PROP.MOMENTUM:       
            if current_px > data['Close_t5'] and data['Close_t5'] > data['Close_t10']:
                center_px = current_px + (self.grid_size+1) * current_vol * self.vol_grid_scale
            elif current_px < data['Close_t5'] and data['Close_t5'] < data['Close_t10']:
                center_px = current_px - (self.grid_size+1) * current_vol * self.vol_grid_scale
            else:
                center_px = None
        else:
            center_px = None

        return center_px

    def place_grid_order(
            self,            
            center_px:float,
            current_vol:float,
            ts_prop: TS_PROP,
            date:datetime,             
        ):
        '''
            Generate Grid Orders based on current price. Meanwhile also return StopLoss for the Grid
            current_px:     neutral price, center of the grids
            current_vol:    the historical volatiltiy used to determine the grid spacing
            ts_prop:        current price time-series porperties (only used for logging)
            date:           only used for backtest
        '''        
        grid_scales = list(range(-self.grid_size,self.grid_size+1,1))        
        grid_prices = [center_px + x * current_vol * self.vol_grid_scale for x in grid_scales]
        stoploss = (center_px - self.vol_stoploss_scale * current_vol * self.vol_grid_scale, center_px + self.vol_stoploss_scale * current_vol * self.vol_grid_scale)

        self.grid_id += 1
        order_id = f'{self.__str__()}_gridid{self.grid_id}_grid'

        # grid quantity
        quantity = self.position_size / center_px
        quantity = round(quantity, self.qty_decimal)
        self.logger.info('creating {} {} grid orders of {} {} at grid center price {}....'.format(self.grid_size * 2, ts_prop.name, quantity, self.instrument, round(center_px, self.price_decimal)))
        
        for i, px in enumerate(grid_prices):
            if grid_scales[i] == 0:
                continue

            side = 'SELL' if grid_scales[i] > 0 else 'BUY'            
            order_id_ = order_id + str(i)     
            px = round(px, self.price_decimal)             

            # place grid orders
            self.executor.place_order(
                instrument=self.instrument,
                side=side,
                order_type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=px, 
                date=date,          # only used for backtest      
                order_id=order_id_
            )

        self.stoploss = stoploss

    def get_all_orders(self) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)
        '''
        df_orders = self.executor.get_all_orders(self.instrument)
        df_orders = df_orders[df_orders['clientOrderId'].str.startswith(self.__str__())]

        # find the grid_id of orders
        df_orders['grid_id'] = df_orders['clientOrderId'].apply(lambda x: int(re.search(r"(?<=gridid)\d+(?=_)", x)[0]))

        # find the grid trade type (grid/stoploss/close)
        df_orders['grid_type'] = df_orders['clientOrderId'].apply(lambda x: x.split('_')[-1]).apply(lambda x: re.sub(r'[0-9]', '', x))        

        '''
            Binance API does not show fill price for market orders, we need to fill it ourselves. 
            We round to nearest interval and take the open price as market price
        '''
        if type(self.executor) is ExecutorBinance and len(df_orders):
            mkt_orders = df_orders[df_orders['type'] == 'MARKET']
            df_orders = df_orders[df_orders['type'] != 'MARKET']
            
            mkt_orders['Date'] = mkt_orders['updateTime'].dt.round(self.interval_round)            
            mkt_orders = pd.merge(mkt_orders, self.df[['Date', 'Open']], how='left', validate='m:1')
            mkt_orders['price'] = np.where(mkt_orders['price'] > 0, mkt_orders['price'], mkt_orders['Open'])
            mkt_orders = mkt_orders.drop(columns=['Open', 'Date'])
            df_orders = pd.concat([df_orders, mkt_orders])            

        return df_orders
    
    def cancel_all_orders(self):
        '''
            Cancel all orders created traded by this object
        '''                                
        self.logger.info('cancelling all orders.....')
        df_orders = self.get_all_orders()
        df_orders = df_orders[df_orders['status'] != 'FILLED']                
        df_orders = df_orders[df_orders['status'] != 'CANCELED']
        self.executor.cancel_order(self.instrument, df_orders)      