from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Strategy import TS_PROP, GRID_TYPE, GRID_STATUS, STATUS, StrategyBase
from strategy_v3.Strategy.Performance import GridPerformance
from datetime import datetime
from utils.stats import time_series_half_life, time_series_hurst_exponent
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import math
import re

class GridArithmeticStrategy(StrategyBase, GridPerformance):

    def __init__(self, 
                 instrument:str, 
                 interval:str,
                 refresh_interval:int = 60,
                 grid_size:int = 5,
                 vol_lookback:float = 30,
                 vol_grid_scale: float = 1,
                 vol_stoploss_scale: float = 7,
                 position_size: float = 100,
                 hurst_exp_mr_threshold: float = 0.5,
                 hurst_exp_mo_threshold: float = float('inf'),
                 price_decimal: int = 2,
                 qty_decimal: int = 5,                             
                 status: str = STATUS.RUN,
                 start_date: str = None,
                 verbose: bool = True,                 
        ):
        '''
            instrument:             The instrument to trade
            interval:               time interval to trade
            refresh_interval:       frequency of function execute() is called (in mintues)
            grid_size:              Size of the Grid, this refers to one directional, i.e. if grids size = 5, that means 5 buy ordes + 5 sell orders
            vol_lookback:           Lookback period to determine the historical volatility
            vol_grid_size:          Grid spacing in terms of historical volatility
            vol_stoploss_scale:     Stoploss distance from edge of the grids in terms of historical volatility
            position_size:          Position for each orders in terms of $USD
            hurst_exp_threshold:    Maxmium hurst exponent ratio to put a grid trade
            price_decimal:          rounding decimal of price
            qty_decimal:            rounding decimal of quantity
            status:                 user can set status to control the strategy behavior
            start_date:             indicate the start time of the strategy so that we can extract the whole performance history of a strategy. By default, the time is based on GMT+8 and converted to UTC
            verbose:                True to print the log message
        '''
        super().__init__(
            instrument=instrument,
            interval=interval,
            refresh_interval=refresh_interval,
            price_decimal=price_decimal,
            qty_decimal=qty_decimal,
            status=status,
            start_date=start_date,
            verbose=verbose,
        )        
        
        self.grid_size = grid_size        
        self.vol_lookback = vol_lookback
        self.vol_grid_scale = vol_grid_scale
        self.vol_stoploss_scale = vol_stoploss_scale
        self.position_size = position_size
        self.hurst_exp_mr_threshold = hurst_exp_mr_threshold
        self.hurst_exp_mo_threshold = hurst_exp_mo_threshold        

        # this saves the current grid stats
        self.grid_id = 0
        self.grid_type = None
        self.stoploss = (float('-inf'), float('inf'))                

        # grid type name acronym map
        self.grid_char_to_type = {''.join([s[0] for s in x.split('_')]): x for x in GRID_TYPE._member_names_}
        self.grid_type_to_char = {x: ''.join([s[0] for s in x.split('_')]) for x in GRID_TYPE._member_names_}        

    def __str__(self):
        return 'grid_{}'.format(self.strategy_id)

    def set_strategy_id(self, id:str, reload: bool = True):
        '''
            Set strategy id. this is used to identify the orders from same strategy instance
            id:     The strategy name
            reload: True to reload all previous property (e.g. grid_id, grid_type) from latest
        '''
        super().set_strategy_id(id)

        if reload and not self.is_backtest():
            df_orders = self.get_all_orders(limit=10)
            if len(df_orders) > 0:
                grid_type = df_orders.iloc[-1]['grid_type']
                grid_id = df_orders.iloc[-1]['grid_id']

                if self.grid_type is None:
                    self.grid_type = GRID_TYPE[grid_type]

                if self.grid_id == 0:
                    self.grid_id = grid_id

    def load_data(self, lookback):
        super().load_data(lookback)        
        df = self.df
        
        '''
            for backtest, we cannot use close price for the same interval as this implies lookahead bias
            for actual trading, we can use the close price which represents the latest price within the same interval. 

            However, we still use T-1 close because latest close doesn't imply full time interval. This could lead to underestimate of vol metrics.
                e.g. latest close can be close since last 1s, 30s, 1m, 5m (assume interval is 5m)
        '''

        df['close_std'] = df['Close'].rolling(self.vol_lookback).std().shift(1)        
        df['close_sma'] = df['Close'].rolling(self.vol_lookback).mean().shift(1)

        df["half_life"] = df['Close'].rolling(100).apply(lambda x: time_series_half_life(x)).shift(1)
        df["hurst_exponent"] = df["Close"].rolling(100).apply(lambda x: time_series_hurst_exponent(x)).shift(1)
        df['hurst_exponent_avg'] = df["hurst_exponent"].rolling(self.vol_lookback).mean()

        # Average True Range
        df['tr'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1)))
        df['atr'] = df['tr'].rolling(self.vol_lookback).mean().shift(1)

        df[['Close_t5', 'Low_t5', 'High_t5']] = df[['Close', 'Low', 'High']].shift(5)
        df[['Close_t10', 'Low_t10', 'High_t10']] = df[['Close', 'Low', 'High']].shift(10)
        df[['Close_t20', 'Low_t20', 'High_t20']] = df[['Close', 'Low', 'High']].shift(20)                     

        # compute rolling metrics based on time-series half-life
        std_hl = []
        close_hl = []

        close = df['Close'].values        
        half_life = df['half_life'].values

        for i in range(len(close)):            
            if not math.isnan(half_life[i]):
                # we want to calculate the average of one complete mean-reversion cycle, so we lookback half-life * 2
                lookback = round(half_life[i]) * 2

                # not include i otherwise could be lookahead bias
                close_ = close[max(i-lookback, 0):i]
                close_mean = np.mean(close_)
                std = np.std(close_)
            else:
                close_mean = std = math.nan
                
            close_hl.append(close_mean)       
            std_hl.append(std)

        df['close_sma_hl'] = close_hl             
        df['close_std_hl'] = std_hl        

        self.df = df    

    def execute(self, data=None):
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
        vol = data['atr']

        if vol is None or math.isnan(vol):
            self.logger.error('historical volatility is null. exit.'.format(date.strftime('%Y-%m-%d %H:%M:%S')))
            return 

        if hurst_exponent is None or math.isnan(hurst_exponent):
            self.logger.error('hurst exponent is null. exit.'.format(date.strftime('%Y-%m-%d %H:%M:%S')))
            return        
        
        grid_status = self.get_grid_status()
        ts_prop = self.get_ts_prop(data)        
        self.logger.info('status: {}, grid_status: {}, ts_prop: {}, hurst_exponent: {:.2f}.'.format(self.status.name, grid_status.name, ts_prop.name, hurst_exponent))

        '''
            When status is STOP, return
        '''
        if self.status == STATUS.STOP:            
            return           

        '''
            When status is RUN, grid status is IDLE and time-series is not random => place grid orders            
        '''
        if self.status == STATUS.RUN and grid_status == GRID_STATUS.IDLE :#and ts_prop != TS_PROP.RANDOM:          
            center_px, stoploss, grid_type = self.derive_grid_center_px(data, ts_prop)
            current_vol = vol
            current_px = open if self.is_backtest() else close

            if center_px is not None and stoploss is not None:
                self.place_grid_order(center_px, current_px, current_vol, grid_type, date)
                self.stoploss = stoploss

        '''
            Only used for backtest - fill the orders using high and low
        '''        
        self.executor.fill_orders(date, low, high)        
        grid_status = self.get_grid_status()

        '''
            After filling, check if position is active netural. If yes, cancel all orders
        '''        
        if grid_status == GRID_STATUS.NEUTRAL:
            self.logger.info('status: {}.'.format(grid_status.name))      
            self.cancel_all_orders()            

        '''
            Check the close price and determine if we need stop loss        
            for real trading, we don't need a actual price for stoploss
        '''        
        if grid_status == GRID_STATUS.ACTIVE and (close < self.stoploss[0] or close > self.stoploss[1]):                         
            stop_px = self.stoploss[0] if close < self.stoploss[0] else self.stoploss[1]
            stop_px = round(stop_px, self.price_decimal)
            self.logger.info('stop loss are triggered at {}.'.format(stop_px))
            
            stop_px = stop_px if self.is_backtest() else None            
            self.cancel_all_orders()
            self.close_out_positions('stoploss', stop_px, date, force=True)

        '''
            If status is terminate, cancel all orders and close out the positions
            Update status to STOP so the strategy won't run
        '''        
        if self.status == STATUS.TERMINATE:
            self.cancel_all_orders()                        
            self.close_out_positions('close', None, None, force=True)
            self.status = STATUS.STOP
            ExecuteSetup(self.strategy_id).update("status", STATUS.STOP.name, type(self))

    def get_grid_status(self) -> GRID_STATUS:
        if self.is_idle():
            return GRID_STATUS.IDLE
        elif self.is_active_neutral():
            return GRID_STATUS.NEUTRAL
        else:
            return GRID_STATUS.ACTIVE
        
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

    def is_active_neutral(self) -> bool:
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
                            force: bool = False
                            ):
        '''
            Close out all outstanding positions based on Grid orders

            type:  reason of the close out. Either stoploss or close (end of strategy)
            price: only for backtest, MARKET ORDER does not need price
            date:  only used for backtest
            force: force to close out entire position (usually done manually)
        '''
        date = date if date is not None else self.get_current_time()
        all_orders = self.get_all_orders(query_all=force)    

        if not self.is_backtest() and not force:                          
            all_orders = all_orders[all_orders['time'] > self.execute_start_time]           

        '''
            Force to flatten all delta based on LTD orders
        '''
        if force:            
            filled = all_orders[all_orders['status'] == 'FILLED']
            if self.grid_type is None:
                self.grid_type = GRID_TYPE[all_orders.iloc[-1]['grid_type']]

            if self.grid_id is None:
                self.grid_id = all_orders.iloc[-1]['grid_id']

        else:
            last_grid = all_orders[all_orders['clientOrderId'].str.contains(f'_gridid{self.grid_id}_')]  
            filled = last_grid[last_grid['status'] == 'FILLED']

        filled_net_qty = filled['NetExecutedQty'].sum()   
        filled_net_qty = round(filled_net_qty, self.qty_decimal)

        if abs(filled_net_qty) > 0:
            self.logger.info('closing out net position of {} {}....'.format(filled_net_qty, self.instrument))

            if price is not None:
                price = round(price, self.price_decimal)

            grid_type_char = self.grid_type_to_char[self.grid_type.name]                        
            order_id = f'{self.__str__()}_gridid{self.grid_id}_{grid_type_char}_{type}'
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
                stopPrice=None,
            )
        else:
            self.logger.info('nothing to close out because of no oustanding positions....'.format(date.strftime('%Y-%m-%d %H:%M:%S')))

    def derive_grid_center_px(
            self,
            data: DataFrame,   
            ts_prop: TS_PROP,         
        ) -> tuple[float, tuple, GRID_TYPE]:
        '''
            Calculate the grid center price
                - for mean reverting, we use current price as grid center price
                - for momentum, we use a current price +- x * vol * vol_grid_scale to follow the price momentum.
                  meanwhile, there is also a extra momentum filter to make sure there is short-term momentum
                
            data:       the input data to execute()
            ts_prop     price-series property
        '''
        current_px = data['Open'] if self.is_backtest() else data['Close']                
        close_sma = data['close_sma']
        current_vol = data['atr']      
        center_px = stoploss = grid_type = None
        
        #if ts_prop == TS_PROP.MEAN_REVERT:            
            # make sure the current price are same as moving average
        if current_px > close_sma - current_vol * self.vol_grid_scale and current_px < close_sma + current_vol * self.vol_grid_scale:
            center_px = current_px
            stoploss = (center_px - (self.grid_size + self.vol_stoploss_scale) * current_vol * self.vol_grid_scale, center_px + (self.grid_size + self.vol_stoploss_scale) * current_vol * self.vol_grid_scale)
            grid_type = GRID_TYPE.MEAN_REVERT

        #elif ts_prop == TS_PROP.MOMENTUM:    
        # conservative approach on momentum filters
        elif current_px > close_sma and current_px > data['High_t5'] and data['Low_t5'] > data['High_t10']:                            
            '''
                Momentum Up Order
                    Upper Bound = same as mean revert. vol_grid_scale references to center price
                    Lower Bound = need to make sure it won't trigger immediately. reference to 
                        1. current price and 
                        2. vol_stoploss_scale => this is derivation tolerance from the grid boundary
            '''
            center_px = current_px + (self.grid_size+1) * current_vol * self.vol_grid_scale                      
            stoploss = (
                current_px - self.vol_stoploss_scale * current_vol * self.vol_grid_scale,
                center_px + (self.grid_size + self.vol_stoploss_scale) * current_vol * self.vol_grid_scale
            )                
            grid_type = GRID_TYPE.MOMENTUM_UP                

        elif current_px < close_sma and current_px < data['Low_t5'] and data['High_t5'] < data['Low_t10']:
            '''
                Momentum Down Order
                    Upper Bound = need to make sure it won't trigger immediately. reference to                         
                        1. current price and 
                        2. vol_stoploss_scale => this is derivation tolerance from the grid boundary
                    Lower Bound = same as mean revert. vol_grid_scale references to center price
            '''
            center_px = current_px - (self.grid_size+1) * current_vol * self.vol_grid_scale
            stoploss = (
                center_px - (self.grid_size + self.vol_stoploss_scale) * current_vol * self.vol_grid_scale,
                current_px + self.vol_stoploss_scale * current_vol * self.vol_grid_scale
            )
            grid_type = GRID_TYPE.MOMENTUM_DOWN

        return center_px, stoploss, grid_type

    def place_grid_order(
            self,            
            center_px:float,
            current_px: float,
            current_vol:float,
            grid_type: GRID_TYPE,
            date:datetime,             
        ):
        '''
            Generate Grid Orders based on current price. Meanwhile also return StopLoss for the Grid
            current_px:     neutral price, center of the grids
            current_vol:    the historical volatiltiy used to determine the grid spacing
            ts_prop:        current price time-series porperties (only used for logging)
            date:           only used for backtest
        '''               

        grid_type_char = self.grid_type_to_char[grid_type.name]
        grid_scales = list(range(-self.grid_size,self.grid_size+1,1))      
        grid_space = current_vol * self.vol_grid_scale

        grid_prices = [center_px + x * grid_space for x in grid_scales]       
        self.grid_id += 1     
        self.grid_type = grid_type

        # grid quantity
        quantity = self.position_size / center_px
        quantity = round(quantity, self.qty_decimal)                
        self.logger.info('creating {} {} grid orders of {} {} at grid center price {} with grid space {}....'.format(self.grid_size * 2, grid_type.name, quantity, self.instrument, round(center_px, self.price_decimal), round(grid_space, self.price_decimal)))
        
        for i, px in enumerate(grid_prices):
            if grid_scales[i] == 0:
                continue

            side = 'SELL' if grid_scales[i] > 0 else 'BUY'  
            order_id = f'{self.__str__()}_gridid{self.grid_id}_{grid_type_char}_grid{i}'            
            px = round(px, self.price_decimal)

            stopPrice = None
            order_type = 'LIMIT'

            '''
                for MOMENTUM UP grid orders, we need to use STOP_LOSS_LIMIT buy orders so that it won't trigger the LIMIT order immediately.
                for MOMENTUM DOWN grid orders, we need to use STOP_LOSS_LIMIT sell orders so that it won't trigger the LIMIT order immediately.            
            '''            
            if grid_type == GRID_TYPE.MOMENTUM_UP and side == 'BUY' and px > current_px:
                order_type = 'STOP_LOSS_LIMIT'
                stopPrice = px

            elif grid_type == GRID_TYPE.MOMENTUM_DOWN and side == 'SELL' and px < current_px:
                order_type = 'STOP_LOSS_LIMIT'
                stopPrice = px                        

            # place grid orders            
            self.executor.place_order(
                instrument=self.instrument,
                side=side,
                order_type=order_type,
                timeInForce='GTC',
                quantity=quantity,
                price=px, 
                stopPrice=stopPrice,
                date=date,          # only used for backtest      
                order_id=order_id
            )             

    def get_all_orders(self, 
                       query_all: bool = False,
                       trade_details: bool = False,     
                       limit: int = 1000,  
                       start_date: datetime = None,
                       end_date: datetime = None,                                       
                       ) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)
            query_all:      True if we want to get all orders. Otherwise, make one request to executor (e.g. Binance only return 1000 orders)
            trade_details:  True if we want to add trade details (e.g. fill price, commission etc....)
        '''
        df_orders = super().get_all_orders(query_all=query_all, trade_details=trade_details, limit=limit, start_date=start_date, end_date=end_date)
                
        # some manual adjusting on orders id
        df_orders['clientOrderId'] = np.where(df_orders['orderId'] == 249865056, 'grid_SOLFDUSDv1_gridid4_MD_close', df_orders['clientOrderId'])

        # find the grid_id of orders
        df_orders['grid_id'] = df_orders['clientOrderId'].apply(lambda x: int(re.search(r"(?<=gridid)\d+(?=_)", x)[0]))

        # find the grid trade type (grid/stoploss/close)
        df_orders['grid_tt'] = df_orders['clientOrderId'].apply(lambda x: x.split('_')[-1]).apply(lambda x: re.sub(r'[0-9]', '', x))        

        # find the grid type (MR/MU/MD)        
        def grid_type_find(x):
            try:
                grid_type = re.search(r"gridid\d+_\w\w_\w+", x)[0].split('_')[1]
                grid_type = self.grid_char_to_type[grid_type]
                return grid_type
            except:
                return ''              
        df_orders['grid_type'] = df_orders['clientOrderId'].apply(grid_type_find)          

        return df_orders
    
    def run(self):
        '''
            Actual function to exectue the strategy repeatedly
        '''
        super().run(lookback='12 Hours Ago')        