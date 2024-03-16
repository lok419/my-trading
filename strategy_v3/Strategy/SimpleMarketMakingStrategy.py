from datetime import datetime
from pandas.core.frame import DataFrame
from strategy_v3.Strategy import STATUS, StrategyBase, MarketMakingPerformance
import numpy as np
import re
from strategy_v3.Strategy.Constant import TS_PROP

from utils.stats import time_series_hurst_exponent

class SimpleMarketMakingStrategy(StrategyBase, MarketMakingPerformance):    

    def __init__(self, 
                 instrument: str, 
                 interval: str,   
                 refresh_interval: int = 60,
                 vol_lookback: int = 20,    
                 gamma: float = 0.4,                 
                 spread_adv_factor: float = 0.01,
                 target_position: float = 0,                
                 position_size: float = 50,
                 hurst_exp_mr_threshold: float = 0.5,                 
                 hurst_exp_mo_threshold: float = 0.7,
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
            gamma:                  inventory risk aversion factor
            target_position:        target position to keep during market making (e.g. if i have directional views, i want to have target position)
            position_size:          position size of each limit order
            hurst_exp_mr_threshold: Hurst Exponent Ratio threshold for mean reverting
            hurst_exp_mo_threshold: Hurst Exponent Ratio threshold for momentum
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

        self.vol_lookback = vol_lookback
        self.gamma = gamma              
        self.spread_adv_factor = spread_adv_factor
        self.hurst_exp_mr_threshold = hurst_exp_mr_threshold
        self.hurst_exp_mo_threshold = hurst_exp_mo_threshold
        self.target_position = target_position
        self.position_size = position_size  
        self.mm_id = 0

    def __str__(self):
        return 'smm_{}'.format(self.strategy_id)
    
    def set_strategy_id(self, id:str, reload: bool = True):
        '''
            Set strategy id. this is used to identify the orders from same strategy instance
            id:     The strategy name
            reload: True to reload all previous property (e.g. grid_id, grid_type) from latest
        '''
        super().set_strategy_id(id)

        if reload and not self.is_backtest() and self.mm_id == 0:
            df_orders = self.get_all_orders()
            if len(df_orders) > 0:                
                mm_id = df_orders.iloc[-1]['mm_id']                                
                self.mm_id = mm_id

    def get_ts_prop(self, data: dict) -> TS_PROP:
        hurst_exponent = data['hurst_exponent']
        if hurst_exponent < self.hurst_exp_mr_threshold:
            return TS_PROP.MEAN_REVERT        
        elif hurst_exponent > self.hurst_exp_mo_threshold:
            return TS_PROP.MOMENTUM        
        else:
            return TS_PROP.RANDOM

    def execute(self, data):
        '''
            Execute function which is called repeatedly for each tick
        '''
        date = data['Date']
        vol = data['atr']      
        adv = data['adv']
        hurst_exponent = data['hurst_exponent']

        current_position = self.get_current_position()
        ts_prop = self.get_ts_prop(data)

        if ts_prop != TS_PROP.MEAN_REVERT or self.status != STATUS.RUN:    
        #if self.status != STATUS.RUN:    
            self.logger.info('status: {}, ts_prop: {}, hurst_exponent: {:.2f}, inv: {}'.format(self.status.name, ts_prop.name, hurst_exponent, round(current_position, self.qty_decimal)))
            self.cancel_all_orders(limit=50, silence=True)            
            if round(abs(current_position), self.qty_decimal) != 0:                                
                self.close_out_positions()        
            return
        
        df_bids, df_asks = self.get_order_book()    
        r, spread, order_bid, order_ask, mid_px = self.derive_bid_ask_order(current_position, df_bids, df_asks, adv, vol)        

        self.cancel_all_orders(limit=50, silence=True)
        self.logger.info('status: {}, ts_prop: {}, hurst_exponent: {:.2f}, inv: {}, mid: {}, r: {}, spread: {}, vol: {}, adv: {}. creating new bid ask orders.....'.format(
            self.status.name, ts_prop.name, hurst_exponent, 
            round(current_position, self.qty_decimal), 
            round(mid_px, self.price_decimal), 
            round(r, self.price_decimal), 
            round(spread, self.price_decimal),
            round(vol, self.price_decimal), 
            round(adv, self.qty_decimal)
        ))             
        self.place_bid_ask_order([order_bid], [order_ask], mid_px, date)

    def run(self):
        '''
            Actual function to execute the strategy repeatedly
        '''        
        super().run(lookback='2 hours ago', refresh_interval=self.refresh_interval)

    def derive_bid_ask_order(self, current, df_bids, df_asks, adv, vol):
        '''
            Derive the bid ask price of the maker LIMIT order based on Stoikov Maket Making Model

            1. Reservation Price which determines the mid price of the bid-ask order, it is determined by
                - distance between current position and target position
                - inventory risk (volatility / inventory risk aversion factor)
                - time until market close (NOT APPLICABLE FOR CRYPTO)  

            2. Spread - spread is aimed to capture x% of the ADV
                - flow target = x% * ADV
                - spread_bid = volume between best bid to this bid such that volume = flow target / 2                
                - spread_ask = volume between best ask to this ask such that volume = flow target / 2                
                - target_spread = spread_ask - spread_bid
                - Basically this is the spreads expected to capture x% of ADV                
        '''
        flow_target = adv * self.spread_adv_factor
        spread_ask = df_asks[df_asks['quantity_cum'] > flow_target/2]['price'].min()
        spread_bid = df_bids[df_bids['quantity_cum'] > flow_target/2]['price'].max()        
        spread = spread_ask - spread_bid  
        mid_px = (df_asks.iloc[0]['price'] + df_bids.iloc[0]['price']) / 2

        r = mid_px - (current - self.target_position) * self.gamma * vol ** 2
        order_bid = r - spread/2
        order_ask = r + spread/2

        return r, spread, order_bid, order_ask, mid_px

    def place_bid_ask_order(self, 
                            bids: list[float], 
                            asks: list[float], 
                            mid_px: float,
                            date: datetime):
        '''
            Place the optimal bid-ask spread orders        
            bids:  List of bid orders
            asks:  List of ask orders
            date:  date of order (only used for backtest)
        '''

        quantity = self.position_size / mid_px
        quantity = round(quantity, self.qty_decimal)        
        self.mm_id += 1        
        
        i = 0                    
        for bid in bids:                
            order_id = f'{self.__str__()}_id{self.mm_id}_mm{i}'
            bid = round(bid, self.price_decimal)
            self.executor.place_order(
                instrument=self.instrument,
                side='BUY',
                order_type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=bid,      
                stopPrice=None,           
                date=date,          # only used for backtest      
                order_id=order_id
            ) 
            i += 1 

        for ask in asks:
            order_id = f'{self.__str__()}_id{self.mm_id}_mm{i}'
            ask = round(ask, self.price_decimal)
            self.executor.place_order(
                instrument=self.instrument,
                side='SELL',
                order_type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=ask,    
                stopPrice=None,             
                date=date,          # only used for backtest      
                order_id=order_id
            ) 
            i += 1               

    def load_data(self, lookback:str|datetime, lookback_end:str|datetime=None) -> DataFrame:        
        '''
            Load all hisotorical price data
        '''
        df = super().load_data(lookback, lookback_end) 
        df['close_std'] = df['Close'].rolling(self.vol_lookback).std().shift(1)
        df['close_sma'] = df['Close'].rolling(self.vol_lookback).mean().shift(1)
        df['adv'] = df['Volume'].rolling(self.vol_lookback).mean().shift(1)
        df["hurst_exponent"] = df["Close"].rolling(100).apply(lambda x: time_series_hurst_exponent(x)).shift(1)

        # Average True Range
        df['tr'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1)))
        df['atr'] = df['tr'].rolling(self.vol_lookback).mean().shift(1)
        self.df = df
    
    def get_all_orders(self, 
                       query_all: bool = False,
                       trade_details: bool = False,     
                       limit: int = 1000,                               
                       ) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)
            query_all:      True if we want to get all orders. Otherwise, make one request to executor (e.g. Binance only return 1000 orders)
            trade_details:  True if we want to add trade details (e.g. fill price, commission etc....)
        '''

        df_orders = super().get_all_orders(query_all=query_all, trade_details=trade_details, limit=limit)   

        # find the grid_id of orders
        df_orders['mm_id'] = df_orders['clientOrderId'].apply(lambda x: int(re.search(r"(?<=id)\d+(?=_)", x)[0]))

        # find the grid trade type (grid/stoploss/close)
        df_orders['mm_tt'] = df_orders['clientOrderId'].apply(lambda x: x.split('_')[-1]).apply(lambda x: re.sub(r'[0-9]', '', x))             

        return df_orders      
    
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
            order_id = order_id = f'{self.__str__()}_id{self.mm_id}_{type}'
            super().close_out_positions(type=type, price=price, order_id=order_id, date=date)