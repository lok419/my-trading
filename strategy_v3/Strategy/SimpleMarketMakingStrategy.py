from datetime import datetime, timedelta
from pandas.core.frame import DataFrame
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Strategy import STATUS, StrategyBase
from strategy_v3.Strategy.Performance import MarketMakingPerformance
from strategy_v3.Strategy.Constant import TS_PROP
from utils.stats import time_series_hurst_exponent
import numpy as np
import re
import pandas as pd


class SimpleMarketMakingStrategy(StrategyBase, MarketMakingPerformance):    

    def __init__(self, 
                 instrument: str, 
                 interval: str,   
                 refresh_interval: int = 60,
                 vol_lookback: int = 20,    
                 gamma: float = 0.4,                 
                 spread_flow_factor: float = 0.1,
                 px_skew_flow_factor: float = 1,
                 target_position: float = 0,                
                 position_size: float = 50,
                 hurst_exp_mr_threshold: float = 0.5,                 
                 hurst_exp_mo_threshold: float = 0.7,
                 cb_interval: int = 30,
                 cb_px_chg_threshold: float = 0.02,                 
                 status: str = STATUS.RUN,
                 start_date: str = None,
                 verbose: bool = True,   
                 comment: str = '',
        ):
        '''
            instrument:             The instrument to trade
            interval:               time interval to trade            
            refresh_interval:       frequency of function execute() is called (in mintues)
            gamma:                  inventory risk aversion factor
            spread_flow_factor:     This determines the spread based on market order flow.
            px_skew_flow_factor:    This determines the mid-price skewness based on market order flow.            
            target_position:        target position to keep during market making (e.g. if i have directional views, i want to have target position)
            position_size:          position size of each limit order
            hurst_exp_mr_threshold: Hurst Exponent Ratio threshold for mean reverting
            hurst_exp_mo_threshold: Hurst Exponent Ratio threshold for momentum
            cb_interval:            Circuit Breaker interval before re-starting the strategy
            cb_px_chg_threshold:    The price change threshold for trigger the CB            
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

        self.vol_lookback = vol_lookback
        self.gamma = gamma              
        self.spread_flow_factor = spread_flow_factor
        self.px_skew_flow_factor = px_skew_flow_factor
        self.hurst_exp_mr_threshold = hurst_exp_mr_threshold
        self.hurst_exp_mo_threshold = hurst_exp_mo_threshold
        self.target_position = target_position
        self.position_size = position_size  
        self.mm_id = 0

        self.cb_interval = cb_interval
        self.cb_px_chg_threshold = cb_px_chg_threshold
        self.cb_end_ts = self.get_current_time() - timedelta(minutes=15)        

    def __str__(self):
        return 'smm_{}'.format(self.strategy_id)
    
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

        if reload and not self.is_backtest() and self.mm_id == 0:
            if force_reload_all:
                df_orders = self.get_all_orders(start_date=self.start_date)
            else:
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
        vol = data['close_std']      
        adv = data['adv']        
        hurst_exponent = data['hurst_exponent']

        current_position = self.get_current_position()
        ts_prop = self.get_ts_prop(data)        
        self.cb_check(data)
        
        if ts_prop != TS_PROP.MEAN_REVERT or self.status != STATUS.RUN:
            self.logger.info('status: {}, ts_prop: {}, hurst_exponent: {:.2f}, inv: {}'.format(self.status.name, ts_prop.name, hurst_exponent, round(current_position, self.qty_decimal)))
            self.cancel_all_orders(limit=50, silence=True)            
            if round(abs(current_position), self.qty_decimal) != 0:                                
                self.close_out_positions()        
            return
        
        df_bid, df_ask = self.get_order_book(limit=2000)    
        df_trades_bid, df_trades_ask = self.get_aggregate_trades(start_date=f'{self.refresh_interval} seconds ago')
        r, spread, order_bid, order_ask, mkt_sprd, best_bid, best_ask, mid_px, vwmp, ar_bid, ar_ask, ar_skew, vwmp2 = self.derive_bid_ask_order(current_position, df_bid, df_ask, df_trades_bid, df_trades_ask, adv, vol)                

        self.cancel_all_orders(limit=50, silence=True)
        self.logger.info('status: {}, ts_prop: {}, hurst_exponent: {:.2f}, inv: {}, mid: {}, vwmp: {}, skew: {}, skew_inv: {}, ar_skew: {}, r: {}, spread: {}, vol: {}, adv: {}'.format(
            self.status.name, ts_prop.name, hurst_exponent, 
            round(current_position, self.qty_decimal), 
            round(mid_px, self.price_decimal), 
            round(vwmp, self.price_decimal), 
            round(vwmp - mid_px, self.price_decimal), 
            round(r - vwmp, self.price_decimal), 
            round(ar_skew, self.qty_decimal),
            round(r, self.price_decimal), 
            round(spread, self.price_decimal),
            round(vol, self.price_decimal), 
            round(adv, self.qty_decimal),            
        ))             

        self.place_bid_ask_order([order_bid], [order_ask], mid_px, date)

        data = {
            'vol': vol,
            'adv': adv,
            'hurst_exponent': hurst_exponent,
            'ts_prop': ts_prop.name,
            'inv': current_position,
            'reservation_price': r,
            'mid_price': mid_px,
            'market_spread': mkt_sprd,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'vwmp': vwmp,
            'skew': vwmp - mid_px,            
            'skew_2': vwmp2 - mid_px,            
            'skew_inv': r - vwmp,
            'spread': spread,
            'order_bid': order_bid,
            'order_ask': order_ask,
            'ar_bid': ar_bid,
            'ar_ask': ar_ask,
            'ar_skew': ar_skew,
            'comment': self.comment,
        }
        self.log_data(data)

    def run(self):
        '''
            Actual function to execute the strategy repeatedly
        '''        
        super().run(lookback='2 hours ago')

    def derive_bid_ask_order(self, current, df_bid, df_ask, df_trades_bid, df_trades_ask, adv, vol):
        '''
            Derive the bid ask price of the maker LIMIT order based on Stoikov Maket Making Model

            1. Reservation Price which determines the mid price of the bid-ask order, it is determined by
                - distance between current position and target position
                - inventory risk (volatility / inventory risk aversion factor)
                - time until market close (NOT APPLICABLE FOR CRYPTO)  
                - mid price is replaced by volume weighted mid price

            2. Spread - spread is aimed to capture x% of the ADV
                - flow target = x% * ADV
                - spread_bid = volume between best bid to this bid such that volume = flow target / 2                
                - spread_ask = volume between best ask to this ask such that volume = flow target / 2                
                - target_spread = spread_ask - spread_bid
                - Basically this is the spreads expected to capture x% of ADV                
        '''
        
        # flow_target = adv * self.spread_flow_factor
        # spread_ask = df_ask[df_ask['quantity_cum'] > flow_target/2]['price'].min()
        # spread_bid = df_bid[df_bid['quantity_cum'] > flow_target/2]['price'].max()        
        # spread2 = spread_ask - spread_bid
        
        best_bid = df_bid.iloc[0]['price']
        best_ask = df_ask.iloc[0]['price']        
        mid_px = (best_ask + best_bid)/2
        mkt_sprd = best_ask - best_bid

        # MO arrival rate
        bid_interval = (df_trades_bid['time'].max() - df_trades_bid['time'].min()).seconds
        ask_interval = (df_trades_ask['time'].max() - df_trades_ask['time'].min()).seconds
        ar_bid = df_trades_bid['quantity'].sum()/bid_interval
        ar_ask = df_trades_ask['quantity'].sum()/ask_interval
        ar_skew = ar_ask - ar_bid

        # just for reference
        ar_skew_sum = ar_skew * self.refresh_interval
        if ar_skew > 0:
            vwmp_skew = df_ask[df_ask['quantity_cum'] > abs(ar_skew_sum)].iloc[0]['price'] - best_ask
        else:
            vwmp_skew = df_bid[df_bid['quantity_cum'] > abs(ar_skew_sum)].iloc[0]['price'] - best_bid
        vwmp2 = mid_px + vwmp_skew

        # MO arrival to LOB
        ar_bid_next = ar_bid * self.refresh_interval
        ar_ask_next = ar_ask * self.refresh_interval
        bid_chg = df_bid[df_bid['quantity_cum'] > abs(ar_bid_next)].iloc[0]['price'] - best_bid
        ask_chg = df_ask[df_ask['quantity_cum'] > abs(ar_ask_next)].iloc[0]['price'] - best_ask        
        vwmp_skew = (ask_chg + bid_chg) * self.px_skew_flow_factor
        vwmp = mid_px + vwmp_skew

        # Target Spread
        spread = (mkt_sprd + ask_chg - bid_chg) * self.spread_flow_factor

        r = vwmp - (current - self.target_position) * self.gamma * vol ** 2
        order_bid = min(r - spread/2, best_bid)
        order_ask = max(r + spread/2, best_ask)
        spread = order_ask - order_bid

        return r, spread, order_bid, order_ask, mkt_sprd, best_bid, best_ask, mid_px, vwmp, ar_bid, ar_ask, ar_skew, vwmp2

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
        df['close_chg'] = df['Close'].diff().shift(1)

        df['close_chg_pct_t1'] = (df['Close'] / df['Close'].shift(1) - 1).shift(1)
        df['close_chg_pct_t2'] = (df['Close'] / df['Close'].shift(2) - 1).shift(1)
        df['close_chg_pct_t3'] = (df['Close'] / df['Close'].shift(3) - 1).shift(1)
        df['close_chg_pct_t4'] = (df['Close'] / df['Close'].shift(4) - 1).shift(1)
        df['close_chg_pct_t5'] = (df['Close'] / df['Close'].shift(5) - 1).shift(1)

        df['adv'] = df['Volume'].rolling(self.vol_lookback).mean().shift(1)
        df["hurst_exponent"] = df["Close"].rolling(100).apply(lambda x: time_series_hurst_exponent(x)).shift(1)

        # Average True Range
        df['tr'] = np.maximum(df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift(1)), np.abs(df['Low'] - df['Close'].shift(1)))
        df['atr'] = df['tr'].rolling(self.vol_lookback).mean().shift(1)
        self.df = df

    def cb_check(self, data: dict):
        '''
            Function to determine if trigger the circuit breaker
            The MM strategy often be destroyed by the adverse market change, we want to freeze the strategy whenever it happens
        '''
        ts = self.get_current_time()     

        chg_t1 = abs(data['close_chg_pct_t1'])
        chg_t2 = abs(data['close_chg_pct_t2'])
        chg_t3 = abs(data['close_chg_pct_t3'])
        chg_t4 = abs(data['close_chg_pct_t4'])
        chg_t5 = abs(data['close_chg_pct_t5'])
        chg_max = max(chg_t1, chg_t2, chg_t3, chg_t4, chg_t5)

        if abs(chg_max) > self.cb_px_chg_threshold:
            self.cb_end_ts = self.get_current_time() + timedelta(minutes=self.cb_interval)            
            if self.status != STATUS.CIRCUIT_BREAKER:
                self.prev_status = self.status
                self.status = STATUS.CIRCUIT_BREAKER
                ExecuteSetup(self.strategy_id).update("status", STATUS.CIRCUIT_BREAKER.name, type(self))     

        elif ts > self.cb_end_ts and self.status == STATUS.CIRCUIT_BREAKER:
            self.status = self.prev_status
            ExecuteSetup(self.strategy_id).update("status", self.status.name, type(self))                  
    
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

        # find the grid_id of orders
        df_orders['mm_id'] = df_orders['clientOrderId'].apply(lambda x: int(re.search(r"(?<=id)\d+(?=_)", x)[0]))

        # find the grid trade type (grid/stoploss/close)
        df_orders['mm_tt'] = df_orders['clientOrderId'].apply(lambda x: x.split('_')[-1]).apply(lambda x: re.sub(r'[0-9]', '', x))             

        return df_orders     
    
    def close_out_positions(self,                                                         
                            type:str = 'close',
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
            order_id = order_id = f'{self.__str__()}_id{self.mm_id}_{type}'
            super().close_out_positions(type=type, price=price, order_id=order_id, date=date, offset=offset)

    def log_data(self, data: dict = dict()):
        '''
            Log the strategy data for each execute()
        '''
        try:
            log = {}
            # strategy parameters            
            log['date'] = self.get_current_time()
            log['strategy_id'] = self.strategy_id
            log['interval'] = self.interval
            log['refresh_interval'] = self.refresh_interval
            log['gamma'] = self.gamma
            log['spread_flow_factor'] = self.spread_flow_factor
            log['px_skew_flow_factor'] = self.px_skew_flow_factor
            log['target_position'] = self.target_position
            log['position_size'] = self.position_size
            log['hurst_exp_mo_threshold'] = self.hurst_exp_mo_threshold
            log['hurst_exp_mr_threshold'] = self.hurst_exp_mr_threshold
            log['status'] = self.status.name   

            # execute data
            log.update(data)
            log = pd.DataFrame([log])

            log_df = self.get_log_data()
            if log_df is not None:
                log_df = pd.concat([log_df, log])
            else:
                log_df = log

            log_df.to_hdf(self.log_path, key='log', format='table')

        except Exception as e:
            self.logger.error(e)