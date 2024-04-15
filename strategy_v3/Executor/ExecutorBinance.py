from strategy_v3.Executor import ExecutorModel
from account import Binance
from datetime import datetime
from binance.exceptions import BinanceAPIException 
from pandas.core.frame import DataFrame
import numpy as np

class ExecutorBinance(ExecutorModel):

    def __init__(self):
        self.binance = Binance()

    def set_logger(self, logger):
        self.logger = logger

    def init_session(func):
        def inner(self, *args, **kwargs):            
            self.__init__()
            return func(self, *args, **kwargs)     
           
        return inner

    @init_session
    def place_order(self,                    
                    instrument:str,
                    side:str,
                    order_type:str,
                    timeInForce:str,
                    quantity:float,
                    price:float,
                    stopPrice: float,                    
                    order_id:str,
                    date:datetime,
                    ):            
        params = dict()
        params['symbol'] = instrument
        params['side'] = side
        params['type'] = order_type
        params['newClientOrderId'] = order_id
        params['quantity'] = np.format_float_positional(quantity)

        if timeInForce is not None:
            params['timeInForce'] = timeInForce

        if price is not None and order_type != 'MARKET':        
            params['price'] = str(price)         

        if stopPrice is not None and order_type == 'STOP_LOSS_LIMIT':
            params['stopPrice'] = str(stopPrice)
             
        try:
            self.binance.place_order(**params)            
        except BinanceAPIException as e:
            self.logger.error(e)            

            if e.code == -2010:
                # APIError(code=-2010): Stop price would trigger immediately. 
                params['type'] = 'LIMIT'
                if 'stopPrice' in params:                    
                    del params['stopPrice']
                self.logger.error('replace with {} {} order at {}.'.format(params['symbol'], params['type'], params['price']))
                self.binance.place_order(**params)
            else:
                raise(e)            
    
    @init_session
    def cancel_order(self, 
                     instrument:str, 
                     df_orders:DataFrame):
        order_id = df_orders['orderId'].values        
        for id in order_id:
            try:
                self.binance.cancel_order(instrument, id)        
            except BinanceAPIException as e:
                self.logger.error(e)
                if e.code == -2011:            
                    self.logger.error(f'error when canceling order {id}...')
                else:
                    raise(e)
        
    @init_session
    def get_all_orders(self, instrument:str, **params):        
        df_orders = self.binance.get_all_orders(instrument, **params)                        
        df_orders['NetExecutedQty'] = np.where(df_orders['side'] == 'BUY', 1, -1) * df_orders['executedQty']
        return df_orders
    
    def fill_orders(self, *args, **kwargs):
        pass

    @init_session
    def add_trading_fee(self, instrument:str, df_orders: DataFrame) -> DataFrame:
        '''
            Binance has actual trading fee for all trades, but it is quoted in either cryto or fiat currency which is is hard to interpret in PNL terms.
            Here we just code it ourselves based on binance tiered fee

            MAKER - Limit orders, but GTC LIMIT orders can be both MAKER or TAKER, we just assume all MARKER here
            TAKER - Market orders
        '''        
        fees = self.binance.get_trading_fee(instrument=instrument).iloc[0]
        marker_fee = fees['makerCommission']
        taker_fee = fees['takerCommission']
        df_orders['trading_fee'] = np.where(df_orders['type'].str.contains('LIMIT'), marker_fee, taker_fee) * df_orders['executedQty'] * df_orders['price']
        df_orders['trading_fee'] = np.where(df_orders['status'] == 'FILLED', df_orders['trading_fee'], 0)
        df_orders['trading_fee'] = df_orders['trading_fee'].astype(float)

        return df_orders
    
    @init_session
    def get_order_book(self, instrument, limit: float=1000) -> tuple[DataFrame, DataFrame]:
        '''
            Get Live Order book from binance
        '''
        return self.binance.get_order_book(instrument=instrument, limit=limit)
    
    @init_session
    def get_aggregate_trades(self, instrument, start_date: datetime|str) -> tuple[DataFrame, DataFrame]:
        '''
            Get aggregated trades
        '''
        return self.binance.get_aggregate_trades(instrument=instrument, start_date=start_date)


        

        


        

