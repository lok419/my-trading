from strategy_v3.Executor import ExecutorModel
from account import Binance
from datetime import datetime
import numpy as np
from pandas.core.frame import DataFrame

class ExecutorBinance(ExecutorModel):

    def __init__(self):
        self.binance = Binance()        

    def set_logger(self, logger):
        self.logger = logger

    def place_order(self,                    
                    instrument:str,
                    side:str,
                    order_type:str,
                    timeInForce:str,
                    quantity:float,
                    price:float,
                    order_id:str,
                    date:datetime,
                    ):
        
        params = dict()
        params['symbol'] = instrument
        params['side'] = side
        params['type'] = order_type
        params['newClientOrderId'] = order_id
        params['quantity'] = quantity

        if timeInForce is not None:
            params['timeInForce'] = timeInForce

        if not(price is None or order_type == 'MARKET'):
            params['price'] = str(price)         
             
        self.binance.place_order(**params)
    
    def cancel_order(self, 
                     instrument:str, 
                     df_orders:DataFrame):
        order_id = df_orders['orderId'].values        
        for id in order_id:
            self.binance.cancel_order(instrument, id)        
        
    def get_all_orders(self, instrument:str):        
        df_orders = self.binance.get_all_orders(instrument)                        
        df_orders['NetExecutedQty'] = np.where(df_orders['side'] == 'BUY', 1, -1) * df_orders['executedQty']
        return df_orders
    
    def fill_orders(self, *args, **kwargs):
        pass

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
        df_orders['trading_fee'] = np.where(df_orders['type'] == 'LIMIT', marker_fee, taker_fee) * df_orders['executedQty'] * df_orders['price']
        df_orders['trading_fee'] = np.where(df_orders['status'] == 'FILLED', df_orders['trading_fee'], 0)
        df_orders['trading_fee'] = df_orders['trading_fee'].astype(float)

        return df_orders


        

        


        

