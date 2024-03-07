from account import Binance
from strategy_v3.Executor import ExecutorModel
from datetime import datetime
from pandas.core.frame import DataFrame
import pandas as pd
import numpy as np
import math

class ExecutorBacktest(ExecutorModel):

    def __init__(self):        
        self.orders = pd.DataFrame(columns=['symbol', 'clientOrderId', 'price', 'fill_price', 'stopPrice', 'origQty', 'executedQty', 'status', 'side', 'type', 'timeInForce', 'updateTime', 'time', 'orderId'])
        self.order_counter = 0

    def set_logger(self, logger):
        self.logger = logger

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

        # Assume we execute at whatever price for market orders        
        status = 'FILLED' if order_type == 'MARKET' else 'NEW'
        executedQty = quantity if order_type == 'MARKET' else 0
        fill_price = price if order_type == 'MARKET' else math.nan

        order_dict = {
            'symbol': instrument,
            'clientOrderId': order_id,
            'price': price,
            'stopPrice': stopPrice,
            'fill_price': fill_price,
            'origQty': quantity,
            'executedQty': executedQty,
            'status': status,
            'side': side,
            'type': order_type,
            'timeInForce': timeInForce,
            'updateTime': date,
            'time': date,
            'orderId': self.order_counter,
        }
        self.order_counter += 1
        self.orders = pd.concat([self.orders, pd.DataFrame([order_dict])])    

    def cancel_order(self, 
                     instrument:str, 
                     df_orders:DataFrame):
        
        order_id = df_orders['orderId'].values
        self.orders['status'] = np.where(self.orders['orderId'].isin(order_id), 'CANCELED', self.orders['status'])        
        
    def get_all_orders(self, 
                       instrument:str, 
                       **params) -> DataFrame:
        df_orders = self.orders[self.orders['symbol'] == instrument]        
        df_orders['NetExecutedQty'] = np.where(df_orders['side'] == 'BUY', 1, -1) * df_orders['executedQty']
        return df_orders
    
    def fill_orders(self, 
                    date: datetime, 
                    low: float, 
                    high: float):

        fill_filters = (
            (self.orders['status'] != 'FILLED')&\
            (self.orders['status'] != 'CANCELED')&\
            (self.orders['price'] >= low) &\
            (self.orders['price'] <= high)   
        )        

        filled_num = len(fill_filters[fill_filters == True])
        if filled_num > 0:
            self.logger.info('[{}] filling {} orders...'.format(date.strftime('%Y-%m-%d %H:%M:%S'), filled_num))

        self.orders['status'] = np.where(fill_filters, 'FILLED', self.orders['status'])
        self.orders['executedQty'] = np.where(fill_filters, self.orders['origQty'], self.orders['executedQty'])
        self.orders['updateTime'] = pd.to_datetime(np.where(fill_filters, date, self.orders['updateTime']))    
        self.orders['fill_price'] = np.where(fill_filters, self.orders['price'], self.orders['fill_price'])

    def add_trading_fee(self, instrument:str, df_orders: DataFrame) -> DataFrame:
        '''
            We use the same fee stucture from binance for backtesting
        
            Binance has actual trading fee for all trades, but it is quoted in either cryto or fiat currency which is is hard to interpret in PNL terms.
            Here we just code it ourselves based on binance tiered fee

            MAKER - Limit orders, but GTC LIMIT orders can be both MAKER or TAKER, we just assume all MARKER here
            TAKER - Market orders
        '''        
        fees = Binance().get_trading_fee(instrument=instrument).iloc[0]
        marker_fee = fees['makerCommission']
        taker_fee = fees['takerCommission']
        df_orders['trading_fee'] = np.where(df_orders['type'].str.contains('LIMIT'), marker_fee, taker_fee) * df_orders['executedQty'] * df_orders['price']
        df_orders['trading_fee'] = np.where(df_orders['status'] == 'FILLED', df_orders['trading_fee'], 0)
        df_orders['trading_fee'] = df_orders['trading_fee'].astype(float)

        return df_orders
    
    def get_order_book(self, instrument) -> tuple[DataFrame, DataFrame]:
        pass
