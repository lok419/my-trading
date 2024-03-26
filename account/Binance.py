from binance.client import Client
from utils.credentials import BINANCE_API_KEY, BINANCE_API_SECRET
from pandas.core.frame import DataFrame
from account import AccountModel
from datetime import datetime
from utils.logging import get_logger
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np

class Binance(AccountModel):
    '''
        Ideally we put all the utils from Binance heres
    '''    
    def __init__(self, target_tz:str = 'HongKong'):
        '''
            Binance API has a default timezone of UTC (i.e. GMT+0). For our convenience, we want to convert it to HongKong time (or any deseried time zone)
            target_tz:  timezone to convert for any UTC time from Binance
        '''
        self.client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)                
        self.target_tz = target_tz
        self.logger = get_logger('Binance')

        # bianance API doesn't return empty dataframe if empty, we need to create our own to avoid any error in stratety
        self.default_orders = DataFrame(columns=['symbol', 'clientOrderId', 'price', 'origQty', 'executedQty', 'status', 'side', 'type', 'timeInForce', 'updateTime', 'time', 'orderId'], )        
        self.default_orders['updateTime'] = pd.to_datetime(self.default_orders['updateTime'])
        self.default_orders['time'] = pd.to_datetime(self.default_orders['time'])
        self.default_orders ['updateTime'] = self.default_orders['updateTime'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        self.default_orders ['time'] = self.default_orders['time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)

    def get_client(self) -> Client:
        """
            Get Binance Client object
        """
        return self.client

    def get_all_instruments(self) -> DataFrame:
        """
            Get all listed tickers and their prices
        """
        all_tickers = self.client.get_all_tickers()
        all_tickers = pd.DataFrame(all_tickers)
        return all_tickers
    
    def get_account(self) -> dict:
        """
            Get General Account Summary
        """
        account = self.client.get_account()
        return account
    
    def get_position(self) -> DataFrame:
        """
            Get Currnet Balances in Account
        """
        account = self.client.get_account()
        balances = pd.DataFrame(account['balances'])
        return balances
    
    def get_historical_instrument_price(self, 
                             instrument:str,
                             interval:str=Client.KLINE_INTERVAL_1DAY,
                             start_str:str=None,
                             end_str:str=None,
                             ) -> DataFrame:
        '''
            Get Historical Klines from Binance 
            instrument:  Instrument symbol
            interval:    Interval of the klines. Available options are 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
            start_str:   The start time of the klines   (e.g. "1 day ago UTC", "1 Dec, 2017", "1 Jan, 2018", "2023-01-01")
            end_str:     The end time of the klines     (e.g. "1 day ago UTC", "1 Dec, 2017", "1 Jan, 2018", "2023-01-01")
        '''
        
        klines = self.client.get_historical_klines(symbol=instrument, interval=interval, start_str=start_str, end_str=end_str)
        klines = pd.DataFrame(klines, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])        

        klines['Date'] = pd.to_datetime(klines['Date'], unit='ms')        
        klines['Close Time'] = pd.to_datetime(klines['Close Time'], unit='ms')                

        klines['Date'] = klines['Date'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        klines['Close Time'] = klines['Close Time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)

        # the API return string for all columsn
        num_cols = [col for col in klines.columns if col not in ['Date', 'Close Time']]
        klines[num_cols] = klines[num_cols].astype(float)
        return klines
    
    def get_instrument_price(self, 
                             instrument:str,
                             interval:str=Client.KLINE_INTERVAL_1DAY,
                             ) -> DataFrame:
        '''
            Get Klines from Binance 
            instrument:  Instrument symbol
            interval:    Interval of the klines. Available options are 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
        '''
        
        klines = self.client.get_klines(symbol=instrument, interval=interval)
        klines = pd.DataFrame(klines, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

        klines['Date'] = pd.to_datetime(klines['Date'], unit='ms')        
        klines['Close Time'] = pd.to_datetime(klines['Close Time'], unit='ms')

        klines['Date'] = klines['Date'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        klines['Close Time'] = klines['Close Time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)

        # the API return string for all columsn
        num_cols = [col for col in klines.columns if col not in ['Date', 'Close Time']]
        klines[num_cols] = klines[num_cols].astype(float)
        return klines    
    
    def place_order(self,
                    **params                    
        ) -> dict:
        
        '''
            Place an order via Binance
                symbol:                     required
                side:                       required
                type:                       required
                timeInForce (str):          required if limit order
                quantity (decimal):         required
                quoteOrderQty (decimal):    amount the user wants to spend (when buying) or receive (when selling) of the quote asset, applicable to MARKET orders
                price (str):                required
                newClientOrderId (str)      A unique id for the order. Automatically generated if not sent.
                icebergQty (decimal)        Used with LIMIT, STOP_LOSS_LIMIT, and TAKE_PROFIT_LIMIT to create an iceberg order.
                newOrderRespType (str)      Set the response JSON. ACK, RESULT, or FULL; default: RESULT.
                recvWindow (int) :          the number of milliseconds the request is valid for            

                side:
                    SIDE_BUY = 'BUY'
                    SIDE_SELL = 'SELL'

                order_type:
                    ORDER_TYPE_LIMIT = 'LIMIT'
                    ORDER_TYPE_MARKET = 'MARKET'
                    ORDER_TYPE_STOP_LOSS = 'STOP_LOSS'
                    ORDER_TYPE_STOP_LOSS_LIMIT = 'STOP_LOSS_LIMIT'
                    ORDER_TYPE_TAKE_PROFIT = 'TAKE_PROFIT'
                    ORDER_TYPE_TAKE_PROFIT_LIMIT = 'TAKE_PROFIT_LIMIT'
                    ORDER_TYPE_LIMIT_MAKER = 'LIMIT_MAKER'

                timeInForce:
                    TIME_IN_FORCE_GTC = 'GTC'  # Good till cancelled
                    TIME_IN_FORCE_IOC = 'IOC'  # Immediate or cancel
                    TIME_IN_FORCE_FOK = 'FOK'  # Fill or kill
        '''                    
        order = self.client.create_order(**params)                
        return order
    
    def get_all_orders(self,
                       instrument:str,                           
                       start_date: datetime = datetime(2000,1,1, tzinfo=ZoneInfo('HongKong')),
                       end_date: datetime = datetime(2100,1,1, tzinfo=ZoneInfo('HongKong')),
                       trade_details: bool = False,
                       limit: int = 1000,
                       ) -> DataFrame:
        '''
            All all orders from Binance
            instrument:     symbol of the crypto
            query_all:      given binance only return last 1000 orders. True if we want to query all orders by split
            start_date:     start_date to query
            end_date:       end_date to query
            trade_details:  true to add more trade details (filled px, commission.....)

            Binance API returns max orders of 1000, in order to query all orders, we need to query by batch
        '''                     
        orders = []        
        start_date_ts = int(datetime.timestamp(start_date)) * 1000
        end_date_ts = int(datetime.timestamp(end_date)) * 1000
        date_ts = end_date_ts

        while date_ts >= start_date_ts:
            orders_page = self.client.get_all_orders(symbol=instrument, limit=limit, endTime=date_ts)
            self.logger.debug('Fetching {} orders...'.format(len(orders_page)))

            min_date_ts = min([o['time'] for o in orders_page])
            orders = orders_page + orders                

            if len(orders_page) < 1000:
                break

            if min_date_ts < start_date_ts:
                break

            date_ts = min_date_ts - 1            
             
        orders = pd.DataFrame(orders)

        if len(orders) == 0:
            return self.default_orders

        orders['price'] = orders['price'].astype(float)
        orders['origQty'] = orders['origQty'].astype(float)
        orders['executedQty'] = orders['executedQty'].astype(float)
        orders['cummulativeQuoteQty'] = orders['cummulativeQuoteQty'].astype(float)

        orders['updateTime'] = pd.to_datetime(orders['updateTime'], unit='ms')        
        orders['time'] = pd.to_datetime(orders['time'], unit='ms')        
        orders['workingTime'] = pd.to_datetime(orders['workingTime'], unit='ms')                

        orders['updateTime'] = orders['updateTime'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        orders['time'] = orders['time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        orders['workingTime'] = orders['workingTime'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        orders = orders[orders['time'] >= start_date]
        orders = orders[orders['time'] <= end_date]

        if trade_details:
            '''
                one orderID could have multiple trades, need to calculate the wavg filled price
            '''
            trades = self.get_all_trades(instrument, start_date=start_date, end_date=end_date)            
            trades['trade'] = 1
            trades = trades[['orderId', 'price', 'commission', 'commissionAsset', 'isBuyer', 'isMaker', 'isBestMatch', 'qty', 'trade']]
            trades = trades.rename(columns={'price': 'fill_price'})
            trades['fill_price'] *= trades['qty']
            trades = trades.groupby(['orderId']).agg({
                'fill_price': 'sum',
                'commission': 'sum',
                'commissionAsset': 'first',
                'isBuyer': 'sum',
                'isMaker': 'sum',
                'isBestMatch': 'sum',
                'qty': 'sum',
                'trade': 'sum',
            }).reset_index()
            trades['fill_price'] /= trades['qty']
            trades = trades.drop(columns=['qty'])

            orders = pd.merge(orders, trades, on=['orderId'], how='left', validate='1:1')
        
        return orders
    
    def cancel_order(self,
                     instrument:str,
                     order_id:str,
                     ) -> dict:
        '''
            cancel a order 
        '''

        result = self.client.cancel_order(symbol=instrument, orderId=order_id)
        return result
    
    def get_open_orders(self,
                        instrument:str) -> DataFrame:        
        '''
            get open orders
        '''
        orders = self.client.get_open_orders(symbol=instrument)        
        orders = pd.DataFrame(orders)        

        if len(orders) == 0:
            return self.default_orders

        orders['price'] = orders['price'].astype(float)
        orders['origQty'] = orders['origQty'].astype(float)
        orders['executedQty'] = orders['executedQty'].astype(float)
        orders['cummulativeQuoteQty'] = orders['cummulativeQuoteQty'].astype(float)

        orders['updateTime'] = pd.to_datetime(orders['updateTime'], unit='ms')        
        orders['time'] = pd.to_datetime(orders['time'], unit='ms')        
        orders['workingTime'] = pd.to_datetime(orders['workingTime'], unit='ms')        

        orders['updateTime'] = orders['updateTime'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        orders['time'] = orders['time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)
        orders['workingTime'] = orders['workingTime'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)

        return orders
    
    def cancel_all_orders(self,
                          instrument:str):
        '''
            Cancel all orders
        '''
        orders = self.get_open_orders(instrument)
        order_id = orders['orderId'].values
        for id in order_id:
            self.cancel_order(instrument, id)

    def get_trading_fee(self, instrument:str = None) -> DataFrame:
        '''
            Get all commission from binance (not the actual comission for traded orders)
        '''
        params = dict()
        if instrument is not None:
            params['symbol'] = instrument        

        fees = self.client.get_trade_fee(**params)
        fees = pd.DataFrame(fees)
        fees['makerCommission'] = fees['makerCommission'].astype(float)
        fees['takerCommission'] = fees['takerCommission'].astype(float)
        return fees
    
    def get_all_trades(self, 
                       instrument: str,                                      
                       start_date: datetime = datetime(2000,1,1, tzinfo=ZoneInfo('HongKong')),
                       end_date: datetime = datetime(2100,1,1, tzinfo=ZoneInfo('HongKong')),
        ) -> pd.DataFrame:
        '''
            Get all executed trades
            instrument:     symbol of the crypto
            query_all:      Given binance only return last 1000 orders. True if we want to query all orders by split
            start_date:     start_date to query
            end_date:       end_date to query            

            Binance API returns max orders of 1000, in order to query all orders, we need to query by batch

            OrderID in trades are not unique (i.e. one order can have multiple trades)
        '''
        trades = []        
        start_date_ts = int(datetime.timestamp(start_date)) * 1000
        end_date_ts = int(datetime.timestamp(end_date)) * 1000
        date_ts = end_date_ts

        while date_ts >= start_date_ts:
            trades_page = self.client.get_my_trades(symbol=instrument, limit=1000, endTime=date_ts)
            self.logger.debug('Fetching {} orders...'.format(len(trades_page)))

            min_date_ts = min([o['time'] for o in trades_page])
            trades = trades_page + trades                

            if len(trades_page) < 1000:
                break

            if min_date_ts < start_date_ts:
                break

            date_ts = min_date_ts - 1            
    

        if len(trades) == 0:
            return DataFrame()

        trades = pd.DataFrame(trades)
        trades['time'] = pd.to_datetime(trades['time'], unit='ms')       
        trades['time'] = trades['time'].dt.tz_localize('UTC').dt.tz_convert(self.target_tz)

        trades['price'] = trades['price'].astype(float)	
        trades['qty'] = trades['qty'].astype(float)
        trades['quoteQty'] = trades['quoteQty'].astype(float)
        trades['commission'] = trades['commission'].astype(float)

        trades = trades[trades['time'] >= start_date]
        trades = trades[trades['time'] <= end_date]

        return trades
    
    def get_order_book(self, instrument:str, limit=1000) -> tuple[DataFrame, DataFrame]:
        '''
            Get Live Order Books
        '''
        orders = self.client.get_order_book(symbol=instrument, limit=limit)

        df_bid = pd.DataFrame({'price': list(zip(*orders['bids']))[0], 'quantity': list(zip(*orders['bids']))[1]})
        df_ask = pd.DataFrame({'price': list(zip(*orders['asks']))[0], 'quantity': list(zip(*orders['bids']))[1]})

        df_bid['price'] = df_bid['price'].astype('float')
        df_bid['quantity'] = df_bid['quantity'].astype('float')
        df_bid = df_bid.sort_values('price', ascending=False)
        df_bid['quantity_cum'] = df_bid['quantity'].cumsum()

        df_ask['price'] = df_ask['price'].astype('float')
        df_ask['quantity'] = df_ask['quantity'].astype('float')
        df_ask = df_ask.sort_values('price')
        df_ask['quantity_cum'] = df_ask['quantity'].cumsum()

        return df_bid, df_ask
    
    def get_aggregate_trades(self, instrument:str, start_date:datetime|str) -> tuple[DataFrame, DataFrame]:
        '''
            Get Market Aggregated Trades
        '''        
        agg_trades = self.client.aggregate_trade_iter(symbol=instrument, start_str=start_date)

        trades = [x for x in agg_trades]
        trades = pd.DataFrame(trades)
        trades.columns = ['aggregate tradeId', 'price', 'quantity', 'first tradeId', 'last tradeId', 'time', 'isBuyerMaker', 'isBestMatch']
        trades['time'] = pd.to_datetime(trades['time'], unit='ms')        
        trades['time'] = trades['time'].dt.tz_localize('UTC').dt.tz_convert(Binance().target_tz)
        trades['price'] = trades['price'].astype(float)
        trades['quantity'] = trades['quantity'].astype(float)

        df_trades_bid = trades[trades['isBuyerMaker'] == True]
        df_trades_ask = trades[trades['isBuyerMaker'] == False]

        return df_trades_bid, df_trades_ask
        