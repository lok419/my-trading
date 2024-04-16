from time import sleep
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Executor import ExecutorModel, ExecutorBacktest
from strategy_v3.DataLoader import DataLoaderModel
from strategy_v3.Strategy import STATUS, StrategyModel
from datetime import datetime, timedelta
from utils.logging import get_logger
from pandas.core.frame import DataFrame
from zoneinfo import ZoneInfo
from requests.exceptions import Timeout, ConnectionError
from binance.exceptions import BinanceAPIException
from strategy_v3.Misc import CustomException
import pandas as pd
import random
import traceback
import os


class StrategyBase(StrategyModel):    

    def __init__(self, 
                 instrument:str, 
                 interval:str,        
                 refresh_interval:int = 60,            
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
            price_decimal:          rounding decimal of price
            qty_decimal:            rounding decimal of quantity
            status:                 user can set status to control the strategy behavior
            start_date:             indicate the start time of the strategy so that we can extract the whole performance history of a strategy. The time is based on HongKong Time
            verbose:                True to print the log message
        '''
        self.instrument = instrument
        self.interval = interval       
        self.refresh_interval = refresh_interval       
        self.qty_decimal = qty_decimal
        self.price_decimal = price_decimal  

        self.timezone = ZoneInfo('HongKong')
        self.start_date = start_date        
        
        self.executor = None
        self.data_loader = None

        # strategy_id is used to identify the strategy from list of orders
        self.strategy_id = str(round(random.random() * 1e6))
        self.logger = get_logger(self.__str__())
        self.status = status

        if not verbose:
            self.logger.setLevel('CRITICAL')        

        # 5m -> 5mins for round function
        self.interval_round = self.interval + 'in' if self.interval.endswith('m') else self.interval
        self.interval_min = int(self.interval.replace('m', ''))
        self.execute_start_time = self.get_current_time()        

    def __str__(self):
        return '{}_{}'.format("".join([x for x in self.__class__.__name__ if x.isupper()]), self.strategy_id)
    
    @property
    def status(self) -> STATUS:
        return self._status
    
    @status.setter
    def status(self, status: str|STATUS):        
        if type(status) is STATUS:
            self._status = status
        else:
            try:
                self._status = STATUS._member_map_[status]
            except:
                self.logger.error(f'unknown status {status}...')    

    @property
    def start_date(self) -> datetime:
        return self._start_date
    
    @start_date.setter
    def start_date(self, start_date: str|datetime):
        if type(start_date) is datetime:
            self._start_date = start_date.astimezone(tz=self.timezone)
        else:
            try:
                self._start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S').astimezone(tz=self.timezone) 
            except Exception as e:                            
                self._start_date = None                    

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

    def set_strategy_id(self, 
                        strategy_id :str,                        
                        ):
        '''
            Set strategy id. this is used to identify the orders from same strategy instance
            strategy_id: The strategy name            
        '''
        self.strategy_id = strategy_id
        self.logger.name = self.__str__()                      
        self.log_path = os.path.dirname(__file__) + f'/log/{strategy_id}.h5'  

    def is_backtest(self):
        return type(self.executor) is ExecutorBacktest
    
    def get_current_time(self) -> datetime:
        return pd.to_datetime(datetime.now(tz=self.timezone))

    def load_data(self, lookback:str|datetime, lookback_end:str|datetime=None) -> DataFrame:
        if lookback is None or lookback == '':
            lookback = self.start_date.strftime('%Y-%m-%d %H:%M:%S%z')

        df = self.data_loader.load_price_data(self.instrument, self.interval, lookback, lookback_end=lookback_end)
        self.df = df
        return self.df

    def get_order_book(self, limit: float=1000) -> tuple[DataFrame, DataFrame]:
        '''
            Load order book data
        '''  
        self.df_bids, self.df_asks = self.executor.get_order_book(self.instrument, limit=limit) 
        return self.df_bids, self.df_asks
    
    def get_aggregate_trades(self, start_date) -> tuple[DataFrame, DataFrame]:
        '''
            Get market aggregate trades
        '''
        df_trades_bid, df_trades_ask = self.executor.get_aggregate_trades(self.instrument, start_date=start_date)
        return df_trades_bid, df_trades_ask 

    def execute(self, data):
        pass    

    def run(self):
        pass

    def close_out_positions(self,                                                         
                            type:str = 'close',
                            price: float = None,
                            order_id: str = None,                         
                            date:datetime = None,                             
                            ):
        '''
            Generic function to close out all outstanding positions

            type:  reason of the close out. Either stoploss or close (end of strategy)
            price: only for backtest, MARKET ORDER does not need price
            date:  only used for backtest
            force: force to close out entire position (usually done manually)
        '''
        filled_net_qty = self.get_current_position()

        if abs(filled_net_qty) > 0:
            self.logger.info('closing out net position of {} {}....'.format(filled_net_qty, self.instrument))

            if price is not None:
                price = round(price, self.price_decimal)
        
            if order_id is None:
                order_id = f'{self.__str__()}_{type}'

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
            self.logger.info('nothing to close out because of no oustanding positions....')
        
    
    def is_delta_neutral(self) -> bool:
        '''
            Check if delta neutral by querying all orders
        '''
        net_pos = self.get_current_position()
        return abs(net_pos) == 0
    
    def get_current_position(self) -> float:
        '''
            Get current net position
        '''
        all_orders = self.get_all_orders()    
        filled = all_orders[all_orders['status'] == 'FILLED']
        filled_net_qty = filled['NetExecutedQty'].sum()   
        filled_net_qty = round(filled_net_qty, self.qty_decimal)
        return filled_net_qty
                        
    def get_all_orders(self,                        
                       trade_details: bool = False,     
                       limit: int = 1000,     
                       start_date: datetime = None,
                       end_date: datetime = None
                       ) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)            
            trade_details:  True if we want to add trade details (e.g. fill price, commission etc....)
            limit:          number of orders per query
            start_date:     query start date of the orders
            end_date:       query end date of the orders
        '''
        start_date = start_date if start_date is not None else self.get_current_time().floor('1d')
        end_date = end_date if end_date is not None else datetime(2100,1,1, tzinfo=self.timezone)
        
        df_orders = self.executor.get_all_orders(self.instrument, trade_details=trade_details, limit=limit, start_date=start_date, end_date=end_date)
        df_orders = df_orders[df_orders['clientOrderId'].str.startswith(self.__str__())]
        return df_orders        
    
    def cancel_all_orders(self, limit: int=1000, silence: bool=False):
        '''
            Cancel all orders created traded by this object

            Limit:  orders limit when we search the orders
        '''    
        if not silence:                            
            self.logger.info('cancelling all orders.....')
            
        df_orders = self.get_all_orders(limit = limit)
        df_orders = df_orders[df_orders['status'] != 'FILLED']                
        df_orders = df_orders[df_orders['status'] != 'CANCELED']
        self.executor.cancel_order(self.instrument, df_orders)    

    def update_strategy_params(self):                  
        strategy_setup = ExecuteSetup(self.strategy_id)

        strategy_params = strategy_setup.read()
        for key, value in strategy_params.items():
            value_org = getattr(self, key)                

            # Exceptional case on status as this is an ENUM but we stored as string, we need cast it to string before comparison
            value_org = value_org.name if key == 'status' else value_org
            value_org = value_org.strftime('%Y-%m-%d %H:%M:%S') if type(value_org) == datetime else value_org
            
            if value != value_org:
                self.logger.info(f'update {key} from {value_org} to {value}')
                setattr(self, key, value)

    def sanity_check_data(self, df: DataFrame, data: dict):
        '''
            Sanity check the input data and make sure data are latest
        '''
        dt_now = self.get_current_time()
        interval = df['Date'].diff().iloc[-1].seconds
        since_last = (dt_now - data['Date']).seconds

        if since_last >= interval:
            raise CustomException(f'data last updated time is more than {interval}')
        
    def is_close_period(self):
        '''
            Strategy close period            
            Every night 23:55 - 00:05, we close out all positions and the pnl/orders for next day will be referenced to this.
            Because of the number of orders, sometime it is confused when we reference to an time where the market making is in process, and the pnl/orders wil be mess-up
        '''
        time = self.get_current_time()
        return (time.hour == 23 and time.minute >= 55) or (time.hour == 0 and time.minute <= 5)

    def run(self, lookback:str):
        '''
            Actual function to execute the strategy repeatedly
        '''
        try:
            while True:    
                try:                    
                    self.update_strategy_params()
                    self.load_data(lookback)                

                    df = self.df
                    data = self.df.iloc[-1]                
                    self.sanity_check_data(df, data)

                    if self.is_close_period():
                        self.logger.info('strategy close period......')
                        self.cancel_all_orders(silence=True)
                        self.close_out_positions()
                    else:
                        self.execute(data)    

                    sleep(self.refresh_interval)

                except Timeout as e:
                    traceback.print_exception(e)
                    self.logger.error(e)        
                    self.logger.error('handled explicitly. retring....')

                except ConnectionError as e:
                    traceback.print_exception(e)
                    self.logger.error(e)     
                    self.logger.error('retrying')
                    sleep(60)

                except BinanceAPIException as e:
                    traceback.print_exception(e)
                    self.logger.error(e)                
                    if e.code == -1021:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
                    elif e.code == -1001:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
                    elif e.code == -1003:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
                    elif e.code == -1099:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
                    else:
                        raise(e)    

                except IndexError as e:
                    traceback.print_exception(e)
                    if 'MarketMakingStrategy' in str(self.__class__):
                        self.logger.error('handled explicitly. retring....')
                        sleep(self.refresh_interval)
                    else:
                        raise(e)
                                                                                                                  
                except CustomException as e:
                    traceback.print_exception(e)
                    self.logger.error(e)    
                    self.logger.error('retrying.....')

        except KeyboardInterrupt as e:    
            traceback.print_exception(e)      
            self.logger.error(e)                
            self.cancel_all_orders()
            self.close_out_positions()

        except Exception as e:   
            traceback.print_exception(e)             
            self.logger.error(e)                
            self.cancel_all_orders()
            self.close_out_positions()

    def log_data(self, data: dict = dict()):
        '''
            Log the strategy data for each execute()
        '''
        raise('Not Implemented.')

    def get_log_data(self) -> DataFrame:
        '''
            Get the strategy data log
        '''
        try:
            return pd.read_hdf(self.log_path, key='log', mode='r')
        except Exception as e:
            self.logger.error(e)
            return None