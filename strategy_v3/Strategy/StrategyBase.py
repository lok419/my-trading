from time import sleep
from strategy_v3.ExecuteSetup import ExecuteSetup
from strategy_v3.Executor import ExecutorModel, ExecutorBacktest
from strategy_v3.DataLoader import DataLoaderModel
from strategy_v3.Strategy import STATUS, StrategyModel
from datetime import datetime
from utils.logging import get_logger
from pandas.core.frame import DataFrame
from zoneinfo import ZoneInfo
from requests.exceptions import Timeout
from binance.exceptions import BinanceAPIException
from strategy_v3.Misc import CustomException
import pandas as pd
import random
import traceback


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
            start_date:             indicate the start time of the strategy so that we can extract the whole performance history of a strategy. By default, the time is based on GMT+8 and converted to UTC
            verbose:                True to print the log message
        '''
        self.instrument = instrument
        self.interval = interval       
        self.refresh_interval = refresh_interval       
        self.qty_decimal = qty_decimal
        self.price_decimal = price_decimal   
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
            self._start_date = start_date
        else:
            try:
                self._start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            except:            
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
                        id:str,                        
                        ):
        '''
            Set strategy id. this is used to identify the orders from same strategy instance

            id:     The strategy name            
        '''
        self.strategy_id = id
        self.logger.name = self.__str__()        

    def is_backtest(self):
        return type(self.executor) is ExecutorBacktest
    
    def get_current_time(self) -> datetime:
        return pd.to_datetime(datetime.now(tz=ZoneInfo("HongKong")))

    def load_data(self, lookback:str|datetime, lookback_end:str|datetime=None) -> DataFrame:
        if lookback is None or lookback == '':
            lookback = self.start_date.strftime('%Y-%m-%d %H:%M:%S')

        df = self.data_loader.load_price_data(self.instrument, self.interval, lookback, lookback_end=lookback_end)
        self.df = df
        return self.df

    def get_order_book(self, limit: float=1000) -> tuple[DataFrame, DataFrame]:
        '''
            Load order book data
        '''  
        self.df_bids, self.df_asks = self.executor.get_order_book(self.instrument, limit=limit) 
        return self.df_bids, self.df_asks

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
        all_orders = self.get_all_orders(query_all=True)    
        filled = all_orders[all_orders['status'] == 'FILLED']
        filled_net_qty = filled['NetExecutedQty'].sum()   
        filled_net_qty = round(filled_net_qty, self.qty_decimal)        

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
        all_orders = self.get_all_orders(query_all=True)    
        filled = all_orders[all_orders['status'] == 'FILLED']
        filled_net_qty = filled['NetExecutedQty'].sum()   
        filled_net_qty = round(filled_net_qty, self.qty_decimal)
        return filled_net_qty
                        
    def get_all_orders(self, 
                       query_all: bool = False,
                       trade_details: bool = False,     
                       limit: int = 1000,                                                                
                       ) -> DataFrame:
        '''
            Get all orders created by this object (using __str__ to determine if created by this object)

            query_all:      True if we want to get all orders. Otherwise, make one request to executor (e.g. Binance only return 1000 orders)
            trade_details:  True if we want to add trade details (e.g. fill price, commission etc....)
            limit:          orders to retrieve
        '''
        df_orders = self.executor.get_all_orders(self.instrument, query_all=query_all, trade_details=trade_details, limit=limit, start_date=self.start_date)
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

                    self.execute(data)    
                    sleep(self.refresh_interval)

                except Timeout as e:
                    traceback.print_exception(e)
                    self.logger.error(e)        
                    self.logger.error('handled explicitly. retring....')

                except BinanceAPIException as e:
                    traceback.print_exception(e)
                    self.logger.error(e)                
                    if e.code == -1021:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
                    elif e.code == -1001:
                        self.logger.error('handled explicitly. retring....')
                        sleep(30)
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