import pandas as pd
import warnings
import sys
import json
from strategy_v3.Strategy import GridArithmeticStrategy
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.ExecuteSetup import ExecuteSetup
from datetime import datetime
from zoneinfo import ZoneInfo
from time import sleep
from requests.exceptions import Timeout
from binance.exceptions import BinanceAPIException 

warnings.filterwarnings('ignore')

if __name__ == '__main__':    

    strategy_id = sys.argv[0]    
    strategy_setup = ExecuteSetup(strategy_id)
    strategy_params = strategy_setup.read()
    
    strategy = GridArithmeticStrategy(**strategy_params)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)
    
    try:
        while True:    
            try:
                strategy.load_data('1 Days Ago')
                df = strategy.df.copy()
                data = df.iloc[-1]
                dt_now = pd.to_datetime(datetime.now(tz=ZoneInfo("HongKong")))
                interval = df['Date'].diff().iloc[-1].seconds
                since_last = (dt_now - data['Date']).seconds
                assert since_last <= interval
                strategy.execute(data)        
                sleep(60)
            except Timeout as e:
                strategy.logger.error(e)        
                strategy.logger.error('handled explicitly. retring....')        
            except BinanceAPIException as e:
                strategy.logger.error(e)
                if e.code == '-1021':
                    sleep(30)
                else:
                    raise(e)                                

    except KeyboardInterrupt as e:        
        strategy.logger.error(e)        
        strategy.cancel_all_orders()
        strategy.close_out_positions()

    except Exception as e:
        strategy.logger.error(e)        
        strategy.cancel_all_orders()
        strategy.close_out_positions()