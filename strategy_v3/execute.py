import pandas as pd
import warnings
import sys
import json
import traceback
from pandas.core.frame import DataFrame
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

# /usr/local/bin/python3 /Users/lok419/Desktop/JupyterLab/Trading/strategy_v3/execute.py SOLv1

def sanity_check_data(df: DataFrame, data: dict):
    '''
        Sanity check the input data and make sure data are latest
    '''
    dt_now = pd.to_datetime(datetime.now(tz=ZoneInfo("HongKong")))
    interval = df['Date'].diff().iloc[-1].seconds
    since_last = (dt_now - data['Date']).seconds
    assert since_last <= interval

def update_strategy_params(strategy: GridArithmeticStrategy, strategy_setup: ExecuteSetup):
    '''
        This allow user to update the strategy parameters on the fly by changing the execute_setup.json        
    '''
    strategy_params = strategy_setup.read()
    for key, value in strategy_params.items():
        value_org = getattr(strategy, key)        
        # Exceptional case on status as this is an ENUM but we stored as string, we need cast it to string before comparison
        value_org = value_org.name if key == 'status' else value_org
        
        if value != value_org:
            strategy.logger.info(f'update {key} from {value_org} to {value}')
            setattr(strategy, key, value)

if __name__ == '__main__':    

    strategy_id = sys.argv[1]    
    strategy_setup = ExecuteSetup(strategy_id)
    strategy_params = strategy_setup.read()    
    
    strategy = GridArithmeticStrategy(**strategy_params)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)    
    
    try:
        while True:    
            try:                
                update_strategy_params(strategy, strategy_setup)
                                
                # 360 data points
                strategy.load_data('1 Days Ago')                                
                df = strategy.df
                data = df.iloc[-1]                
                sanity_check_data(df, data)            
                strategy.execute(data)    

                sleep(60)

            except Timeout as e:
                strategy.logger.error(e)        
                strategy.logger.error('handled explicitly. retring....')

            except BinanceAPIException as e:
                strategy.logger.error(e)                
                if e.code == '-1021':
                    strategy.logger.error('handled explicitly. retring....')
                    sleep(30)
                else:
                    raise(e)                                

    except KeyboardInterrupt as e:  
        traceback.print_exc()      
        strategy.logger.error(e)        
        strategy.cancel_all_orders()
        strategy.close_out_positions()

    except Exception as e:
        traceback.print_exc()
        strategy.logger.error(e)        
        strategy.cancel_all_orders()
        strategy.close_out_positions()