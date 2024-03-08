import pandas as pd
import warnings
import sys
import traceback
from pandas.core.frame import DataFrame
from strategy_v3.Strategy import GridArithmeticStrategy, StrategyModel
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.ExecuteSetup import ExecuteSetup
from datetime import datetime
from zoneinfo import ZoneInfo
from time import sleep
from requests.exceptions import Timeout
from binance.exceptions import BinanceAPIException
from strategy_v3.Misc import CustomException

warnings.filterwarnings('ignore')

# /usr/local/bin/python3 /Users/lok419/Desktop/JupyterLab/Trading/strategy_v3/execute.py SOLv1

if __name__ == '__main__':    

    strategy_id = sys.argv[1]    
    strategy_setup = ExecuteSetup(strategy_id)
    strategy_params = strategy_setup.read()    
    
    strategy = GridArithmeticStrategy(**strategy_params)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id)    
    strategy.run()        
