import pandas as pd
import warnings
import sys
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.DataLoader import DataLoaderBinance
from strategy_v3.ExecuteSetup.StrategyFactory import StrategyFactory

warnings.filterwarnings('ignore')

# /usr/local/bin/python3 /Users/lok419/Desktop/JupyterLab/Trading/strategy_v3/execute.py SOLv1

if __name__ == '__main__':    

    strategy_id = sys.argv[1]    
    strategy = StrategyFactory().get(strategy_id)    
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id(strategy_id, reload=True)    
    strategy.run()        
