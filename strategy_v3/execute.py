import pandas as pd
import warnings
from strategy_v3.Strategy import GridArithmeticStrategy
from strategy_v3.Executor import ExecutorBinance
from strategy_v3.DataLoader import DataLoaderBinance
from datetime import datetime
from zoneinfo import ZoneInfo
from time import sleep
from requests.exceptions import Timeout

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    strategy = GridArithmeticStrategy(
        instrument = 'BTCFDUSD',
        interval = '5m',
        grid_size = 5,
        vol_lookback = 30,
        vol_grid_scale = 0.2,
        vol_stoploss_scale = 7,
        position_size = 500,
        hurst_exp_mr_threshold = 0.5,
        hurst_exp_mo_threshold = 0.6,
    )

    strategy.set_price_decimal(2)
    strategy.set_qty_decimal(5)
    strategy.set_data_loder(DataLoaderBinance())
    strategy.set_executor(ExecutorBinance())
    strategy.set_strategy_id('v1')
    
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
                sleep(30)
            except Timeout as e:
                strategy.logger.error(e)        
                strategy.logger.error('handled explicitly. retring....')        

    except KeyboardInterrupt:
        # if we interrupt manully, do nothing
        pass                
    except Exception as e:
        strategy.logger.error(e)        
        strategy.cancel_all_orders()
        strategy.close_out_positions()    