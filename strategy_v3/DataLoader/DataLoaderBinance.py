from strategy_v3.DataLoader import DataLoaderModel
from pandas.core.frame import DataFrame
from account.Binance import Binance
from datetime import datetime

class DataLoaderBinance(DataLoaderModel):

    def __init__(self):
        self.binance = Binance()
                
    def load_price_data(self,
                        instrument:str,
                        interval:str,
                        lookback:str|datetime,
                        lookback_end:str|datetime = None,
                        ) -> DataFrame:        
        
        df = self.binance.get_historical_instrument_price(instrument, interval=interval, start_str=lookback, end_str=lookback_end)
        return df
        
        
