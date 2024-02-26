from strategy_v3.DataLoader import DataLoaderModel
from pandas.core.frame import DataFrame
from account import Binance

class DataLoaderBinance(DataLoaderModel):

    def __init__(self):
        self.binance = Binance()
                
    def load_price_data(self,
                        instrument:str,
                        interval:str,
                        lookback_period:str
                        ) -> DataFrame:        
        
        df = self.binance.get_historical_instrument_price(instrument, interval=interval, start_str=lookback_period)
        return df
        
        
