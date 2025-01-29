from datetime import datetime
from utils.logging import get_logger
from utils.data import get_sp500_tickers, get_yahoo_data_formatted
from utils.db import duck
import strategy_v4.Data.features as features_
import pandas as pd

class DataLayer(object):

    def __init__(self,                  
                 start_date: datetime, 
                 end_date: datetime,
                 instruments: list[str] = get_sp500_tickers(),
                 name:str = 'sp500'            
        ):        
        '''
            Args:
                start_date: start date
                end_date: end date
                instruments: list of tickers for yahoo API
        '''

        self.start_date = start_date
        self.end_date = end_date
        self.instruments = instruments
        self.logger = get_logger('Data Layer')
        
        self.logger.info(f"start_date: {start_date:%Y-%m-%d}")
        self.logger.info(f"end_date: {end_date:%Y-%m-%d}")                        
        self.db_object = f'model_{name}_{start_date:%Y%m%d}_{end_date:%Y%m%d}'

    def load(self):
        '''
            Load the price data from Yahoo
        '''
        self.px = get_yahoo_data_formatted(self.instruments, self.start_date, self.end_date)

    def process(self):
        df = self.px.copy()
        df.columns.names = ['Feature', 'Stock']        

        '''
            Iterate all features function under "feature.py"
        '''
        funcs_name = [x for x in dir(features_) if x.startswith('gen_feature_')]
        for name in funcs_name:
            func = getattr(features_, name)
            df = func(df)

        '''
            Transform the data into column wise
        '''        
        df = df.stack(level='Stock').reset_index()

        # some unexpected adjusted close are added from yahoo api
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])
            
        self.df = df

    def upload(self):
        '''
            upload data to database / file
        '''
        self.df.to_parquet(f'data/parquet/{self.db_object}.parquet')

    def get(self) -> pd.DataFrame:
        return pd.read_parquet(f'data/parquet/{self.db_object}.parquet')
    