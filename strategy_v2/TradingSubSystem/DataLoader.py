import yfinance
import pandas as pd
from utils.data_helper import add_bday
from utils.data import get_yahoo_data
from functools import cache

class DataLoader(object):
    '''
        This class contains all the data related functions which is extended from TradingSubSystem class
    '''
    def __init__(self):
        pass

    @cache
    def load_price_data_yahoo(self):
        """
            Load Daily OHCL data from Yahoo Finance API

            There is a bug from Yahoo API which does not include price at End Date, so we need to extend the end date a bit
        """                
        data_start = add_bday(self.start_date, -self.offset)    
        data_end = self.end_date

        # Bug from yahoo API, it doesn't include price at end_date            
        px = get_yahoo_data(tickers=tuple(self.instruments), interval="1d",auto_adjust=True, start=data_start, end=add_bday(data_end, 10))

        # make sure we don't change the cache results
        px = px.copy()
        
        # If there is only one instruments, restructure the data to include name in columns
        if len(self.instruments) == 1:
            px.columns = pd.MultiIndex.from_product([px.columns, self.instruments])

        px = px[data_start: data_end]
        self.data['px'] = px