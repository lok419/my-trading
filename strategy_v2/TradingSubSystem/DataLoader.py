import yfinance
import pandas as pd
from utils.data_helper import add_bday
from utils.data import get_yahoo_data_formatted
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
        self.data['px'] = get_yahoo_data_formatted(self.instruments, data_start, data_end)