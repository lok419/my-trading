from utils.logging import get_logger
from datetime import datetime
from pandas.core.frame import DataFrame
from abc import abstractmethod
import pandas as pd

class StrategyBase(object):
    def __init__(self):                
        self.data = {}
        self.position = pd.DataFrame()    
        self.logger = get_logger(self.__class__.__name__)

    def set_instruments(self, instruments:list|str):
        self.instruments = instruments
        return self

    def set_data(self, data:dict[DataFrame]):
        self.data = data
        return self
    
    def get_data(self) -> dict:
        return self.data
        
    def set_start_date(self, start_date: datetime):
        self.start_date = start_date
        return self
        
    def set_end_date(self, end_date:datetime):
        self.end_date = end_date
        return self     

    @abstractmethod
    def load_data(self):
        '''
            In case the data wasn't passed or additional data is needed            
        '''

    @abstractmethod
    def preprocess_data(self):
        '''
            Preprocess any necessary data before generating the position
        '''
        pass

    @abstractmethod
    def generate_position(self):
        '''
            Generate the position data. The function should assign "self.position" to a position numpy matrix with size (asset, dates)            
            Position varies between -1 and 1 which indicate the confidence of the forecast.                         
                - Positive maens Long
                - Negative means Short  

            Confidence could still be any values, this is useful when you have muliple strategies, you want to assign different confidence to each strategy
            Actual position sizing will be determined at sub-system level
        '''
        pass        

    @abstractmethod
    def get_position(self) -> pd.DataFrame:
        '''
            return the position after generate_position()         
        '''
        pass        