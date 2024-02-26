from utils.logging import get_logger
from datetime import datetime
from abc import abstractmethod
import pandas as pd

class StrategyBase(object):
    def __init__(self):                
        self.data = {}
        self.position = pd.DataFrame()        

    def set_data(self, data:dict):
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
            Actual position sizing will be determined at sub-system level
        '''
        pass        

    @abstractmethod
    def get_position(self) -> pd.DataFrame:
        '''
            return the position after generate_position()         
        '''
        pass        