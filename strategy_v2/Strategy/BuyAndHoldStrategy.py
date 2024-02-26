import pandas as pd
from pandas.core.api import DataFrame as DataFrame
from .StrategyBase import StrategyBase

class BuyAndHoldStrategy(StrategyBase):      

    def __init__(self, confidence=1):
        if confidence > 2 or confidence < -2:
            raise('confidence has to between -2 and 2')
                
        self.confidence = confidence
        super().__init__()

    def __str__(self) -> str:
        return f'BAH{self.confidence}'

    def generate_position(self):
        close_px = self.data['px']['Close']
        position = pd.DataFrame(index=close_px.index, columns=close_px.columns).fillna(self.confidence)        
        position = position.loc[self.start_date: self.end_date]
        self.position = position

    def get_position(self) -> DataFrame:
        return self.position