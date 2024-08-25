from pandas.core.api import DataFrame as DataFrame, Series as Series
from strategy_v2.Portfolio import PortfolioBase
import pandas as pd

class PortfolioStandard(PortfolioBase):
    '''
        Standard Portfolio        
        Equally Weighted Portfolios among different trading sub system        
    ''' 

    def __init__(self, 
                 *args,
                 **kwargs,
        ):
        super().__init__(*args, **kwargs)            

    def __str__(self):
        return "Standard Portfolio" + (' ({self.name})' if len(self.name) else '')
    
    def optimize(self):    
        '''
            Equal Weighted for now            
        '''
        # run backtest_subsystems before optimization
        if len(self.position) == 0 and len(self.ret) == 0:
            self.backtest_subsystems()

        self.port_w = pd.DataFrame(index=self.ret.index, columns=self.ret.columns).fillna(1/len(self.systems)) 