from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy import MeanVarianceOpt
from utils.data_helper import add_bday


class MeanVarianceOptRetLookAheadBias(MeanVarianceOpt):      
    '''
        MVO with lookahead bias in expected return. This serves as the maximum pnl that that model can acheve if we can predict the expected return perfectly
    '''
    def __init__(self,     
                 lookahead = 10,          
                 *args,
                 **kwargs,    
                ):         
        self.lookahead = lookahead                               
        super().__init__(*args,**kwargs)

    def __str__(self) -> str:
        return f'MVO - RetLookAheadBias{self.confidence}'    
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return as actual future return, this is used to test the MVO            
        '''        
        ret = self.data['px']['Return']                
        ret = ret[pos_date:add_bday(pos_date,self.lookahead)]
        ret = ret.mean(axis=0).values       

        return ret

    