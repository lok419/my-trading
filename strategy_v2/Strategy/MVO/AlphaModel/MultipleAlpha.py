from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO.AlphaModelBase import AlphaModel
from utils.logging import get_logger

class MultipleAlpha(AlphaModel):      
    '''
        Wrapper which wraps multiple alpha models as the signals to optimize        
    '''
    def __init__(self, models: list[AlphaModel], weights: list[float]):
        self.models = models        
        self.weights = weights        
        self.logger = get_logger(self.__class__.__name__)

        assert len(self.models) == len(self.weights)        

        if sum(self.weights) != 1:
            norm = 1/sum(self.weights)
            self.weights = [w * norm for w in self.weights]
            self.logger.info('Normalizing alpha model weights by {:.2f}'.format(norm))

    def __str__(self) -> str:
        return '&'.join([str(m) for m in self.models])
    
    def preprocess_data(self, data:dict[DataFrame]):
        for model in self.models:
            model.preprocess_data(data)                        
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:        
        for i, (model, weight) in enumerate(list(zip(self.models, self.weights))):
            if i == 0:
                rets = model.expected_return(pos_date) * weight
            else:        
                rets += model.expected_return(pos_date) * weight        
        return rets



        
    