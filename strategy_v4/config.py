from datetime import datetime
from utils.data import get_sp500_tickers
from strategy_v4.Model.LinearModels import *

class DATA_LAYER:
    dataset = 'sp500'
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2024, 1, 29)
    instruments = get_sp500_tickers()    

class MODEL_LAYER:
    label_name = 'return5d'
    label_shift = -5
    test_size = 0.2 
    min_train_size = 50    

    models = {
        'Lasso': {
            'model': LassoReg(),
            'features': ['*'],        
        },

        'Ridge': {
            'model': RidgeReg(),
            'features': ['*'],          
        },

        'LinearRegression': {
            'model': LinearReg(),
            'features': ['*'],        
        },

        'ElasticNet': {
            'model': ElasticNetReg(),
            'features': ['*'],
        }
    }

