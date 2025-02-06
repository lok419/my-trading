from datetime import datetime
from utils.data import get_sp500_tickers, get_djia_tickers
from strategy_v4.Model.LinearModels import *
from strategy_v4.Model.NonLinearModels import *
from sklearn.preprocessing import StandardScaler

class DATA_LAYER:
    dataset = 'djia'
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2024, 1, 29)
    instruments = get_djia_tickers()    

class MODEL_LAYER:
    label_name = 'return5d'
    label_shift = -5
    test_size = 0.2 
    min_train_size = 50    

    models = {
        # Linear Models
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
        },

        # Non-Linear
        'AdaBoostReg': {
            'model': AdaBoostreg(),
            'features': ['*'],
        },

        'BaggingReg': {
            'model': BaggingReg(),
            'features': ['*'],
        },

        'ExtraTreesReg': {
            'model': ExtraTreesReg(),
            'features': ['*'],  
        },

        'GradientBoostingReg': {
            'model': GradientBoostingReg(),            
            'features': ['*'],  
        },

        'HistGradientBoostingReg': {
            'model': HistGradientBoostingReg(),
            'features': ['*'],  
        },

        'RandomForestReg': {
            'model': RandomForestReg(),
            'features': ['*'],  
        },
    }

