import pandas as pd
import numpy as np
from datetime import datetime
from strategy_v4.config import MODEL_LAYER, DATA_LAYER
from utils.logging import get_logger
from pandas.core.frame import DataFrame
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, recall_score, precision_score, f1_score, accuracy_score
from utils.db import duck

def eval_metrics(x):
    '''
        Calculate evaluation metrics
    '''    
    e = {}
    e['r2'] = r2_score(x['actual'], x['pred'])
    e['mse'] = mean_squared_error(x['actual'], x['pred'])
    e['mae'] = mean_absolute_error(x['actual'], x['pred'])
    e['accuracy_score'] = accuracy_score(x['actual'] > 0, x['pred'] > 0)

    e['up_recall'] = recall_score(x['actual'] > 0, x['pred'] > 0)
    e['up_precision'] = precision_score(x['actual'] > 0, x['pred'] > 0)
    e['up_f1_score'] = f1_score(x['actual'] > 0, x['pred'] > 0)

    e['down_recall'] = recall_score(x['actual'] < 0, x['pred'] < 0)
    e['down_precision'] = precision_score(x['actual'] < 0, x['pred'] < 0)
    e['down_f1_score'] = f1_score(x['actual'] < 0, x['pred'] < 0)
    return pd.Series(e)

class Evaluate(object):

    def __init__(self, asset_level=False):
        self.logger = get_logger("Evaluate")
        self.db_object = 'return_predictions'    
        self.asset_level = asset_level  

    def load(self, 
             model_names: str|list[str] = list(MODEL_LAYER.models.keys()), 
             df_pred:DataFrame = None,                          
        ):
        '''
            Load prediction results.
            Args:   
                model_names:    list of model names
                df_pred:        input dataframe if you want to run this class individually                
        '''
        if df_pred is not None:
            self.df_pred = df_pred.copy()
            self.df_pred['model'] = 'custom'
            return
        
        if type(model_names) is str:
            model_names = [model_names]        

        df_pred = []
        for model_name in model_names:
            pred = pd.read_parquet(f'data/parquet/model_pred/{model_name}.parquet')
            pred['model'] = model_name            

            df_pred.append(pred)            
            self.logger.info(f'loaded {model_name}.')

        df_pred = pd.concat(df_pred)
        self.df_pred = df_pred

    def eval(self):        
        '''
            Evaluate the models
        '''      
        self.df_eval = self.df_pred.groupby(['pred_type', 'model']).apply(eval_metrics).reset_index()
        if self.asset_level:
            self.df_eval_asset = self.df_pred.groupby(['pred_type', 'model', 'asset']).apply(eval_metrics).reset_index()

    def format(self, df: DataFrame):
        '''
            Add all model information to the dataframe for reference
        '''
        # data layer
        dl_dict = {key: value for key, value in vars(DATA_LAYER).items() if not key.startswith('_') and key != 'instruments'}
        ml_dict = {key: value for key, value in vars(MODEL_LAYER).items() if not key.startswith('_') and key != 'models'}

        for k, v in dl_dict.items():
            df[k] = v

        for k, v in ml_dict.items():            
            df[k] = v

        for model_name, d in MODEL_LAYER.models.items():
            for k, v in d.items():
                if k != 'model':
                    v = ",".join(v) if type(v) is list else v
                    if k not in df.columns:
                        df[k] = ''
                    df[k] = np.where(df['model'] == model_name, v, df[k])  

        df['time'] = datetime.now()
        return df

    def upload(self):
        '''
            upload the evaluation results to database
        '''
        db = duck(database=self.db_object, read_only=False)        
        db.insert('model_eval', self.format(self.df_eval), append_new_column=True)
        if self.asset_level:
            db.insert('model_pred', self.format(self.df_pred), append_new_column=True)
            db.insert('model_eval_asset', self.format(self.df_eval_asset), append_new_column=True)






    

    