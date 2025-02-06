import pandas as pd
from sklearn.model_selection import train_test_split
from strategy_v4.config import MODEL_LAYER, DATA_LAYER
from strategy_v4.Data.Data import DataLayer
from strategy_v4.Model.Model import Model
from strategy_v4.Model.Preprocess import Preprocess
from tqdm import tqdm
from copy import copy

def run_this(
        model: Model,
        preprocess: Preprocess = None,
        features: list[str] = ['*'],
        label_name :str = MODEL_LAYER.label_name,
        label_shift :int = MODEL_LAYER.label_shift,
        test_size: float = MODEL_LAYER.test_size,  
        min_train_size: int = MODEL_LAYER.min_train_size,
        assets: list[str] = [],      
) -> tuple[pd.DataFrame, dict[str, Model]]:    
    '''
        Run a model per assets
        Args:
            model:          Model Object
            features:       list of features to use
            label_name:     name of label to use    
            label_shift:    shift of label
            test_size:      test size
            assets:         list of assets
    '''
    dl_args = {key: value for key, value in vars(DATA_LAYER).items() if not key.startswith('_')}
    df = DataLayer(**dl_args).get()
    if not len(assets):
        assets = df['asset'].unique()
    
    predictions = []
    models = {}

    with tqdm(total=len(assets)) as pbar: 
        for asset in assets:
            # copy the model object give each asset has independent models
            model_ = copy(model)

            df_ = df[df['asset'] == asset].drop(columns=['asset']).set_index('date')
            df_['label'] = df_[label_name].shift(label_shift)            
            df_ = df_.dropna()

            if len(df_) <= min_train_size:
                print(f"skipped {asset}")
                pbar.update(1)
                continue

            fs = [x for x in df_.columns if x != 'label']
            if features != ['*']:
                fs = [x for x in fs if x in features]

            X, y = df_[fs], df_['label']            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            if preprocess is not None:
                preprocess.fit(X_train)                
                X_train_ = preprocess.transform(X_train)                
                X_test_ = preprocess.transform(X_test)
            else:
                X_train_ = X_train.values
                X_test_ = X_test.values

            model_.fit(X_train_, y_train)        
                        
            pred_in = pd.DataFrame({'actual': y_train, 'pred': model_.predict(X_train_), 'pred_type': 'in-sample', 'asset': asset, 'date': X_train.index})
            pred_out = pd.DataFrame({'actual': y_test, 'pred': model_.predict(X_test_), 'pred_type': 'out-sample', 'asset': asset, 'date': X_test.index})
            pred = pd.concat([pred_in, pred_out])            

            predictions.append(pred)
            models[asset] = model_

            pbar.update(1)        

    df_pred = pd.concat(predictions)   
    return df_pred, models

def run_once(model_name, assets=[]) -> tuple[pd.DataFrame, dict[str, Model]]:
    setup = MODEL_LAYER.models[model_name]
    model = setup['model']
    features = setup['features']        
    preprocess = setup.get('preprocess', None)

    df_pred, models = run_this(model, preprocess, features, assets=assets)    
    return df_pred, models

def run(model_name, assets=[]):
    '''
        run a specified models defined in config for airflow
    '''  
    df_pred, _ = run_once(model_name, assets)        
    file_name = f'data/parquet/model_pred/{model_name}.parquet'
    df_pred.to_parquet(file_name, index=False)     
    print(f"saved {file_name}")              

