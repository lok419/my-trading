from pandas.core.api import DataFrame as DataFrame
from pandas.tseries.offsets import BDay
import xgboost as xgb
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel

class XGB(AlphaModel):      
    '''
        XGB Approach to predict the stock average return's in N days
    '''
    def __init__(self, forecast=10, train_days=30):                                        
        self.forecast = forecast
        self.train_days = train_days
    
    def preprocess_data(self, data:dict):
        '''
            Preprocess OHCL data and create features
        '''    
        super().preprocess_data(data)                    
        temp = self.data['px'].copy()
        temp = temp.swaplevel(0,1,axis=1)
        dfml = {}
        for i in self.instruments:
            dfml[i] = self.create_features(temp[i])            
        self.dfml = dfml                                        
    
    def create_features(self, df):        
        '''
            Create Features for Machine Learning
        '''        
        dfml = df.copy()
        dfml['return'] = 100 * df['Close'].pct_change().fillna(0)
        dfml['return'] = 100 * np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

        dfml[f'ret_{self.forecast}d'] = dfml['return'].rolling(window=self.forecast, min_periods=self.forecast).mean()
        dfml[f'vol_{self.forecast}d'] = dfml['return'].rolling(window=self.forecast, min_periods=self.forecast).std()

        dfml['dayofweek'] = dfml.index.to_series().dt.day_of_week
        dfml['quarter'] = dfml.index.to_series().dt.quarter
        dfml['month'] = dfml.index.to_series().dt.month
        dfml['year'] = dfml.index.to_series().dt.year
        dfml['dayofyear'] = dfml.index.to_series().dt.day_of_year
        dfml['dayofmonth'] = dfml.index.to_series().dt.day

        dfml['vol_prev1'] = dfml[f'vol_{self.forecast}d'].shift(1)
        dfml['vol_prev5'] = dfml[f'vol_{self.forecast}d'].shift(5)
        dfml['vol_prev10'] = dfml[f'vol_{self.forecast}d'].shift(10)

        dfml['ret_prev1'] = dfml[f'ret_{self.forecast}d'].shift(1)
        dfml['ret_prev5'] = dfml[f'ret_{self.forecast}d'].shift(5)
        dfml['ret_prev10'] = dfml[f'ret_{self.forecast}d'].shift(10)

        dfml['o1'] = dfml['Open'].shift(1)
        dfml['o5'] = dfml['Open'].shift(5)
        dfml['o10'] = dfml['Open'].shift(10)

        dfml['c1'] = dfml['Close'].shift(1)
        dfml['c5'] = dfml['Close'].shift(5)
        dfml['c10'] = dfml['Close'].shift(10)

        dfml['v1'] = dfml['Volume'].shift(1)
        dfml['v5'] = dfml['Volume'].shift(5)
        dfml['v10'] = dfml['Volume'].shift(10)

        # label to predict
        dfml['y'] = dfml[f'ret_{self.forecast}d'].shift(-self.forecast)

        dfml = dfml.fillna(method='ffill')
        dfml = dfml.fillna(0)

        return dfml
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
            Using XGB Approach
        '''        
        lookback_end = pos_date - BDay(1)          
        expected_ret = []

        for i in self.instruments:
            df = self.dfml[i]
            df = df[:lookback_end]   
            df = df.tail(self.train_days + self.forecast)  
            assert max(df.index) < pos_date, 'Optimization has lookahead bias'          
            
            X = df.drop(columns=['y'])
            y = df['y']

            # train data needs to be shifted by forcast day because the label does not exist
            tx = X.iloc[:-self.forecast]
            ty = y.iloc[:-self.forecast]

            model = xgb.XGBRegressor(n_estimators=1000,early_stopping_rounds=50)            
            model.fit(tx, ty, eval_set=[(tx,ty)], verbose=False)      

            # use T-1 data to predict            
            pred_y = model.predict([X.iloc[-1]])[0]
            expected_ret.append(pred_y)
            
        return np.array(expected_ret)

    