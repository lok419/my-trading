from pandas.core.api import DataFrame as DataFrame
from pandas.tseries.offsets import BDay
from lightgbm import LGBMRegressor
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel

class LGBM(AlphaModel):      
    '''
        LGBM Approach to predict the stock average return's in N days
    '''
    def __init__(self, forecast=10, train_days=30):                                        
        self.forecast = forecast
        self.train_days = train_days   

    def __str__(self):
        return f'{super().__str__()}({self.forecast},{self.train_days})'     
    
    def preprocess_data(self, data:dict[DataFrame]):
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

        dfml['volume%_3d'] = 100 * df['Volume'] / df['Volume'].rolling(window=3, min_periods=3).sum()
        dfml['volume%_5d'] = 100 * df['Volume'] / df['Volume'].rolling(window=5, min_periods=5).sum()
        dfml['volume%_10d'] = 100 * df['Volume'] / df['Volume'].rolling(window=10, min_periods=10).sum()

        dfml['ret_3d'] = dfml['return'].rolling(window=3, min_periods=3).mean()
        dfml['ret_5d'] = dfml['return'].rolling(window=5, min_periods=5).mean()
        dfml['ret_10d'] = dfml['return'].rolling(window=10, min_periods=10).mean()

        dfml['vol_20d'] = dfml['return'].rolling(window=20, min_periods=20).std()
        dfml['vol_40d'] = dfml['return'].rolling(window=40, min_periods=40).std()

        # dfml['vol_prev1'] = dfml['vol_20d'].shift(1)
        # dfml['vol_prev5'] = dfml['vol_20d'].shift(5)
        # dfml['vol_prev10'] = dfml['vol_20d'].shift(10)        

        # dfml['ret_prev1'] = dfml[f'ret_{self.forecast}d'].shift(1)
        # dfml['ret_prev5'] = dfml[f'ret_{self.forecast}d'].shift(5)
        # dfml['ret_prev10'] = dfml[f'ret_{self.forecast}d'].shift(10)


        # label to predict
        dfml['y'] = dfml[f'ret_{self.forecast}d'].shift(-self.forecast)

        dfml = dfml.fillna(method='ffill')
        dfml = dfml.fillna(0)

        return dfml
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:
        '''
            Expected return based on lookback periods
            return a array of returns
            Using LGBM Approach
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

            model = LGBMRegressor(random_state=123, verbosity=-1)        
            model.fit(tx, ty, eval_set=(tx,ty))

            # use T-1 data to predict            
            pred_y = model.predict([X.iloc[-1]])[0]
            expected_ret.append(pred_y)
            
        return np.array(expected_ret)

    