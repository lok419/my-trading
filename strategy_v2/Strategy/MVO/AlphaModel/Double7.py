from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel
from utils.data_helper import add_bday


class Double7(AlphaModel):      
    '''
        Double 7's Strategy        
        1. stocks above 200days MA
        2. entry when closes at 7days low 
        3. exit when closes at 7days high     
    '''
    def __init__(self, periods:int=7):
        self.periods = periods
        
    def __str__(self) -> str:
        return f"Double{self.periods}"
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:

        lookback_end = add_bday(pos_date, -1)
        expected_ret = []

        df = self.data['px']
        df = df[:lookback_end]
        df = df.tail(200)        

        assert max(df.index) < pos_date, 'Optimization has lookahead bias'             

        df_close = df['Close']                
        df_200ma = df_close.rolling(200).mean()        
        df_7d_low = df_close.rolling(self.periods).min()
        df_7d_high = df_close.rolling(self.periods).max()

        df_entry = 1 * ((df_close == df_7d_low) & (df_close > df_200ma))
        df_exit = -1 * ((df_close == df_7d_high) | (df_close < df_200ma))

        df_sig = df_entry + df_exit
        df_pos = df_sig.copy()

        # re-create the position based on signals
        for i in range(1, len(df_pos)):
            df_pos.iloc[i] = df_pos.iloc[i] + df_pos.iloc[i-1]
            df_pos.iloc[i] = np.clip(df_pos.iloc[i], 0, 1)

        expected_ret = df_pos.iloc[-1].values
            
        return np.array(expected_ret)
    