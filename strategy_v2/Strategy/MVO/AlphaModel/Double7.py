from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel


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
        return f"Double({self.periods})"
    
    def preprocess_data(self, data):
        super().preprocess_data(data)

        df = self.data['px']           

        # signals on T is based on T-1 Close and before
        df_close = df['Close'].shift(1)
        
        df_200ma = df_close.rolling(200).mean()        
        df_7d_low = df_close.rolling(self.periods).min()
        df_7d_high = df_close.rolling(self.periods).max()
        df_ret = df_close.pct_change().rolling(self.periods).mean().abs().clip(0, 1)

        df_entry = df_ret * ((df_close == df_7d_low) & (df_close > df_200ma))
        df_exit = -1 * ((df_close == df_7d_high) | (df_close < df_200ma))

        df_sig = (df_entry + df_exit).fillna(0)
        df_pos = df_sig.copy()

        # re-create the position based on signals
        for i in range(1, len(df_pos)):
            df_pos.iloc[i] = df_pos.iloc[i] + df_pos.iloc[i-1]
            df_pos.iloc[i] = np.clip(df_pos.iloc[i], 0, 1)

        self.df_pos = df_pos
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:
        expected_ret = np.array(self.df_pos.loc[:pos_date].iloc[-1])

        return expected_ret
    