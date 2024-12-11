from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel
from utils.ta import rsi


class RSI(AlphaModel):      
    '''
        Short Period RSI Mean Reverting signals

        2-period RSI indicators sorts of capturing the short terms mean reverting behavior of the stocks

            1. spot price > 200MA
            2. entry when 2-period RSI < 5
            3. exit when price > 5MA        
    '''
    def __init__(self, rsi_windows=2, rsi_threshold=10):           
        self.rsi_windows = rsi_windows
        self.rsi_threshold = rsi_threshold

    def __str__(self) -> str:
        return f"RSI({self.rsi_threshold},{self.rsi_windows})"

    def preprocess_data(self, data):
        super().preprocess_data(data)        

        df = self.data['px']

        # signals on T is based on T-1 Close and before
        df_close = df['Close'].shift(1)

        df_rsi = rsi(df_close, window=self.rsi_windows)
        df_200ma = df_close.rolling(200).mean()
        df_5ma = df_close.rolling(5).mean()
        df_ret = df_close.pct_change().rolling(self.rsi_windows).mean().abs().clip(0, 1)

        # for RSI threshold < 50, we treat this as buy strategy
        # for RSI threshold > 50, we treat this as sell strategy (still return a positive number)
        # use rolling average abs returns to determine size of the signals
        if self.rsi_threshold < 50:
            df_entry = df_ret * ((df_rsi < self.rsi_threshold) & (df_close > df_200ma))
            df_exit = -1 * (df_close > df_5ma)
        else:
            df_entry = df_ret * ((df_rsi > self.rsi_threshold) & (df_close > df_200ma))
            df_exit = -1 * (df_close < df_5ma)
        
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
    