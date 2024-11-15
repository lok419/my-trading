from pandas.core.api import DataFrame as DataFrame
import numpy as np
from datetime import datetime
from strategy_v2.Strategy.MVO import AlphaModel
from utils.data_helper import add_bday
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
        return f"RSI{self.rsi_threshold}|{self.rsi_windows}"
    
    def expected_return(self, pos_date: datetime) -> np.ndarray:

        lookback_end = add_bday(pos_date, -1)
        expected_ret = []

        df = self.data['px']
        df = df[:lookback_end]
        df = df.tail(200)

        assert max(df.index) < pos_date, 'Optimization has lookahead bias'

        df_close = df['Close']        
        df_rsi = rsi(df_close, window=self.rsi_windows)
        df_200ma = df_close.rolling(200).mean()
        df_5ma = df_close.rolling(5).mean()

        # for RSI threshold < 50, we treat this as buy strategy
        # for RSI threshold > 50, we treat this as sell strategy (still return a positive number)
        if self.rsi_threshold < 50:
            df_entry = 1 * ((df_rsi < self.rsi_threshold) & (df_close > df_200ma))
            df_exit = -1 * (df_close > df_5ma)
        else:
            df_entry = 1 * ((df_rsi > self.rsi_threshold) & (df_close > df_200ma))
            df_exit = -1 * (df_close < df_5ma)

        df_sig = df_entry + df_exit
        df_pos = df_sig.copy()

        # re-create the position based on signals
        for i in range(1, len(df_pos)):
            df_pos.iloc[i] = df_pos.iloc[i] + df_pos.iloc[i-1]            
            df_pos.iloc[i] = np.clip(df_pos.iloc[i], 0, 1)            

        expected_ret = df_pos.iloc[-1].values
            
        return np.array(expected_ret)
    