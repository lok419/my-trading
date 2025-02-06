from datetime import datetime
from utils.logging import get_logger
from utils.data import get_sp500_tickers, get_yahoo_data_formatted
from utils.ta import rsi
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

'''
    All function starts with gen_feature_ will be called in data layer
'''
def gen_feature_return(df: DataFrame) -> DataFrame: 
    windows = [1,3,5,10,20,60]
    ret = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)        
    for w in windows:
        x = ret.rolling(window=w, min_periods=w).mean()
        name = f'return{w}d'
        df = append_features(df, x, name)
    return df

def gen_feature_rsi(df: DataFrame) -> DataFrame: 
    windows = [2,14,28,60]
    for w in windows:
        x = rsi(df['Close'], w)
        name = f'rsi{w}d'
        df = append_features(df, x, name)
    return df

def gen_feature_volume(df: DataFrame) -> DataFrame:
    windows = [1,3,5,10,20,60]
    vol = df['Volume']
    for w in windows:
        x = vol.rolling(window=w, min_periods=w).mean()
        name = f'volume{w}d'
        df = append_features(df, x, name)
    return df

def gen_feature_std(df: DataFrame) -> DataFrame:
    windows = [5,10,20,60]
    ret = 100 * np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

    for w in windows:
        x = ret.rolling(window=w, min_periods=w).std()
        name = f'std{w}d'
        df = append_features(df, x, name)

    return df

def gen_feature_mr_ratio(df: DataFrame) -> DataFrame:
    price_rev = (df['High'] - df['Low']) / 2 - df['Close']
    df = append_features(df, price_rev, 'price_rev')
    return df

def append_features(df: DataFrame, fs: DataFrame, name: str):
    fs.columns = pd.MultiIndex.from_product([[name], fs.columns])
    return pd.concat([df, fs], axis=1)