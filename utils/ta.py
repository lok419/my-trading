from pandas.core.frame import DataFrame
from pandas.core.series import Series
from ta.momentum import RSIIndicator
import pandas as pd

def rsi(df: DataFrame|Series, window: int = 14, columns=[]) -> DataFrame:
    '''
        RSI function from ta library but handling of input is a dataframe
    '''
    
    df = df.copy()
    if type(df) is Series:
        return RSIIndicator(df, window=window).rsi()        
        
    columns = df.columns if columns == [] else columns
    for column in columns:
        df[column] = RSIIndicator(df[column], window=window).rsi()        
    return df

def merge_ta(df: DataFrame, df_ta: DataFrame, ta_name=''):
    '''
        Merge the ta dataframe to the main stock price dataframe, following formats from yfinance

        i.e.
        df.columns = [('Close', 'StockA'), ('Close', 'StockB'), ('Close', 'StockC')....]
        df_ta.columns = ['StockA', 'StockB'....]
        
        Here adding one more index layer to df => ('New TA', 'StockA'), ('New TA', 'StockB'), ('New TA', 'StockC')
    '''
    df_ta.columns = pd.MultiIndex.from_product([[ta_name], df_ta.columns])
    df = df.merge(df_ta, left_index=True, right_index=True)
    return df
