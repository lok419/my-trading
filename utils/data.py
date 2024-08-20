from pandas.core.frame import DataFrame
import pandas as pd
import yfinance as yf
import requests
from functools import cache
from datetime import datetime
from utils.data_helper import add_bday

@cache
def get_sp500_tickers():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url)[0]
    tickers = list(table['Symbol'].values)
    tickers = list(map(lambda x: x.replace(".", "-"), tickers))
    return tickers

@cache
def get_risk_free_rate(is_daily = True):
    rf = yf.download(tickers='^IRX',interval="1d",auto_adjust=True).reset_index()
    rf = rf.set_index('Date')
    rf = rf['Close']
    rf /= 100
    if is_daily:
        rf = (1+rf) ** (1/365) -1
    return rf

@cache
def get_latest_risk_free_rate(is_daily = True):
    rf = get_risk_free_rate(is_daily = is_daily)
    return rf[-1]

def get_sp500_constituent_hist():
    sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(sp500_url)[1]
    table = table.reset_index()
    table['Date'] = table['Date']['Date'].apply(lambda x: datetime.strptime(x, '%B %d, %Y'))

    table_added = table[['Date', 'Added', 'Reason']]
    table_added = table_added.droplevel(0, axis=1)
    table_added = table_added[['Date', 'Ticker', 'Reason']]
    table_added.columns = ['date', 'symbol', 'reason']
    table_added['type'] = 'add'

    table_removed = table[['Date', 'Removed', 'Reason']]
    table_removed = table_removed.droplevel(0, axis=1)
    table_removed = table_removed[['Date', 'Ticker', 'Reason']]
    table_removed.columns = ['date', 'symbol', 'reason']
    table_removed['type'] = 'remove'

    table = pd.concat([table_added, table_removed])
    table = table[~table['symbol'].isnull()]    

    return table

@cache
def get_us_listed_stocks_table():
    '''
        Get all stock data from NAQDAS stock screener
        download = true enables the API to return the full data with industry / sector fields
    '''

    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:84.0) Gecko/20100101 Firefox/84.0"}
    url = "https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=999999&download=true"
    res = requests.get(url, headers=headers)
    res = res.json()
    rows = res['data']['rows']

    df = pd.DataFrame(rows)    
    df['lastsale'] = df['lastsale'].replace('', '0').apply(lambda x: float(x.replace('$', '') or '0'))
    df['netchange'] = df['netchange'].replace('', '0').astype('float')
    df['pctchange'] = df['pctchange'].replace('', '0').apply(lambda x: float(x.replace('%', '') or '0'))
    df['volume'] = df['volume'].replace('', '0').astype('float')
    df['marketCap'] = df['marketCap'].replace('', '0').astype('float')

    return df

@cache
def get_us_listed_tickers():
    df = get_us_listed_stocks_table()
    symbols = list(df['symbol'].unique())
    return symbols

@cache
def get_yahoo_data(*args, **kwargs):
    '''
        This is a cached wrapper to Yahoo API so that we can cache the data globally
    '''    
    return yf.download(*args, **kwargs)

def get_yahoo_data_formatted(instruments: list[str], start_date: datetime, end_date: datetime) -> DataFrame:

    # Bug from yahoo API, it doesn't include price at end_date            
    px = get_yahoo_data(tickers=tuple(instruments), interval="1d",auto_adjust=True, start=start_date, end=add_bday(end_date, 10))
    px = px.copy()

    # If there is only one instruments, restructure the data to include name in columns
    if len(instruments) == 1:
        px.columns = pd.MultiIndex.from_product([px.columns, instruments])

    px = px[start_date: end_date]
    return px