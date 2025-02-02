from utils.data import get_latest_risk_free_rate
from utils.data_helper import title_case
from functools import cache
from IPython.display import display
from datetime import datetime
from plotly.subplots import make_subplots
from itertools import cycle
from pandas.tseries.offsets import BDay

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

def cumulative_return(r):        
    cum_return = np.cumprod(r+1)        
    return cum_return
    
def cumlative_log_return(r):
    cum_return = 1 + np.cumsum(np.log(r+1))
    return cum_return

def annualized_return(r):    
    expected_return = np.mean(r) * 252
    return expected_return

def annualized_volatility(r):    
    std = np.std(r) * np.sqrt(252)
    return std

def annualized_sharpe_ratio(r):    
    rf = get_latest_risk_free_rate()
    try:
        sr = (np.mean(r) - rf) / np.std(r) * np.sqrt(252)
    except FloatingPointError as e:
        sr = 0
    return sr

def annualized_volatility_ts(r, windows=20):
    std = pd.Series(r).rolling(windows).std() * np.sqrt(252)
    return std

def drawdown(r):
    val = cumulative_return(r)
    val_dd = []
    peak = val[0]
    for v in val:
        peak = max(peak, v)        
        dd = (v - peak) / peak
        val_dd.append(dd)
    return val_dd

def maximum_drawdown(r):
    dd = drawdown(r)
    max_dd = np.min(dd)
    return max_dd

def performance_summary(r, strategy=""):
    res = {}    
    res['strategy'] = strategy
    res['cumulative_return'] = cumulative_return(r)[-1]
    res['annualized_return'] = annualized_return(r)
    res['annualized_volatility'] = annualized_volatility(r)
    res['annualized_sharpe_ratio'] = annualized_sharpe_ratio(r)
    res['maximum_drawdown'] = maximum_drawdown(r)        
    return res

def performance_summary_table(rets_dict, benchmark=[]):
    """
        Return summary tables given arrays of time-series daily returns for different strategy
        Args:
            rets:         Array of daily return in time-series
            strategies:   Array of strategy names corresponding to r 
            benchmark:    Array of benchmarks
    """      
    if len(benchmark) > 0:
        r = list(rets_dict.values())[0]    
        start_date = r.index.min()
        end_date = r.index.max()
        for b in benchmark:            
            rets_dict[b] = get_benchmark_return(b, start_date, end_date)

    table = pd.DataFrame()
    for strategy, r in rets_dict.items():
        perf = performance_summary(r)
        del perf['strategy']
        measures = [title_case(x) for x in list(perf.keys())]        
        temp = pd.DataFrame({'Measure': measures, strategy: list(perf.values())})                     
        if table.empty:
            table = temp
        else:
            table = table.merge(temp, on='Measure', how='outer', validate='1:1')

    table['Measure'] = pd.Categorical(table['Measure'], categories=measures, ordered=True)
    table = table.set_index('Measure').sort_index()
       
    return table

@cache
def get_benchmark_return(symbol:tuple|str, start_date:datetime, end_date:datetime):
    """
        Get Return for a benchmarks from Yahoo                
    """  

    px = yf.download(tickers=symbol,interval="1d", auto_adjust=True, start=start_date, end=end_date + BDay(1))
    px_close = px['Close']
    ret = px_close / px_close.shift(1) - 1
    
    if type(symbol) is tuple and len(symbol) > 1:
        ret = ret.sum(axis=1)/len(symbol)
        
    ret = ret.fillna(0)
    return ret 

def performance_summary_plot(r: pd.Series|dict|pd.DataFrame, strategy: str="", benchmark: list=[]):
    '''
        Basic Summary Plots given return series
        r:          Either daily returns in pandas series or dictionary of returns in in pandas series keyed by strategy name
        strategy:   Strategy name of given return. Ignore if r is a dictionary
        benchmark:  List of benchmark to add for comparison
    '''

    if type(r) is pd.Series:
        r = r.fillna(0)        
        rets_dict = {}
        rets_dict[strategy] = r 
        start_date = r.index.min()
        end_date = r.index.max()
    else:
        rets_dict = r.copy()
        start_date = None
        end_date = None
        for s in rets_dict:            
            r[s] = r[s].fillna(0)
            start_date = min(r[s].index.min(), start_date) if start_date is not None else r[s].index.min()
            end_date = max(r[s].index.max(), start_date) if start_date is not None else r[s].index.max()      
    
    if len(benchmark) > 0:        
        for b in benchmark:            
            rets_dict[b] = get_benchmark_return(b, start_date, end_date)

    display(performance_summary_table(rets_dict))
    
    fig = make_subplots(
        rows=3, cols=1,        
        row_heights=[0.6, 0.3, 0.3],
        subplot_titles=[
            'Strategy Cumulative Log Return',
            'Daily Return (%)',
            'Volatility - 1m and 3m (%)',
        ],    
        vertical_spacing=0.05,
        shared_xaxes=True,        
    )    
    fig.update_layout(
        width=1500, height=1200,
        xaxis_showticklabels=True, 
        xaxis2_showticklabels=True, 
        xaxis3_showticklabels=True,
        hovermode='x',
    )

    fig1 = make_subplots(
        rows=1, cols=1,                
        subplot_titles=[            
            'Return Distribution (%)',
        ],    
        vertical_spacing=0.05,        
    ) 
    fig1.update_layout(width=800, height=500, hovermode='x')

    colors = cycle(plotly.colors.DEFAULT_PLOTLY_COLORS)    

    for s, r in rets_dict.items():      
        c = colors.__next__()
        fig.add_trace(go.Scatter(x=r.index, y=cumlative_log_return(r), name=s, legendgroup=s, marker=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=r.index, y=100*r, name=s, legendgroup=s, showlegend=False, marker=dict(color=c)), row=2, col=1)
        fig['layout']['yaxis2']['title']= 'Return (%)'

        v20 = r.rolling(20).std()*100*np.sqrt(252)
        v60 = r.rolling(60).std()*100*np.sqrt(252)

        fig.add_trace(go.Scatter(x=v20.index, y=v20, name=f'1m - {s}', legendgroup=s, showlegend=False, marker=dict(color=c), line=dict(width=3)), row=3, col=1)
        fig.add_trace(go.Scatter(x=v60.index, y=v60, name=f'3m - {s}', legendgroup=s, showlegend=False, marker=dict(color=c), line=dict(dash='dash')), row=3, col=1)
        fig['layout']['yaxis3']['title']= 'Vol (%)'

        # only compare days when strategy has non-zero return 
        hist_r = r[r!=0]

        fig1.add_trace(go.Histogram(x=100*hist_r, name=s, legendgroup=s, showlegend=True, marker=dict(color=c), xbins=dict(size=0.2)), row=1, col=1)
        fig1['layout']['xaxis']['title']= 'Return (%)'
        fig1['layout']['yaxis']['title']= 'Count'
        fig1.update_layout(barmode='overlay')
        fig1.update_traces(opacity=0.5)              

    fig.show()
    fig1.show()

def plot_price_ohcl(df, instrument=''):
    """
        OHCL Candlestick plots for one instrument
    """

    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.1, 
        subplot_titles=[
            'Price OHLC', 
            'Volume',
            'Volatility',
        ],     
        row_heights=[0.8, 0.2, 0.3],  

    )
    fig.update_layout(
        title=instrument,
        width=1500, height=1200,
        xaxis_showticklabels=True, 
        xaxis2_showticklabels=True,     
        hovermode='x',
    )
    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    # Price OHCL
    td = df['Date'].diff().iloc[-1]

    ma20 = df['Close'].rolling(20).mean()
    ma40 = df['Close'].rolling(40).mean()
    ma60 = df['Close'].rolling(60).mean()    

    ma20_name = f'SMA {_td_format(td*20)}'
    ma40_name = f'SMA {_td_format(td*40)}'
    ma60_name = f'SMA {_td_format(td*60)}'

    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", legendgroup='OHCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=ma20, name=ma20_name, legendgroup=ma20_name, marker=dict(color=colors[0]), visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=ma40, name=ma40_name, legendgroup=ma40_name, marker=dict(color=colors[1]), visible='legendonly'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=ma60, name=ma60_name, legendgroup=ma60_name, marker=dict(color=colors[2]), visible='legendonly'), row=1, col=1)

    # Volume
    df_up = df[df['Close'] > df['Open']]
    df_down = df[df['Close'] <= df['Open']]
    fig.add_trace(go.Bar(x=df_up['Date'], y=df_up['Volume'], showlegend=False, marker=dict(color=colors[2])), row=2, col=1)
    fig.add_trace(go.Bar(x=df_down['Date'], y=df_down['Volume'], showlegend=False, marker=dict(color=colors[3])), row=2, col=1)    

    # Volatility
    vol20 = 100 * df['Close'].pct_change().rolling(20).std()
    vol40 = 100 * df['Close'].pct_change().rolling(40).std()
    vol60 = 100 * df['Close'].pct_change().rolling(60).std()

    vol20_name = f'Vol {_td_format(td*20)}'
    vol40_name = f'Vol {_td_format(td*40)}'
    vol60_name = f'Vol {_td_format(td*60)}'

    fig.add_trace(go.Scatter(x=df['Date'], y=vol20, name=vol20_name, showlegend=True, marker=dict(color=colors[0]), line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=vol40, name=vol40_name, showlegend=True, marker=dict(color=colors[1]), line=dict(width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=vol60, name=vol60_name, showlegend=True, marker=dict(color=colors[2]), line=dict(width=2)), row=3, col=1)
    fig['layout']['yaxis3']['title']= '%'

    fig.update(layout_xaxis_rangeslider_visible=False)
    fig.show()

def _td_format(td_object):
    '''
        Format the timedelta object to string
    '''
    seconds = int(td_object.total_seconds())
    periods = [
        ('Y', 60*60*24*365),
        ('M', 60*60*24*30),
        ('d', 60*60*24),
        ('h', 60*60),
        ('m', 60),
        ('s', 1)
    ]
    
    for period_name, period_seconds in periods:
        if seconds > period_seconds:                        
            break

    is_int = seconds/period_seconds == int(seconds/period_seconds)
    num_str = str(seconds//period_seconds) if is_int else str(round(seconds/period_seconds, 1))
    return f'{num_str}{period_name}'


if __name__ == '__main__':
    r = get_benchmark_return('^SPX', datetime(2022,1,1), datetime(2023,11,2))         
    table = performance_summary_table({'strategy1': r, 'strategy2': r})
    display(table)
    






    













        



                 

