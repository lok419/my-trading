from pandas.tseries.offsets import BDay
from datetime import datetime
from functools import cache
import pandas_market_calendars as mcal
import pandas as pd
import pytz
import numpy as np
import math

def bucketize(arr, bucket_size):
    output = []
    i = 0
    while i < len(arr):
        output.append(arr[i:i+bucket_size])
        i += bucket_size
    return output

@cache
def trading_holidays(market='NYSE'):
    cal = mcal.get_calendar(market)
    holdiays = cal.holidays().holidays
    return pd.to_datetime(holdiays)

def add_bday(date, offset, market='NYSE'):
    new_date = date + BDay(offset)
    holidays = trading_holidays(market)

    while new_date in holidays:                    
        if offset > 0:
            new_date = new_date + BDay(1)
        elif offset <= 0:
            new_date = new_date + BDay(-1)
    return new_date

def get_today(offset=0, tz='US/Eastern', market='NYSE'):
    date = datetime.now(pytz.timezone(tz)).date()    
    if offset != 0:
        date = add_bday(date, offset, market=market)
    date = pd.to_datetime(date)
    return date 

def is_ny_trading_hours(time):
    tz = 'US/Eastern'
    market = 'NYSE'
    time = time.astimezone(pytz.timezone(tz))

    nyse_open = time.replace(hour=9, minute=30, second=0, microsecond=0)
    nyse_close = time.replace(hour=16, minute=0, second=0, microsecond=0)
        
    is_holiday = time.date() in trading_holidays(market)
    is_weekend = time.date().weekday() >= 5
    is_market_hours = nyse_open <= time <= nyse_close

    return not is_holiday and not is_weekend and is_market_hours

def title_case(x):
    return x.title().replace('_', ' ')

def count_digit(x: str) -> int:    
    try:
        return len(np.format_float_positional(float(x), trim='-').split('.')[1])            
    except:
        return 0   
    
def round_down(number, decimal_places):
    factor = 10 ** decimal_places
    return math.floor(number * factor) / factor

if __name__ == "__main__":    
    hols = trading_holidays()
    d = datetime(2023,10,27) 

    next = add_bday(d, 1)
    last_b = add_bday(add_bday(d, 1), -1)

    print(hols)
    print(next)
    print(last_b)
    print(get_today())