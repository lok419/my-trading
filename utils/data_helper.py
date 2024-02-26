from pandas.tseries.offsets import BDay
from datetime import datetime
from functools import cache
import pandas_market_calendars as mcal
import pandas as pd
import pytz

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

def title_case(x):
    return x.title().replace('_', ' ')

if __name__ == "__main__":    
    hols = trading_holidays()
    d = datetime(2023,10,27) 

    next = add_bday(d, 1)
    last_b = add_bday(add_bday(d, 1), -1)

    print(hols)
    print(next)
    print(last_b)
    print(get_today())