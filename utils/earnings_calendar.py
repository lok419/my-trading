import datetime
import requests
import pandas as pd
import os
import numpy as np
import json
from utils.logging import get_logger
from utils.Threads import CustomThreadPool
from datetime import datetime
from bs4 import BeautifulSoup

file = os.path.dirname(os.path.realpath(__file__)) + "/Data/earnings_calendar.h5"
logger = get_logger('Earning Calendar')

def get_earnings_by_symbols_from_yahoo(symbols, refresh=False):
    """
        Get Many Earning Calendar from Yahoo webside for a ticker
    """

    ec_file = get_earnings_calendar_file('yahoo')

    if not len(ec_file):
        symbols_to_get = symbols
    else:
        symbols_stored = ec_file['Symbol'].unique()
        symbols_to_get = np.setdiff1d(symbols, symbols_stored)
        symbols_to_read = np.intersect1d(symbols, symbols_stored)
        ec_file = ec_file[ec_file['Symbol'].isin(symbols_to_read)]    

        if not len(symbols_to_get):
            return ec_file

    ec_yahoo = CustomThreadPool().submit_many(True, get_earnings_by_symbol_from_yahoo, [[s] for s in symbols_to_get])
    ec_yahoo = pd.concat(ec_yahoo)
    
    if len(symbols_to_get) and len(ec_yahoo):        
        upsert_earnings_calendar_file(ec_yahoo, 'yahoo', 'Symbol')
        ec = pd.concat([ec_file, ec_yahoo])
    else:
        ec = ec_file

    return ec

def get_earnings_by_symbol_from_yahoo(symbol):
    """
        Get Earning Calendar from Yahoo webside for a ticker
    """

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0'}         
        is_fetch_all = False
        size = 100
        offset = 0
        cnt = 0
        ec = []

        while not is_fetch_all:
            url = f'https://finance.yahoo.com/calendar/earnings?symbol={symbol}&offset={offset}&size={size}'
            page = requests.get(url=url, headers=headers)
            try:        
                tables = pd.read_html(page.text)        
                table = tables[0]            
                ec.append(table)
                offset += 100
                cnt += len(table)
            except ValueError:
                is_fetch_all = True
            
        logger.info(f"Fetched {cnt} rows for {symbol} from Yahoo Earning Calendar.")

        if len(ec):
            ec = pd.concat(ec)        
            ec['EPS Estimate'] = pd.to_numeric(ec['EPS Estimate'], errors='coerce')
            ec['Reported EPS'] = pd.to_numeric(ec['Reported EPS'], errors='coerce')
            ec['Surprise(%)'] = pd.to_numeric(ec['Surprise(%)'], errors='coerce')             
            ec['Earnings Date Time'] = ec['Earnings Date'].apply(lambda x: datetime.strptime(x[:-3], '%b %d, %Y, %I %p'))
            ec['Date'] = pd.to_datetime(ec['Earnings Date Time'].dt.date)
        else:
            ec = pd.DataFrame()

    except Exception as e:
        logger.error(f'fetching errors for {symbol}:')
        logger.error(e)
        ec = pd.DataFrame()        

    return ec

def get_earnings_calendar_from_yahoo(date):
    """
        Get Earning Calendar from Yahoo webside, need to set headers as Browser        
    """

    date_str = date.strftime('%Y-%m-%d')        
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/114.0'}     

    try:        
        ec = []
        is_fetch_all = False
        size = 100
        offset = 0
        cnt = 0

        while not is_fetch_all:
            url = f'https://finance.yahoo.com/calendar/earnings?day={date_str}&offset={offset}&size={size}'
            page = requests.get(url=url, headers=headers)
            try:        
                tables = pd.read_html(page.text)        
                table = tables[0]
                table['Date'] = pd.to_datetime(date.date())
                ec.append(table)
                offset += 100
                cnt += len(table)

            except ValueError:
                is_fetch_all = True
            
        logger.info(f"Fetched {cnt} names from Yahoo Earning Calendar: {date_str}.")    
                
        if len(ec):
            ec = pd.concat(ec)        
            ec['EPS Estimate'] = pd.to_numeric(ec['EPS Estimate'], errors='coerce')
            ec['Reported EPS'] = pd.to_numeric(ec['Reported EPS'], errors='coerce')
            ec['Surprise(%)'] = pd.to_numeric(ec['Surprise(%)'], errors='coerce')         
        else:
            # Append an empty pandas such that we know this date is fetched
            ec = pd.DataFrame({'Date': [pd.to_datetime(date.date())]})

    except Exception as e:
        logger.error(f'fetching errors on {date_str}:')
        logger.error(e)
        ec = pd.DataFrame()

    return ec

def get_earnings_calendar_from_nasdaq(date):
    date_str = date.strftime('%Y-%m-%d') 
    headers = {
        "Accept":"application/json, text/plain, */*",
        "Accept-Encoding":"gzip, deflate, br",
        "Accept-Language":"en-US,en;q=0.9",
        "Origin":"https://www.nasdaq.com",
        "Referer":"https://www.nasdaq.com",
        "User-Agent":"your user agent..."
    }
 
    url = 'https://api.nasdaq.com/api/calendar/earnings?' 
    payload = {"date": date_str}     
    source = requests.get( url=url, headers=headers, params=payload, verify=True ) 
    data = source.json()
    ec = pd.DataFrame(data['data']['rows'])
    cnt = len(ec)
    if not len(ec):          
        ec = pd.DataFrame()

    ec['Date'] = pd.to_datetime(date.date())
    logger.info(f"Fetched {cnt} names from NASDAQ Earning Calendar: {date_str}.")    
    return ec

def upsert_earnings_calendar_file(new_ec, source, key):
    """
        Given the earning calendar dataframe, upsert to original HDF5 files

        Args:
            source: keys of the hdf5 file
            key:    key to upsert
    """
    ec_file = get_earnings_calendar_file(source)
    if len(ec_file):
        symbols = new_ec[key].unique()        
        ec_file = ec_file[~ec_file[key].isin(symbols)]
        ec_file = pd.concat([ec_file, new_ec])
    else:
        ec_file = new_ec

    file = os.path.dirname(os.path.realpath(__file__)) + "/Data/earnings_calendar.h5"
    ec_file.to_hdf(file, key=source, format='table')

def get_earnings_calendar_file(source='yahoo'):
    """
        Get the earnings calendar from HDF5 
    """
    try:        

        file = os.path.dirname(os.path.realpath(__file__)) + "/Data/earnings_calendar.h5"
        ec = pd.read_hdf(file, key=source, mode='r')
    except Exception as e:
        ec = pd.DataFrame()
    return ec

def get_earnings_by_from_investcom(start_date, end_date=None, refresh=False):

    if not end_date:
        end_date = start_date        
        if not refresh:
            ec_file = get_earnings_calendar_file(source='investcom')
            if len(ec_file):
                ec_file = ec_file[ec_file['Date'] == start_date]
                if len(ec_file):
                    return ec_file

    is_fetch_all = False
    last_time_scope = 0
    limit_from = 0
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    ec = []

    while not is_fetch_all:

        curl_req = '''curl 'https://www.investing.com/earnings-calendar/Service/getCalendarFilteredData' \
            -H 'authority: www.investing.com' \
            -H 'accept: */*' \
            -H 'accept-language: en,en-US;q=0.9,ru-RU;q=0.8,ru;q=0.7' \
            -H 'content-type: application/x-www-form-urlencoded' \
            -H 'cookie: adBlockerNewUserDomains=1665484384; udid=854d35ee786dd208ca53bbf8d7bf7190; adtech_uid=cdd38139-1087-4a82-a10d-fad09c89cbc0%3Ainvesting.com; top100_id=t1.-1.1109927342.1665484387569; last_visit=1665448387571%3A%3A1665484387571; t3_sid_NaN=s1.1184232784.1665484387571.1665484387587.1.1.1.1; tmr_lvid=f04c2ffb9543f5d52eccf7ab9376872d; tmr_lvidTS=1665484391416; _ym_uid=1665484391864483144; _ym_d=1665484391; tmr_reqNum=6; PHPSESSID=255atufalen6v0k4qnm8834m89; geoC=RU; browser-session-counted=true; user-browser-sessions=2; gtmFired=OK; __cflb=02DiuGRugds2TUWHMkimMbdK71gXQtrngzLrqXfiyUNYt; protectedMedia=2; pms={"f":2,"s":2}; adbBLk=1; _gid=GA1.2.1570303324.1667159192; G_ENABLED_IDPS=google; r_p_s_n=1; reg_trk_ep=exit popup banner; editionPostpone=1667159197680; adsFreeSalePopUp=3; g_state={"i_p":1667169060380,"i_l":1}; smd=854d35ee786dd208ca53bbf8d7bf7190-1667168108; _gat_allSitesTracker=1; nyxDorf=OD9jMWY3MnA3YGBpZzRkeGUzMm4zKmFiZ28yOA%3D%3D; __cf_bm=V0D5VdMcp0eoiZG09KZUp_tuvMOerNYlUUkgC48hGQk-1667168113-0-AU9O3zSc5YHozPL5pThX0RnGWfv7y5jGWWuHR7Ks/YT0vl6rO77feB/ngjSJj+4b/igviY+N8TXpBZbbh+BFoXA=; _gat=1; invpc=9; _ga_C4NDLGKVMK=GS1.1.1667168113.2.1.1667168114.59.0.0; _ga=GA1.1.1494444067.1665484391; outbrain_cid_fetch=true' \
            -H 'origin: https://www.investing.com' \
            -H 'referer: https://www.investing.com/economic-calendar/' \
            -H 'sec-ch-ua: "Google Chrome";v="107", "Chromium";v="107", "Not=A?Brand";v="24"' \
            -H 'sec-ch-ua-mobile: ?0' \
            -H 'sec-ch-ua-platform: "Windows"' \
            -H 'sec-fetch-dest: empty' \
            -H 'sec-fetch-mode: cors' \
            -H 'sec-fetch-site: same-origin' \
            -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36' \
            -H 'x-requested-with: XMLHttpRequest' \
            '''

        curl_req += '''--data-raw 'country%5B%5D=5&currentTab=custom&limit_from={}&dateFrom={}&dateTo={}&last_time_scope={}' \
            --compressed
            '''.format(limit_from, start_date_str, end_date_str, last_time_scope)         

        result = os.popen(curl_req).read()
        contents = json.loads(result)    
        soup = BeautifulSoup(contents['data'])
        table = soup.findAll('tr')

        if not contents['rows_num']:
            break        

        for row in table:
            day = row.findAll('td', attrs={'class':'theDay'})
            if len(day):
                date = day[0].text
                date = datetime.strptime(date, '%A, %B %d, %Y')
                continue            

            symbol = row.find('a', attrs={'class':'bold middle'}).text
            if row.find('span', attrs={'class':'marketOpen'}):
                type = row.find('span', attrs={'class':'marketOpen'}).get_attribute_list('data-tooltip')[0]    
            elif row.find('span', attrs={'class':'marketClosed'}):
                type = row.find('span', attrs={'class':'marketClosed'}).get_attribute_list('data-tooltip')[0]
            else:
                type = 'unknown'

            event = {'Date': date, 'Symbol': symbol, 'Type': type}
            ec.append(event)

        # if scroll handler True, still next batch to fetch
        is_fetch_all = not contents['bind_scroll_handler']

        # this is needed to query next betch
        last_time_scope = contents['last_time_scope']                
        limit_from += 1        

    cnt = len(ec)
    logger.info(f"Fetched {cnt} names from Invest.com Earning Calendar: {start_date_str} - {end_date_str}.")

    if cnt:
        ec = pd.DataFrame(ec)
        upsert_earnings_calendar_file(ec, 'investcom', 'Date')
    else:
        ec = pd.DataFrame(columns=['Date', 'Symbol', 'Type'])

    return ec

if __name__ == '__main__':
    x = get_earnings_by_from_investcom(datetime(2023,10,18))        
    


    



    
    




        
         


    
