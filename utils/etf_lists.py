import requests
import pandas as pd
import json
import os
from datetime import datetime
from functools import cache

def get_etf_lists_from_etfdb(region=None, asset_class=None, investment_strategy=None, 
                             sector=None, size=None, commodity_type=None, commodity_exposure=None,
                             real_estate_sector=None):
    '''
        Get ETF List from ETFDB
        Args:
            region:                 U.S. (only for equity)
            asset_class:            equity, commodity
            investment_strategy:    Growth, Value, Blend
            size:                   Large-Cap, Mega-Cap, Micro-Cap, Mid-Cap, Multi-Cap, Small-Cap
            sector:                 ETF by sector
            commodity_type:         ETF by commodity type
    '''

    url = 'https://etfdb.com/api/screener/'

    payload = {}    
    payload['asset_class'] = [asset_class] if asset_class else []
    payload['sectors'] = [sector] if sector else []
    payload['regions'] = [region] if region else []
    payload['investment_strategies'] = [investment_strategy] if investment_strategy else []   
    payload['sizes'] = [size] if size else []
    payload['commodity_types'] = [commodity_type] if commodity_type else []
    payload['commodity_exposures'] = [commodity_exposure] if commodity_exposure else []
    payload['real_estate_sectors'] = [real_estate_sector] if real_estate_sector else []
    payload['page'] = 1    

    headers = {
        "Accept":"application/json",
        "Accept-Language":"en-GB,en-US;q=0.9,en;q=0.8",        
        "Content-Type": "application/json",       
        "Origin":"https://etfdb.com",
        "Referer":"https://etfdb.com/screener/",    
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    }

    payload_dump = json.dumps(payload)
    r = requests.post(url, data=payload_dump, headers=headers)
    data = json.loads(r.content)
    
    pages = data['meta']['total_pages']
    df_etf = []

    for page in range(1, pages+1):
        if page > 1:
            payload['page'] = page
            payload_dump = json.dumps(payload)
            r = requests.post(url, data=payload_dump, headers=headers)
            data = json.loads(r.content)                    
        
        for item in data['data']:    
            row = {}
            row['symbol'] = item['symbol']['text']
            row['name'] = item['name']['text']
            row['price'] = item['price']
            row['assets'] = item['assets']
            row['average_volume'] = item['average_volume']
            row['ytd'] = item['ytd']
            row['asset_class'] = item['asset_class']
            row['sector'] = sector
            row['region'] = region
            row['investment_strategy'] = investment_strategy
            row['size'] = size
            row['commodity_type'] = commodity_type
            row['commodity_exposure'] = commodity_exposure
            row['real_estate_sector'] = real_estate_sector

            df_etf.append(row)

    df_etf = pd.DataFrame(df_etf)
    df_etf['updated_time'] = datetime.today()

    return df_etf

@cache
def get_commodity_etf():
    '''
        Get Commodity ETF
    '''
    etf_commod = []

    commodity_types = [
        'Agriculture',
        'Diversified',
        'Energy',
        'Industrial Metals',
        'Livestock',
        'Precious Metals',
        'Softs',
    ]

    commodity_exposures = [
        'Futures-Based',
        'Physically-Backed',
    ]

    for c in commodity_types:
        etf = get_etf_lists_from_etfdb(asset_class='commodity', commodity_type=c)
        etf_commod.append(etf)

    for c in commodity_exposures:
        etf = get_etf_lists_from_etfdb(asset_class='commodity', commodity_exposure=c)
        etf_commod.append(etf)

    etf_commod = pd.concat(etf_commod)
    etf_commod = etf_commod.groupby(['symbol']).first().reset_index()

    return etf_commod

@cache
def get_real_estate_etf():
    '''
        Get Real Estate ETF
    '''
    etf_real = []
    real_estate_sectors = [
        'Real Estate-Broad',
        'Industrial/Office Real Estate',
        'Mortgage REITs',
        'Residential Real Estate',
        'Retail Real Estate'
    ]

    for r in real_estate_sectors:
        etf = get_etf_lists_from_etfdb(asset_class='real estate', real_estate_sector=r)
        etf_real.append(etf)

    etf_real = pd.concat(etf_real)
    return etf_real

@cache
def get_equity_etf():
    '''
        Get Equity ETF by sizes, styles and sectors
    '''

    sizes = [
        'Large-Cap',
        'Mega-Cap',
        'Micro-Cap',
        'Mid-Cap',
        'Multi-Cap',
        'Small-Cap',
    ]

    styles = [
        'Blend',
        'Growth',
        'Value',
    ]

    sectors = [
        'Consumer Discretionary',
        'Consumer Staples',
        'Energy',
        'Financials',
        'Healthcare',
        'Industrials',
        'Materials',
        'Technology',
        'Telecom',
        'Utilities',    
    ]

    etf_equity = []

    for size in sizes:
        etf = get_etf_lists_from_etfdb(asset_class='equity', size=size)
        etf_equity.append(etf)

    for style in styles:
        etf = get_etf_lists_from_etfdb(asset_class='equity', investment_strategy=style)
        etf_equity.append(etf)

    for sector in sectors:
        etf = get_etf_lists_from_etfdb(asset_class='equity', sector=sector)
        etf_equity.append(etf)

    etf_equity = pd.concat(etf_equity)
    etf_equity = etf_equity.groupby(['symbol']).first().reset_index()

    return etf_equity

def save_all_etf():
    '''
        Save the ETF to hdf5 
    '''
    etf_equity = get_equity_etf()
    etf_commod = get_commodity_etf()
    etf_real = get_real_estate_etf()
    df = pd.concat([etf_equity, etf_commod, etf_real])    

    file = os.path.dirname(os.path.realpath(__file__)) + "/Data/etf_lists.h5"
    df.to_hdf(file, key='etfdb', format='table')

def get_all_etf():
    '''
        Read stored ETF from hdf5
    '''
    file = os.path.dirname(os.path.realpath(__file__)) + "/Data/etf_lists.h5"
    df = pd.read_hdf(file, key='etfdb', mode='r')
    return df

    






