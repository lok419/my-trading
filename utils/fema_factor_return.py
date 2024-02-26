
import pandas as pd
from functools import cache

@cache
def get_daily_ff3_returns():
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    ff3 = pd.read_csv(url, compression='zip', skiprows=3, skipfooter=1, engine='python')
    ff3 = ff3.rename(columns={'Unnamed: 0': 'Date'})
    ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m%d')
    ff3['Mkt-RF'] /= 100
    ff3['SMB'] /= 100
    ff3['HML'] /= 100
    ff3['RF'] /= 100    
    return ff3

@cache
def get_monthly_ff3_returns():
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip'
    ff3 = pd.read_csv(url, compression='zip', skiprows=3, skipfooter=1, engine='python')
    ff3 = ff3.rename(columns={'Unnamed: 0': 'Date'})
    ff3 = ff3[ff3['Date'].apply(lambda x: len(str(x).strip()) == 6)]
    ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m')
    ff3['Mkt-RF'] = ff3['Mkt-RF'].apply(lambda x: float(x)/100)
    ff3['SMB'] = ff3['SMB'].apply(lambda x: float(x)/100)
    ff3['HML'] = ff3['HML'].apply(lambda x: float(x)/100)
    ff3['RF'] = ff3['RF'].apply(lambda x: float(x)/100)
    ff3 = ff3.set_index(['Date'])
    ff3 = ff3.resample('M').last().reset_index()
    return ff3

