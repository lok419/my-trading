import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from strategy_v1.IntradayStrategyBase import IntradayStrategyBase
from utils.logging import get_logger
from utils.data_helper import add_bday, get_today
from utils.earnings_calendar import get_earnings_by_from_investcom, get_earnings_calendar_file
from scipy.stats import pearsonr

class PostEarningsAnnouncementDrift(IntradayStrategyBase):

    def __init__(self, strategy_name="Post Earnings Announcement Drift"):        
        super().__init__(strategy_name)   

        # Defalt Parameters
        self.earnings_lookback = -1
        self.z_threshold = 0.5
        self.beta_threshold = 0.3
        self.min_trades = 5
        self.rolling_windows = 60     
        self.comms_per_round = 4   

    def display_params(self):
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        
        self.logger.info(f'capital:                     {self.capital}')
        self.logger.info(f'is_backtest:                 {self.is_backtest}')            
        self.logger.info(f'open_period_to_use_close:    {self.open_period_to_use_close}')
        self.logger.info(f'earnings_lookback:           {self.earnings_lookback}')
        self.logger.info(f'z_threshold:                 {self.z_threshold}')
        self.logger.info(f'beta_threshold:              {self.beta_threshold}')
        self.logger.info(f'min_trades:                  {self.min_trades}')
        self.logger.info(f'rolling_windows:             {self.rolling_windows}')        
        self.logger.info(f'comms_per_round:             ${self.comms_per_round}')        
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")

    def set_rolling_windows(self, v):
        """
            Rolling window for Zscore
        """    
        self.rolling_windows = v
        return self

    def set_earnings_lookback(self, v):
        """
            Number of earnings event to lookback when calculating the beta, -1 means for looks for all earnings events
        """    
        self.earnings_lookback = v
        return self

    def set_z_threshold(self, v):
        """
            Z Score Threshold to trade
        """    
        self.z_threshold = v
        return self

    def set_beta_threshold(self, v):
        """
            Beta Threshold to trade
        """    
        self.beta_threshold = v
        return self

    def set_min_trades(self, v):
        """
            Minimimum number of open trades per day
            If your position is less than min_trades, your capital per stocks discounted by open_position / min_trades
            This is to prevent to over-allocate captial to one stocks if there is only one position a day
        """    
        self.min_trades = v
        return self        
        
    def preprocess_data(self):
        """
            Core logic of the strategy, the function prepares all nescessary data to generate the daily positions

            The output from this function is also useful to analyize
        """ 
        self.display_params()
        self.logger.info('Preprocessing - Building a earnings release dataset.....')

        df_ec = get_earnings_calendar_file('investcom')
        df_ec = df_ec[df_ec['Symbol'].isin(self.stock_universe)]

        symbols = df_ec['Symbol'].unique()
        missing_symbol = np.setdiff1d(self.stock_universe, symbols)

        # We need to get all price which covers the periods in our earnings calendar file
        start_date = df_ec['Date'].min()
        end_date = self.end_date

        df_ec = df_ec[df_ec['Date'] <= end_date]

        self.logger.info('Stock Universe: {}'.format(len(self.stock_universe)))
        self.logger.info('Stored Names: {}'.format(len(df_ec['Symbol'].unique())))
        self.logger.info('Missing Symbols: {}'.format(missing_symbol))
        self.px = self.load_price_data(start_date, end_date)
        
        df_stats = []        
        symbols = df_ec['Symbol'].unique()

        with tqdm(total = len(symbols)) as pbar:
            for symbol in symbols:
                symbol_ec = df_ec.copy()
                symbol_ec = symbol_ec[symbol_ec['Symbol'] == symbol]        
                symbol_ec = symbol_ec.drop_duplicates(subset=['Date'])

                open_px = self.px[('Open', symbol)]
                close_px = self.px[('Close', symbol)]
                close_ret = close_px.pct_change()

                close_ret_std = close_ret.rolling(self.rolling_windows).std()
                close_ret_mean = close_ret.rolling(self.rolling_windows).mean()            

                # Assume all earnigns are released after market close
                for _, e in symbol_ec.iterrows():
                    type = e['Type']
                    earn_date = e['Date']

                    # after market close - measures return between T Close and T+1 Open
                    # before market open - measures return between T-1 Close and T Open
                    if type == 'After market close':
                        d_next = add_bday(earn_date, 1)
                        d_prev = add_bday(add_bday(earn_date, 1), -1)
                        
                    elif type == 'Before market open':
                        d_next = add_bday(add_bday(earn_date, -1), 1)
                        d_prev = add_bday(earn_date, -1)                
                    else:
                        continue

                    prev_close = close_px.get(d_prev, np.nan)
                    next_close = close_px.get(d_next, np.nan)
                    next_open = open_px.get(d_next, np.nan)
                    ret = (next_open - prev_close) / prev_close

                    mean = close_ret_mean.get(d_prev, np.nan)
                    std = close_ret_std.get(d_prev, np.nan)
                    z = (ret - mean) / std

                    row = {
                        'symbol': symbol,
                        'date': earn_date,
                        'd_prev': d_prev,
                        'd_next': d_next,
                        'type': type,
                        'prev_close': prev_close,
                        'next_open': next_open,
                        'next_close': next_close,
                        'z': z,
                        'rolling_mean': mean,
                        'rolling_std': std,     
                        'position_date': d_next,                    
                    }                       
                    df_stats.append(row)
                pbar.update(1)  

        df_stats = pd.DataFrame(df_stats)
        df_stats['open_ret'] = (df_stats['next_open'] - df_stats['prev_close']) / df_stats['prev_close']
        df_stats['intra_ret'] = (df_stats['next_close'] - df_stats['next_open']) / df_stats['next_open']      
        df_stats['quarter'] = df_stats['date'].apply(lambda x: '{}Q{}'.format(x.year, x.quarter))
        self.df_stats = df_stats
        return self
    
    def generate_position(self):
        """
            Core logic of the strategy, the function generates positions data based on output from preprocess_data()
        """
        self.logger.info('Generating positions.....')

        df_stats = self.df_stats        
        signals = pd.DataFrame(index=self.px.index, columns=self.px['Close'].columns).fillna(0)

        symbols_test = self.df_stats['symbol'].unique()
        df_stats_pos = []

        with tqdm(total=len(symbols_test)) as pbar:
            for symbol in symbols_test:
                df_hist = df_stats[df_stats['symbol'] == symbol]
                df_test = df_hist.copy()        
                df_test = df_test[df_test['position_date'] >= self.start_date]
                df_test = df_test[df_test['position_date'] <= self.end_date]            

                for _, row in df_test.iterrows():
                    d = row['date']
                    d_next = row['d_next']
                    z = row['z']

                    prev_e = df_hist[df_hist['date'] < d]            
                    prev_e = prev_e[~prev_e['open_ret'].isnull()]
                    prev_e = prev_e[~prev_e['intra_ret'].isnull()]
                    prev_e = prev_e.sort_values('date', ascending=False)            
                    
                    if len(prev_e) < self.earnings_lookback:
                        continue
                    
                    if self.earnings_lookback > 0:
                        prev_e = prev_e.head(self.earnings_lookback)            
                    
                    x = prev_e['open_ret'].values
                    y = prev_e['intra_ret'].values

                    try:
                        #beta = np.polyfit(x, y, 1)[0]
                        beta = pearsonr(x,y)[0]
                    except Exception as e:                         
                        self.logger.error('Error calculating beta: {}, {}'.format(symbol, e))                
                        continue

                    row = {}
                    row['date'] = d
                    row['symbol'] = symbol
                    row['beta'] = beta                            

                    if beta < self.beta_threshold and beta > -self.beta_threshold:
                        df_stats_pos.append(row)
                        continue
                                    
                    if z > self.z_threshold:
                        signals.loc[d_next, symbol] = 1 if beta > 0 else -1                                                

                    elif z < -self.z_threshold:
                        signals.loc[d_next, symbol] = -1 if beta > 0 else 1                                            
                    
                    df_stats_pos.append(row)   
                        
                pbar.update(1)

        # normalized the position
        weights = signals.divide(np.maximum(signals.abs().sum(axis=1), self.min_trades), axis='rows')
        position = self.capital * weights / self.px['Open']

        df_stats_pos = pd.DataFrame(df_stats_pos)    
        df_stats = df_stats[df_stats['position_date'] >= self.start_date]  
        df_stats = df_stats[df_stats['position_date'] <= self.end_date]

        df_trade_stats = pd.merge(df_stats, df_stats_pos, how='left', on=['date', 'symbol'], validate='1:1')

        df_trade_stats['signal'] = df_trade_stats.apply(lambda x: signals.loc[x['d_next'], x['symbol']], axis=1)        
        df_trade_stats['weight'] = df_trade_stats.apply(lambda x: weights.loc[x['d_next'], x['symbol']], axis=1)        
        df_trade_stats['position'] = df_trade_stats.apply(lambda x: position.loc[x['d_next'], x['symbol']], axis=1)        
        df_trade_stats['shares'] = df_trade_stats['position'].round()
        df_trade_stats['allocated_capital'] = df_trade_stats['shares'] * df_trade_stats['next_open']

        self.df_trade_stats = df_trade_stats
        self.position = position[self.start_date:self.end_date]
        return self    
    
    def actual_trade(self, date=get_today()):
        """
            Return the actual position for day trading
            There must be some environmental different from backtest, so we create a new function here
        """        
        self.set_is_backtest(False)
        d_today = add_bday(add_bday(date, -1), 1)
        d_prev = add_bday(d_today, -1)

        # snap all earnings data
        get_earnings_by_from_investcom(d_today, refresh=True)
        get_earnings_by_from_investcom(d_prev, refresh=True)
        
        df_ec = get_earnings_calendar_file('investcom')
        df_ec_today = df_ec.copy()
        df_ec_today = df_ec_today[df_ec_today['Symbol'].isin(self.stock_universe)]

        today_before_mkt = (df_ec_today['Date'] == d_today)&(df_ec_today['Type'] == 'Before market open')
        prev_d_after_mkt = (df_ec_today['Date'] < d_today)&(df_ec_today['Date'] >= d_prev)&(df_ec_today['Type'] == 'After market close')

        symbols_trade_today = list(df_ec_today[today_before_mkt|prev_d_after_mkt]['Symbol'].unique())
        df_ec_today = df_ec_today[df_ec_today['Symbol'].isin(symbols_trade_today)]                

        # Here we just get all earnings data which we are going to trade today to speed up the time
        self.set_stock_universe(symbols_trade_today)                
        self.set_start_date(date)
        self.set_end_date(date)
        self.preprocess_data()
        self.generate_position()


if __name__ == "__main__":
    pass

    # start_date = datetime(2021,1,1)
    # end_date = add_bday(get_today(), 0)

    # strategy = PostEarningsAnnouncementDrift()    
    # strategy.set_stock_universe()
    # strategy.set_backtest_start_date(start_date)
    # strategy.set_backtest_start_date(end_date)
    # strategy.preprocess_data()
    # strategy.generate_position()
    # strategy.backtest_summary()

    


