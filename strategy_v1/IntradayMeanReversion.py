import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from strategy_v1.IntradayStrategyBase import IntradayStrategyBase
from utils.logging import get_logger
from utils.data_helper import add_bday, get_today
from pandas.tseries.offsets import BDay

class IntradayMeanReversion(IntradayStrategyBase):

    def __init__(self, strategy_name="Intraday Mean Reversion"):
        super().__init__(strategy_name)   

        # Defalt Parameters
        self.z_rolling_windows = 60
        
        # Z Score Threshold to trade        
        self.z_threshold = 1
        
        # number of lookback days to compute the correlation
        self.corr_rolling_windows = 10
        
        # Beta Threshold to trade                
        self.beta_threshold = 0.3

        """
            Minimimum number of open trades per day
            If your position is less than min_trades, your capital per stocks discounted by open_position / min_trades
            This is to prevent to over-allocate captial to one stocks if there is only one position a day
        """  
        self.min_trades = 5
        self.max_trades = 5          
        self.adv_min = 50e6

    def display_params(self):
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        
        self.logger.info(f'capital:                     {self.capital}')
        self.logger.info(f'is_backtest:                 {self.is_backtest}')        
        self.logger.info(f'open_period_to_use_close:    {self.open_period_to_use_close}')
        self.logger.info(f'z_rolling_windows:           {self.z_rolling_windows}')
        self.logger.info(f'z_threshold:                 {self.z_threshold}')
        self.logger.info(f'corr_rolling_windows:        {self.corr_rolling_windows}')
        self.logger.info(f'beta_threshold:              {self.beta_threshold}')
        self.logger.info(f'min_trades:                  {self.min_trades}')        
        self.logger.info(f'max_trades:                  {self.max_trades}')        
        self.logger.info(f'comms_per_round:             ${self.comms_per_round}')   
        self.logger.info(f'adv_min:                     ${self.adv_min}')   
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")   
        
    def preprocess_data(self):
        """
            Core logic of the strategy, the function prepares all nescessary data to generate the daily positions

            The output from this function is also useful to analyize
        """ 
        self.display_params()
        self.logger.info('Preprocessing - Building a rolling correlation dataset.....')

        self.px = self.load_price_data(self.start_date - BDay(252*10), self.end_date)
        symbols = list(self.px['Close'].columns)        

        l2o_corr_matrix = pd.DataFrame(index=self.px.index)
        h2o_corr_matrix = pd.DataFrame(index=self.px.index)

        # df_stats = []

        open_px = self.px['Open']
        close_px = self.px['Close']
        low_px = self.px['Low']
        high_px = self.px['High']

        c2c_ret = close_px / close_px.shift(1) - 1
        intra_ret = close_px / open_px - 1

        # shift by one days as we trade at market opens, there is no way we know the close price of the days
        c2c_ret_mean = c2c_ret.rolling(self.z_rolling_windows).mean().shift(1)
        c2c_ret_std = c2c_ret.rolling(self.z_rolling_windows).std().shift(1)    

        # Get the Low to Open Return
        l2o_ret = open_px / low_px.shift(1) - 1
        l2o_ret_z = (l2o_ret - c2c_ret_mean) / c2c_ret_std

        # Get the High to Open Return
        h2o_ret = open_px / high_px.shift(1) - 1
        h2o_ret_z = (h2o_ret - c2c_ret_mean) / c2c_ret_std  

        # Low to Open Return and intraday return if low to open returns < threshold
        l2o_ret_exceed_th = l2o_ret[l2o_ret_z < -self.z_threshold]
        l2o_intra_ret_exceed_th = intra_ret[l2o_ret_z < -self.z_threshold]    

        # High to Open Return and intraday return if high to open returns > threshold
        h2o_ret_exceed_th = h2o_ret[h2o_ret_z > self.z_threshold]
        h2o_intra_ret_exceed_th = intra_ret[h2o_ret_z > self.z_threshold]

        # FIXME: The iteration is the performance bottlenect, ideally should fix it
        with tqdm(total=len(symbols)) as pbar:
            for symbol in symbols:

                # correlation when low to open returns < threshold
                l2o_ret_exceed_th_symbol = l2o_ret_exceed_th[symbol]
                l2o_ret_exceed_th_symbol = l2o_ret_exceed_th_symbol[~l2o_ret_exceed_th_symbol.isnull()]
                l2o_intra_ret_exceed_th_symbol = l2o_intra_ret_exceed_th[symbol]
                l2o_intra_ret_exceed_th_symbol = l2o_intra_ret_exceed_th_symbol[~l2o_intra_ret_exceed_th_symbol.isnull()]
                l2o_corr = l2o_ret_exceed_th_symbol.rolling(self.corr_rolling_windows).corr(l2o_intra_ret_exceed_th_symbol)

                # correlation when high to open returns > threshold
                h2o_ret_exceed_th_symbol = h2o_ret_exceed_th[symbol]
                h2o_ret_exceed_th_symbol = h2o_ret_exceed_th_symbol[~h2o_ret_exceed_th_symbol.isnull()]
                h2o_intra_ret_exceed_th_symbol = h2o_intra_ret_exceed_th[symbol]
                h2o_intra_ret_exceed_th_symbol = h2o_intra_ret_exceed_th_symbol[~h2o_intra_ret_exceed_th_symbol.isnull()]
                h2o_corr = h2o_ret_exceed_th_symbol.rolling(self.corr_rolling_windows).corr(h2o_intra_ret_exceed_th_symbol)

                l2o_corr_matrix = l2o_corr_matrix.join(l2o_corr)
                h2o_corr_matrix = h2o_corr_matrix.join(h2o_corr)
                pbar.update(1)
        
        # shift by one days as we trade at market opens, we won't know the latest rolling correlation        
        l2o_corr_matrix = l2o_corr_matrix.fillna(method='ffill').shift(1)
        h2o_corr_matrix = h2o_corr_matrix.fillna(method='ffill').shift(1)

        self.l2o_corr_matrix = l2o_corr_matrix
        self.h2o_corr_matrix = h2o_corr_matrix

        return self        
    
    def generate_position(self):
        """
            Core logic of the strategy, the function generates positions data based on output from preprocess_data()
        """
        self.logger.info('Generating positions.....')        
        
        open_px = self.px['Open']
        close_px = self.px['Close']
        low_px = self.px['Low']
        high_px = self.px['High']  

        # we only know volume and close from T-1
        adv_20d = (self.px['Volume'] * self.px['Close']).rolling(20).mean().shift(1) 
        
        c2c_ret = close_px / close_px.shift(1) - 1
        intra_ret = close_px / open_px - 1    

        # shift by one days as we trade at market opens, there is no way we know the close price of the days
        c2c_ret_mean = c2c_ret.rolling(self.z_rolling_windows).mean().shift(1)
        c2c_ret_std = c2c_ret.rolling(self.z_rolling_windows).std().shift(1)            

        l2o_ret = open_px / low_px.shift(1) - 1
        l2o_ret_z = (l2o_ret - c2c_ret_mean) / c2c_ret_std

        h2o_ret = open_px / high_px.shift(1) - 1
        h2o_ret_z = (h2o_ret - c2c_ret_mean) / c2c_ret_std        

        long_pos_z_th = l2o_ret_z < -self.z_threshold    
        long_pos_beta_th = self.l2o_corr_matrix < -self.beta_threshold
        long_pos = 1 * (long_pos_z_th & long_pos_beta_th)        

        short_pos_z_th = h2o_ret_z > self.z_threshold
        short_pos_beta_th = self.h2o_corr_matrix < -self.beta_threshold
        short_pos = -1 * (short_pos_z_th & short_pos_beta_th)   

        # ADV filters, make sure the stock is liquid enough
        adv_filters = adv_20d > self.adv_min

        signals = (long_pos + short_pos) * adv_filters
        signals = signals[self.start_date:self.end_date]                

        symbols = list(self.px['Close'].columns)
        df_stats = []
        
        # FIXME: This is the performance bottlenect, ideally should fix it
        with tqdm(total=len(symbols)) as pbar:
            for symbol in symbols:

                open_px_symbol = open_px[symbol]
                close_px_symbol = close_px[symbol]
                high_px_symbol = high_px[symbol]
                low_px_symbol = low_px[symbol]
                c2c_ret_mean_symbol = c2c_ret_mean[symbol]
                c2c_ret_std_symbol = c2c_ret_std[symbol]
                l2o_ret_symbol = l2o_ret[symbol]
                l2o_ret_z_symbol = l2o_ret_z[symbol]
                l2o_corr_symbol = self.l2o_corr_matrix[symbol]
                h2o_ret_symbol = h2o_ret[symbol]
                h2o_ret_z_symbol = h2o_ret_z[symbol]                
                h2o_corr_symbol = self.h2o_corr_matrix[symbol]
                intra_ret_symbol = intra_ret[symbol]
                signals_symbol = signals[symbol]
                adv_symbol = adv_filters[symbol]

                ds = signals_symbol[signals_symbol != 0].index
                stats = pd.DataFrame({
                    'date': ds,
                    'symbol': symbol,
                    'signal': signals_symbol[ds],
                    'open': open_px_symbol[ds],
                    'close': close_px_symbol[ds],
                    'last_high': high_px_symbol.shift(1)[ds],
                    'last_low': low_px_symbol.shift(1)[ds],
                    'c2c_ret_mean': c2c_ret_mean_symbol[ds],
                    'c2c_ret_std': c2c_ret_std_symbol[ds],
                    'l2o_ret': l2o_ret_symbol[ds],
                    'l2o_ret_z': l2o_ret_z_symbol[ds],
                    'l2o_beta': l2o_corr_symbol[ds],
                    'h2o_ret': h2o_ret_symbol[ds],
                    'h2o_ret_z': h2o_ret_z_symbol[ds],
                    'h2o_beta': h2o_corr_symbol[ds],
                    'intra_ret': intra_ret_symbol[ds],
                    'adv': adv_symbol[ds],
                    'beta_used': np.where(signals_symbol[ds] > 0, l2o_corr_symbol[ds], h2o_corr_symbol[ds])
                })
                df_stats.append(stats)
                pbar.update(1)

        if len(df_stats) == 0:
            self.df_stats = self.df_trade_stats = pd.DataFrame()
            return

        df_stats = pd.concat(df_stats)        
        df_stats = df_stats.reset_index().drop(columns=['Date'])

        top_beta = df_stats.groupby(['date'])['beta_used'].nsmallest(self.max_trades)
        df_stats = pd.merge(df_stats, top_beta, how='inner', on=['beta_used'])

        total_trades = df_stats.groupby(['date']).size().to_frame('total_trades')
        df_stats = pd.merge(df_stats, total_trades, how='left', on=['date'])
        df_stats['weight'] = df_stats['signal'] / np.maximum(df_stats['total_trades'], self.min_trades)
        df_stats['position'] = self.capital * df_stats['weight'] / df_stats['open']
        df_stats['shares'] = np.round(df_stats['position'])
        df_stats['allocated_captial'] = df_stats['position'] * df_stats['open']

        # filling back the position
        position = pd.DataFrame(index=signals.index, columns=signals.columns).fillna(0)
        for _, r in df_stats.iterrows():
            position.loc[r['date'], r['symbol']] = r['position']

        self.df_stats = self.df_trade_stats = df_stats
        self.position = position
        
        return self    
    
    def actual_trade(self, date=get_today()):
        """
            Return the actual position for day trading
            There must be some environmental different from backtest, so we create a new function here
        """        
        self.set_is_backtest(False)
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

    


