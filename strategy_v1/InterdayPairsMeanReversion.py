from itertools import cycle
from sys import displayhook
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from strategy_v1 import InterdayStrategyBase
from plotly.subplots import make_subplots
from utils.data_helper import add_bday, get_today
from utils.performance import cumulative_return, performance_summary_table
from utils.stats import *
from IPython.display import display

class InterdayPairsMeanReversion(InterdayStrategyBase):

    def __init__(self, strategy_name="Interday Pairs Mean Reversion"):
        super().__init__(strategy_name)   

        # Defalt Parameters
        self.coint_windows = 252        
        self.rebal_blackout = 10        

        self.z_rolling_windows = 180       
        self.z_entry = 1         
        self.z_exit = 0

        # These are threshold used to preprocess the time-series to determine if this is mearn-reverting. This is intetend to be configured as loose threshold
        self.johanson_ci = 0.90
        self.half_life_threshold = 40
        self.hurst_threshold = 0.5

        # These are the threshold control the stop loss
        self.z_stop = 10
        self.non_mean_revert_days_stop = 60


    def display_params(self):
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        
        self.logger.info(f'capital:                     {self.capital}')
        self.logger.info(f'is_backtest:                 {self.is_backtest}')        
        self.logger.info(f'open_period_to_use_close:    {self.open_period_to_use_close}')
        self.logger.info(f'coint_windows:               {self.coint_windows}')        
        self.logger.info(f'rebal_blackout:              {self.rebal_blackout}')
        self.logger.info(f'z_rolling_windows:           {self.z_rolling_windows}')
        self.logger.info(f'z_stop:                      {self.z_stop}')        
        self.logger.info(f'z_entry:                     {self.z_entry}')        
        self.logger.info(f'z_exit:                      {self.z_exit}')        
        self.logger.info(f'johanson_ci:                 {self.johanson_ci}')        
        self.logger.info(f'half_life_threshold:         {self.half_life_threshold}') 
        self.logger.info(f'hurst_threshold:             {self.hurst_threshold}') 
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")    

    def mean_reverting_precheck(self, series):        
        '''
            Check of the series cointegrated or not, three checkings
                1. If there is convergence error in johansen test
                2. If half life less than a threshold
                3. If hurst ratio smaller than a threshold
        '''                
        self.mean_reverting = True        
        try:
            evec, r = time_series_coint_johansen(series, ci=self.johanson_ci)
            pv = np.dot(series, evec)
            half_life = time_series_half_life(pv)
            hurst = time_series_hurst_exponent(pv)
            self.mean_reverting = r >= 1 and self.mean_reverting_quick_check(half_life, hurst)
        except:
            self.mean_reverting = False   

    def mean_reverting_quick_check(self, half_life, hurst):
        '''
            Check two things:
                1. If half life less than a threshold
                2. If hurst ratio smaller than a threshold
        '''                
        return half_life < self.half_life_threshold and hurst < self.hurst_threshold
    
    def trade_position_size(self, z, pos):

        '''
            The determine the next position size given current zscore and position
            we want to average out the trade position

            A Non-Linear function given zscore might make senses here, as we want to trade more near the Z entry, and less then Z is far away from Z entry
        '''

        # Here define how we trade the Zscore (averaging cost)
        z_entry_max = 5
        z_entry_pos = 0.5
        z_entry_pos_max = 1.5

        z_dist = max(min(abs(z), z_entry_max) - self.z_entry, 0)
        pos_size = z_entry_pos + (z_entry_pos_max - z_entry_pos) / (z_entry_max - self.z_entry) * z_dist
        pos_size = max(pos_size, abs(pos))
        return pos_size
        
    def preprocess_data(self, px=None):
        """
            Core logic of the strategy, the function prepares all nescessary data to generate the daily positions

            Args:            
                px: in some case, we want to feed the price rather than calling load_price_data
        """         
        if px is not None:
            self.px = px[self.start_date: self.end_date]
        else:
            self.px = self.load_price_data(self.start_date, self.end_date)

        symbols = self.px['Close'].columns
        assert(len(symbols) == len(self.stock_universe))

        # this is important because the orders of stock price from Yahoo Finance could be different from input
        self.stock_universe = symbols

        px_close = self.px['Close']
        series = px_close.iloc[:self.coint_windows]        
        self.mean_reverting_precheck(series)

        return self     
    
    def generate_position(self):
        """
            Core logic of the strategy, the function generates positions data based on output from preprocess_data()            
        """

        dates = self.px.index
        position = pd.DataFrame(columns=self.stock_universe, index=dates).fillna(0)[self.coint_windows:]

        if not self.mean_reverting:
            self.position = position
            self.df_stats = pd.DataFrame()
            return self

        df_stats = []        
        should_rebal = True
        rebal_blackout_count = self.rebal_blackout

        cur_pos = 0
        last_rebal_date = None
        last_trade_date = None    
        last_exit_date = None                    

        last_rebal_reason = ""        
        last_rebal_reason_temp = ""
        last_exit_reason = ""               
        rolling_not_mean_revert = 0

        px_close = self.px['Close']        
        
        for i, d in enumerate(dates[self.coint_windows:]):            
            '''
                1. Rebalance Logic to regenerate the eigenvectors
            '''
            if should_rebal and rebal_blackout_count >= self.rebal_blackout:

                # normal price cointegration                
                series = px_close.iloc[i:i+self.coint_windows]                

                # check if mean reverting when we rebalance the portfolios based on cointegration windows        
                self.mean_reverting_precheck(series)                

                # if not mean reverting anymore, stop the pairs and use the old eigenvectors
                if not self.mean_reverting:     
                    self.logger.info('Price series is not cointegrated, do not trade and refresh the eigenvector: {}'.format(d.strftime('%Y-%m-%d'))) 
                else:
                    evec, r = time_series_coint_johansen(series, ci=self.johanson_ci)
                    self.logger.info('Rebalance portfolio based on Johanson Test at rank {}: {}'.format(r, d.strftime('%Y-%m-%d')))

                    # update rebal stats
                    last_rebal_date = d
                    last_rebal_reason = last_rebal_reason_temp

                    rolling_pv = np.dot(px_close[add_bday(d, -self.z_rolling_windows):d], evec)
                    rolling_pv_mean = rolling_pv.mean()
                    rolling_pv_std = rolling_pv.std()

                # reset all rebal parameters
                should_rebal = False
                rebal_blackout_count = 0                                                                                                    
                        
            '''
                2. Collect all Mean Reverting Stats
            '''
            rolling_pv = np.dot(px_close[add_bday(d, -self.z_rolling_windows):d], evec)            
            rolling_half_life = time_series_half_life(rolling_pv)
            rolling_hurst = time_series_hurst_exponent(rolling_pv)            
            rolling_not_mean_revert = 0 if self.mean_reverting_quick_check(rolling_half_life, rolling_hurst) else rolling_not_mean_revert + 1

            cur_px = px_close.loc[d]
            cur_pv = np.dot(cur_px, evec)
            cur_pv_pos = np.dot(cur_px, evec * (evec > 0))
            cur_pv_neg = np.dot(cur_px, evec * (evec < 0))            
            cur_z = (cur_pv - rolling_pv_mean) / rolling_pv_std                                    

            '''
                3. Core Trade Logic
            '''
            should_rebal = False

            # 1. This is the scenario that we fail to rebalance because the series is not cointegrated anymore
            if not self.mean_reverting:
                cur_pos = 0  
                should_rebal = True                 
                        
            elif cur_pos == 0 and rolling_not_mean_revert > 0:
                cur_pos = 0                
                should_rebal = True                                              
                last_rebal_reason_temp = 'Non MR'

            # 2. Stop loss, trigger the rebalance
            elif cur_pos != 0 and abs(cur_z) > self.z_stop:
                cur_pos = 0
                last_trade_date = last_exit_date = d
                should_rebal = True              
                last_exit_reason = 'Stop Loss (Z-Score)'                       
                last_rebal_reason_temp = 'Stop Loss (Z-Score)'

            # 3. Stop loss, when the series is non mean revert > x days
            elif cur_pos != 0 and rolling_not_mean_revert > self.non_mean_revert_days_stop:
                cur_pos = 0
                last_trade_date = last_exit_date = d
                should_rebal = True                                              
                last_exit_reason = f'Stop Loss (Non MR > {self.non_mean_revert_days_stop}d)'                 
                last_rebal_reason_temp = f'Stop Loss (Non MR > {self.non_mean_revert_days_stop}d)' 

            # 4. Enter the position
            elif abs(cur_z) > self.z_entry and abs(cur_z) < self.z_stop:
                trade_size = self.trade_position_size(cur_z, cur_pos)
                cur_pos = trade_size * np.sign(cur_z) * -1
                last_trade_date = d

            # 5. Exit the position            
            elif (cur_z < self.z_exit and cur_pos < 0) or (cur_z > -self.z_exit and cur_pos > 0):
                cur_pos = 0
                last_exit_reason = f'Exit @ {self.z_exit}'
                last_trade_date = last_exit_date = d

            # 6. If no nothing happens and cur_pos is zero, should rebalance
            elif cur_pos == 0:
                last_rebal_reason_temp = 'Regular'
                should_rebal = True

            position.loc[d] = cur_pos * evec
            rebal_blackout_count += 1

            '''
                4. Save all required daily trade data
            '''            
            row = {}            
            row['date'] = d
            row['rank'] = r
            row['is_mean_revert'] = self.mean_reverting
            row['rolling_not_mean_revert'] = rolling_not_mean_revert            
            row['portfolio_position'] = cur_pos
            row['pv'] = cur_pv
            row['pv_pos'] = cur_pv_pos
            row['pv_neg'] = cur_pv_neg
            row['pv_mean'] = rolling_pv_mean
            row['pv_std'] = rolling_pv_std
            row['pv_zscore'] = cur_z
            row['pv_half_life'] = rolling_half_life
            row['pv_hurst'] = rolling_hurst
            row['last_trade_date'] = last_trade_date        
            row['last_rebal_date'] = last_rebal_date        
            row['last_rebal_reason'] = last_rebal_reason    
            row['last_exit_date'] = last_exit_date
            row['last_exit_reason'] = last_exit_reason                                

            for i, symbol in enumerate(self.stock_universe):
                row[symbol] = cur_px[symbol]
                row[f'{symbol}_w'] = evec[i]

            df_stats.append(row)                             

        '''
            5. Generate position data in actual number of shares
        '''
        df_stats = pd.DataFrame(df_stats)
        pv_pos = df_stats[['date','pv_pos']]
        pv_pos = pv_pos.set_index('date')

        # position here represents number of shares which depends on the price when it entries, we assume no daily rebalance
        position_entry = (position != 0) & (position.abs().diff() != 0)
        position_hold = (position != 0) & (position.abs().diff() == 0)
        
        position[self.stock_universe] = np.where(position_entry, np.abs(self.capital/pv_pos.values) * position, position)
        position[self.stock_universe] = np.where(position_hold, np.nan, position)
        position = position.fillna(method='ffill')     

        self.df_stats = df_stats
        self.position = position
                        
        return self  

    def plot_price_time_series(self):
        fig = make_subplots(1,1, subplot_titles=['Price'], vertical_spacing=0.03)
        fig.update_layout(width=1200, height=350, title=','.join(self.stock_universe), hovermode='x')
        colors = cycle(plotly.colors.DEFAULT_PLOTLY_COLORS)
        px = self.px['Close']

        for s in self.stock_universe:
            c = colors.__next__()
            fig.add_trace(go.Scatter(x=px.index, y=px[s], name=s, marker=dict(color=c), legendgroup=s), row=1, col=1) 

        fig.add_vline(x=add_bday(self.start_date,self.coint_windows), line_dash="dash", line_color="red", line_width=2)
        fig.show()                           
    
    def backtest_summary(self, benchmark=['^SPX', '^IXIC'], show_stop_loss=False, show_rebal=False, show_annotation=False):
        """
            Generate backtest summary given positions          

            if series is not mean reverting, just plot the prices to show how NOT mean reverting is it                                      
        """
        if self.df_stats.empty:
            self.plot_price_time_series()
            return

        self.generate_backtest_return()
        display(performance_summary_table({self.strategy_name: self.port_ret}, benchmark))

        fig = make_subplots(
            8,1, 
            subplot_titles=['Price', 'Portfolio Value', 'Rolling Z Score', 'Portfolio Position', 'Cumulative Return', 'Daily Return', 'Half-Life', 'Hurst Exponent'], 
            vertical_spacing=0.03,
            shared_xaxes=True
        )
        fig.update_layout(width=1200, height=2000, title=','.join(self.stock_universe), hovermode='x')
        fig.update_layout(
            xaxis_showticklabels=True, 
            xaxis2_showticklabels=True, 
            xaxis3_showticklabels=True,
            xaxis4_showticklabels=True,
            xaxis5_showticklabels=True,
            xaxis6_showticklabels=True,
            xaxis7_showticklabels=True,
        )

        colors = cycle(plotly.colors.DEFAULT_PLOTLY_COLORS)
        px = self.px['Close'].loc[self.position.index]

        for s in self.stock_universe:
            c = colors.__next__()
            fig.add_trace(go.Scatter(x=px.index, y=px[s], name=s, marker=dict(color=c), legendgroup=s), row=1, col=1)    

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS

        fig.add_trace(go.Scatter(x=self.df_stats['date'], y=self.df_stats['pv'], marker=dict(color=colors[0]), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df_stats['date'], y=self.df_stats['pv_zscore'], marker=dict(color=colors[0]), showlegend=False), row=3, col=1)

        fig.add_hline(y=self.z_exit, row=3, col=1, line_dash="dash", line_color="red", line_width=1)
        fig.add_hline(y=-self.z_exit, row=3, col=1, line_dash="dash", line_color="red", line_width=1)

        fig.add_hline(y=self.z_entry, row=3, col=1, line_dash="dash", line_color="green", line_width=1)        
        fig.add_hline(y=-self.z_entry, row=3, col=1, line_dash="dash", line_color="green", line_width=1)

        if self.z_stop < 10:
            fig.add_hline(y=self.z_stop, row=3, col=1, line_dash="dash", line_color="red", line_width=1)
            fig.add_hline(y=-self.z_stop, row=3, col=1, line_dash="dash", line_color="red", line_width=1)

        fig.add_trace(go.Scatter(x=self.df_stats['date'], y=self.df_stats['portfolio_position'], marker=dict(color=colors[0]), showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.port_ret.index, y=cumulative_return(self.port_ret), showlegend=False, marker=dict(color=colors[0])), row=5, col=1)
        fig.add_trace(go.Scatter(x=self.port_ret.index, y=100*self.port_ret, showlegend=False, marker=dict(color=colors[0])), row=6, col=1)

        fig.add_trace(go.Scatter(x=self.df_stats['date'], y=self.df_stats['pv_half_life'], showlegend=False, marker=dict(color=colors[0])), row=7, col=1)
        fig.add_trace(go.Scatter(x=self.df_stats['date'], y=self.df_stats['pv_hurst'], showlegend=False, marker=dict(color=colors[0])), row=8, col=1)
        fig.add_hline(y=self.half_life_threshold, row=7, col=1, line_dash="dash", line_color="green", line_width=3)
        fig.add_hline(y=self.hurst_threshold, row=8, col=1, line_dash="dash", line_color="green", line_width=3)        

        fig['layout']['yaxis']['title'] = '$'
        fig['layout']['yaxis2']['title'] = '$'
        fig['layout']['yaxis5']['title'] = '%'
        fig['layout']['yaxis6']['title'] = '%'
        fig['layout']['yaxis7']['title'] = 'Days'

        # add lines to indicates rebalance and its reason (first rebal is always the first day, so we want to skip it)

        if show_rebal:
            last_rebal = self.df_stats.groupby(['last_rebal_date']).last()['last_rebal_reason'].iloc[1:]
            for d, reason in last_rebal.items():            
                if show_annotation:
                    fig.add_vline(x=d.timestamp() * 1000, row='all', col=1, line_dash="dash", line_color="black", line_width=2, annotation_text=reason)
                else:
                    fig.add_vline(x=d.timestamp() * 1000, row='all', col=1, line_dash="dash", line_color="black", line_width=2)            
                    

        # add lines to indicates stop loss and its reason
        if show_stop_loss:
            last_stop = self.df_stats.groupby(['last_exit_date']).last()['last_exit_reason']
            for d, reason in last_stop.items():            
                if show_annotation:
                    fig.add_vline(x=d.timestamp() * 1000, row='all', col=1, line_dash="dash", line_color="red", line_width=2, annotation_text=reason)                                
                else:
                    fig.add_vline(x=d.timestamp() * 1000, row='all', col=1, line_dash="dash", line_color="red", line_width=2)

        # defein the height of each annotations so it doesn't overlap
        if (show_rebal or show_stop_loss) and show_annotation:
            annotations = fig.to_dict()["layout"]["annotations"]
            annotations_others = list(filter(lambda x: x['xref'] == x['yref'] == 'paper', annotations))
            annotations_ytext = list(filter(lambda x: not(x['xref'] == x['yref'] == 'paper'), annotations))
            all_x_vals = sorted(list(set(map(lambda x: x['x'], annotations_ytext))))        

            for i in range(len(annotations_ytext)):    
                annotations_ytext[i]['y'] = 1 - all_x_vals.index(annotations_ytext[i]['x']) / len(all_x_vals)        
            annotations = annotations_others + annotations_ytext
            fig.update_layout(annotations=annotations)
        
        fig.show()

        self.fig = fig
    
    def actual_trade(self, date=get_today()):
        """
            Return the actual position for day trading
            There must be some environmental different from backtest, so we create a new function here
        """        
        pass

    


