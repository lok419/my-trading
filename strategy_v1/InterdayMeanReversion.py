from itertools import combinations
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from tqdm import tqdm
from strategy_v1 import InterdayPairsMeanReversion, InterdayStrategyBase
from plotly.subplots import make_subplots
from utils.data_helper import add_bday, get_today
from utils.performance import *
from utils.stats import *
from IPython.display import display

class InterdayMeanReversion(InterdayStrategyBase):
    '''
        This strategy class is more a wrapper of many combinations which trades the child strategy

        We define a bunch of portfolios, this strategy class will execute child strategy and combine all positions

        This class also helps to allocate captial to each MR strategy and execute the strategy
    '''

    def __init__(self, strategy_name="Interday Mean Reversion"):
        super().__init__(strategy_name)

        # Number of Mean Reverting Portfolio to trade
        self.portfolio_num = 10        

        # Minimum Sharpe Ratio to include in actual trade portfolios
        self.sharpe_ratio_res = 1

        # Just init once to snap the child strategy parameters
        self.init_strategy()

    def init_strategy(self):
        '''
            We define all individual mean reversion strategy parameters here
        '''
        strategy = InterdayPairsMeanReversion()

        self.coint_windows = strategy.coint_windows
        return strategy

    def display_params(self):
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")        
        self.logger.info(f'capital:                     {self.capital}')
        self.logger.info(f'is_backtest:                 {self.is_backtest}')        
        self.logger.info(f'open_period_to_use_close:    {self.open_period_to_use_close}')        
        self.logger.info(f'portfolio_num:               {self.portfolio_num}')                
        self.logger.info(f"============================== {self.strategy_name} Setup ==============================")

    def set_portfolio_comb(self, ports):        
        symbols = []
        for _, port in ports:
            symbols.extend(port)

        symbols = list(set(symbols))        
        self.ports = ports              
        self.set_stock_universe(symbols) 

    def preprocess_data(self):        
        '''
            Load all pricing data
        '''
        self.px = self.load_price_data(self.start_date, self.end_date)        
        self.logger.info('{} of possible combination of portfolio over total {} stocks'.format(len(self.ports), len(self.stock_universe)))
        return self

    def generate_position(self): 
        '''
            Iterate all possible combinations and runs mean reversion strategy
            Then combine all position together
        '''       
        self.strategy_dict = {}
        self.position = pd.DataFrame(index=self.px.index, columns=self.px['Close'].columns).fillna(0)[self.coint_windows:]                        

        # position count is the total number of portfolios traded given date and symbol
        self.port_count = pd.DataFrame(index=self.px.index, columns=self.px['Close'].columns).fillna(0)[self.coint_windows:]

        # portfolio count is the total number of portfolios traded given date
        self.port_count_total = pd.Series(index=self.px.index).fillna(0)[self.coint_windows:]

        self.df_stats = []
        self.df_stats_details = []

        with tqdm(total=len(self.ports)) as pbar:
            for key, port in self.ports:            
                cols = [(attr, c) for attr in self.px.columns.get_level_values(0).unique() for c in port]
                px_ = self.px[cols]
                
                strategy = self.init_strategy()
                strategy.set_stock_universe(port)
                strategy.set_start_date(self.start_date)
                strategy.set_end_date(self.end_date)
                strategy.preprocess_data(px_)
                strategy.generate_position()
                strategy.generate_backtest_return()
                pbar.update(1)

                if strategy.df_stats.empty:
                    continue
                
                r = strategy.port_ret
                stats = strategy.df_stats.copy()
                stats = stats.drop(columns=list(port) + [p+"_w" for p in list(port)])                
                stats['key'] = str(key)
                stats['symbols'] = ','.join(port)
                last_stat = dict(stats.iloc[-1][['portfolio_position', 'rank', 'is_mean_revert', 'rolling_not_mean_revert', 'pv_half_life', 'pv_hurst', 'pv_zscore']])

                row = {}
                row['symbols'] = ','.join(port)
                row['key'] = key
                row['cumulative_return'] = cumulative_return(r)[-1]
                row['annualized_return'] = annualized_return(r)
                row['annualized_volatility'] = annualized_volatility(r)
                row['annualized_sharpe_ratio'] = annualized_sharpe_ratio(r)
                row['maximum_drawdown'] = maximum_drawdown(r)
                row['actual_trade'] = row['annualized_sharpe_ratio'] > self.sharpe_ratio_res                                
                row = dict(row, **last_stat)

                pos = strategy.position    

                self.df_stats.append(row)
                self.df_stats_details.append(stats)

                self.strategy_dict[port] = strategy                                
                self.position[list(port)] = self.position[list(port)].add(pos[list(port)], fill_value=0)      
                self.port_count[list(port)] = self.port_count[list(port)].add(pos[list(port)] != 0, fill_value=0)
                self.port_count_total += ((pos != 0).sum(axis=1) > 0)
        
        self.df_stats = pd.DataFrame(self.df_stats)        
        self.df_stats = self.df_stats.sort_values('annualized_sharpe_ratio', ascending=False)        
        self.df_stats_details = pd.concat(self.df_stats_details)

        return self    

    def backtest_summary(self, benchmark=['^SPX', '^IXIC']):        
        super().backtest_summary(benchmark, trade_info=False)    

        position = self.position
        port_count_total = self.port_count_total

        # Total of Trades
        total_trades = position[position != 0].count(axis=1)
        total_long = position[position > 0].count(axis=1)
        total_short = position[position < 0].count(axis=1)        

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        fig = make_subplots(
            rows=2, cols=1,            
            subplot_titles=[                
                'Number of Trades',        
                'Number of Portfolios Traded'
            ],    
            vertical_spacing=0.10,
            shared_xaxes=True,
        )        
        fig.update_layout(width=1300, height=700, hovermode='x')        
        fig.update_layout(
            xaxis_showticklabels=True, 
            xaxis2_showticklabels=True,
        )

        fig.add_trace(go.Scatter(x=total_trades.index, y=total_trades, marker=dict(color=colors[0]), showlegend=True, name='# Trades'), row=1, col=1)
        fig.add_trace(go.Scatter(x=total_long.index, y=total_long, marker=dict(color=colors[1]), showlegend=True, name='# Total Long'), row=1, col=1)
        fig.add_trace(go.Scatter(x=total_short.index, y=total_short, marker=dict(color=colors[2]), showlegend=True, name='# Total Short'), row=1, col=1)        
        fig.add_trace(go.Scatter(x=port_count_total.index, y=port_count_total, marker=dict(color=colors[3]), showlegend=True, name='# Portfolio'), row=2, col=1)
        fig.show()        
    

    
        