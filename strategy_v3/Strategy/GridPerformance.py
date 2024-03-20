from plotly.subplots import make_subplots
from pandas.core.frame import DataFrame
from utils.performance import get_latest_risk_free_rate, maximum_drawdown
from utils.data_helper import title_case
from IPython.display import display
from strategy_v3.Strategy import GRID_TYPE, StrategyPerformance
from datetime import timedelta
import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np

'''
    This is an extension class which consolidates all the pnl or performance related functions (e.g. plots, netting pnl)
'''
class GridPerformance(StrategyPerformance):

    def compute_pnl(self, orders: DataFrame) -> DataFrame:
        capital = self.position_size * self.grid_size  * 2
        df_pnl = super().compute_pnl(orders, capital)
        return df_pnl

    def summary(self, 
                plot_orders:bool=False, 
                lastn: int=20,
                show_pnl_metrics: bool = True,
                save_jpg_path: str = '',                
                ):
        '''
            Summary the performance given the load periods
            plot_orders:        if plot orders on the graphs
            lastn:              only plot last n grid orders lines
            show_pnl_metrics    True to show the performance metrics table
            save_jpg_path       file path to save the figure. default to not save
        '''
        df = self.df
        start_date = df['Date'].min()
        end_date = df['Date'].max() + timedelta(minutes=self.interval_min)

        df_orders = self.get_all_orders(query_all=True, trade_details=True, start_date=start_date, end_date=end_date)
        df_orders = self.executor.add_trading_fee(self.instrument, df_orders)
        df_orders = df_orders[df_orders['updateTime'] >= start_date]
        df_orders = df_orders[df_orders['updateTime'] <= end_date]        

        df_pnl = self.compute_pnl(df_orders)        
        df_pnl_mr = self.compute_pnl(df_orders[df_orders['grid_type'] == GRID_TYPE.MEAN_REVERT.name])        
        df_pnl_mu = self.compute_pnl(df_orders[df_orders['grid_type'] == GRID_TYPE.MOMENTUM_UP.name])        
        df_pnl_md = self.compute_pnl(df_orders[df_orders['grid_type'] == GRID_TYPE.MOMENTUM_DOWN.name])                
        df_pnl_metrics = self.compute_pnl_metrics(df_pnl, df_orders)
            
        # just save the data to object property
        self.df_orders = df_orders
        self.df_pnl = df_pnl
        self.df_pnl_metrics = df_pnl_metrics

        if show_pnl_metrics:
            display(df_pnl_metrics)

        fig = make_subplots(
            rows=5, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,             
            subplot_titles=[
                'Price OHLC',     
                'Cumulated PnL / Return (%)', 
                f'Average True Range / Close Std ({self.vol_lookback} x {self.interval})',
                'Half Life',
                'Hurst Exponent',                
            ],         
            row_heights=[0.8, 0.2, 0.2, 0.2, 0.2],  
        )
        fig.update_layout(
            title=self.instrument,
            width=1500, height=1600,        
            hovermode='x',
        )        

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        sma_name = f'SMA {self.vol_lookback}d'
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", legendgroup='OHCL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df['close_sma'], name=sma_name, legendgroup=sma_name, marker=dict(color=colors[0])), row=1, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)

        if plot_orders:

            grid_orders = df_orders[df_orders['grid_tt'] == 'grid']

            # Sometime the grid id could be duplicated because the strategy (same id) is re-init such that the grid_id resets to 1
            # We need a way to distinguish they are different grids, so we use 15-mins rounded time to group the grid orders together (grid_id + rounded time).
            # This is because we believe all grids orders must be placed immediately and they must fall within the same 15-mins interval
            grid_orders['grid_start_period'] = grid_orders['time'].dt.round('15min')    

            if lastn > 0:
                top_grid_orders = grid_orders.drop_duplicates(['grid_id', 'grid_start_period'])[['grid_id', 'grid_start_period']]
                top_grid_orders = top_grid_orders.sort_values(['grid_start_period', 'grid_id']).tail(lastn)
                grid_orders = pd.merge(grid_orders, top_grid_orders, how='inner', on=['grid_start_period', 'grid_id'], validate='m:1')

            for _, temp in grid_orders.groupby(['grid_id', 'grid_start_period']):            

                # if nothing got filled, then skip it (this is usually the momentum orders)
                if len(temp[temp['status'] == 'FILLED']) == 0:
                    continue

                grid_min_dt = temp['updateTime'].min()
                grid_max_dt = temp['updateTime'].max()    

                # when we draw the grid line, use the order price instead of fill price
                grid_prices = sorted(temp[temp['type'].str.contains('LIMIT')]['price'].values)

                for i, grid in enumerate(grid_prices):                    
                    if i >= self.grid_size:
                        color = 'red'        
                    else:
                        color = 'green'  

                    fig.add_shape(type='line', x0=grid_min_dt, x1=grid_max_dt, y0=grid, y1=grid, line=dict(color=color, dash='dash'), row=1, col=1)
                    
            # grid orders 
            filled_grid_orders = df_orders[(df_orders['status'] == 'FILLED')&(df_orders['grid_tt'] == 'grid')]
            filled_grid_buy = filled_grid_orders[filled_grid_orders['side'] == 'BUY']
            filled_grid_sell = filled_grid_orders[filled_grid_orders['side'] == 'SELL']

            # close orders 
            filled_close_orders = df_orders[(df_orders['status'] == 'FILLED')&(df_orders['grid_tt'] != 'grid')]
            filled_close_buy = filled_close_orders[filled_close_orders['side'] == 'BUY']
            filled_close_sell = filled_close_orders[filled_close_orders['side'] == 'SELL']

            fig.add_trace(go.Scatter(x=filled_grid_buy['updateTime'], y=filled_grid_buy['fill_price'], marker=dict(color='green',size=15), mode='markers', marker_symbol=5, name='Buy'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_grid_sell['updateTime'], y=filled_grid_sell['fill_price'], marker=dict(color='red',size=15), mode='markers', marker_symbol=6, name='Sell'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_close_buy['updateTime'], y=filled_close_buy['fill_price'], marker=dict(color='green',size=15), mode='markers', marker_symbol='x', name='Buy to Close'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_close_sell['updateTime'], y=filled_close_sell['fill_price'], marker=dict(color='red',size=15), mode='markers', marker_symbol='x', name='Sell to Close'),row=1, col=1)

        # Net PnL
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_cum"]*100, name='Cumulative Return (%)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mr["return_cum"]*100, name='Cumulative Return (%) - Mean Revert', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mu["return_cum"]*100, name='Cumulative Return (%) - Momentum Up', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_md["return_cum"]*100, name='Cumulative Return (%) - Momentum Down', visible='legendonly'), row=2, col=1)

        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_cum"], name='Cumulative PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mr["pnl_cum"], name='Cumulative PnL (Fiat) - Mean Revert', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mu["pnl_cum"], name='Cumulative PnL (Fiat) - Momentum Up', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_md["pnl_cum"], name='Cumulative PnL (Fiat) - Momentum Down', visible='legendonly'), row=2, col=1)

        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl"], name='PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mr["pnl"], name='PnL (Fiat) - Mean Revert', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_mu["pnl"], name='PnL (Fiat) - Momentum Up', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl_md["pnl"], name='PnL (Fiat) - Momentum Down', visible='legendonly'), row=2, col=1)

        # Gross PnL                
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_gross_cum"]*100, name='Cumulative Gross Return (%)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_gross_cum"], name='Cumulative Gross PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["trading_fee_cum"], name='Cumulative Trading Fee (Fiat)', visible='legendonly'), row=2, col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['atr'], name='Average True Range'), row=3,col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['close_std'], name='Close Std'), row=3,col=1)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['half_life'], showlegend=False), row=4,col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['hurst_exponent'], showlegend=False), row=5,col=1)        

        if len(save_jpg_path) == 0:
            fig.show()
        else:                        
            fig.write_image(save_jpg_path, format='png')

