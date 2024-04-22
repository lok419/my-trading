from plotly.subplots import make_subplots
from pandas.core.frame import DataFrame
from IPython.display import display
from strategy_v3.Strategy.Performance import StrategyPerformance
from datetime import timedelta
import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np

'''
    This is an extension class which consolidates all the pnl or performance related functions (e.g. plots, netting pnl)
'''
class MarketMakingPerformance(StrategyPerformance):

    def compute_pnl(self, orders: DataFrame) -> DataFrame:
        capital = self.position_size * 2
        df_pnl = super().compute_pnl(orders, capital)
        return df_pnl

    def compute_position(self, orders: DataFrame) -> DataFrame:
        orders = orders
        orders = orders.sort_values('updateTime')
        orders['position'] = orders['NetExecutedQty'].cumsum()
        orders['Date'] = orders['updateTime'].dt.round(self.interval_round)        

        df_pos = orders[['Date', 'position']].groupby(['Date']).last().reset_index()
        df_pos = pd.merge(df_pos, self.df[['Date', 'Close']], how='left', on='Date', validate='1:1')        
        df_pos['position_fiat'] = df_pos['position'] * df_pos['Close']
        df_pos['position_per_size'] = df_pos['position_fiat'] / self.position_size

        return df_pos

    def summary(self, 
                plot_orders:bool=False,                 
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

        df_orders = self.get_all_orders(trade_details=True, start_date=start_date, end_date=end_date)
        df_orders = self.executor.add_trading_fee(self.instrument, df_orders)
        df_orders = df_orders[df_orders['updateTime'] >= start_date]
        df_orders = df_orders[df_orders['updateTime'] <= end_date]        

        df_pnl = self.compute_pnl(df_orders)                
        df_pnl_metrics = self.compute_pnl_metrics(df_pnl, df_orders)
        df_pos = self.compute_position(df_orders)
            
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
                'Position',
                f'Average True Range / Close Std ({self.vol_lookback} x {self.interval})',
                f'Hurst Exponent',                                
            ],         
            row_heights=[0.8, 0.2, 0.2, 0.2, 0.2],  
        )
        fig.update_layout(
            title=self.instrument,
            width=1500, height=1500,        
            hovermode='x',
        )        

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        sma_name = f'SMA {self.vol_lookback}d'
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", legendgroup='OHCL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df['close_sma'], name=sma_name, legendgroup=sma_name, marker=dict(color=colors[0])), row=1, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)
                                
        filled_grid_orders = df_orders[(df_orders['status'] == 'FILLED')]
        filled_grid_buy = filled_grid_orders[filled_grid_orders['side'] == 'BUY']
        filled_grid_sell = filled_grid_orders[filled_grid_orders['side'] == 'SELL']

        fig.add_trace(go.Scatter(x=filled_grid_buy['updateTime'], y=filled_grid_buy['fill_price'], marker=dict(color='green',size=15), mode='markers', marker_symbol=5, name='Buy'),row=1, col=1)
        fig.add_trace(go.Scatter(x=filled_grid_sell['updateTime'], y=filled_grid_sell['fill_price'], marker=dict(color='red',size=15), mode='markers', marker_symbol=6, name='Sell'),row=1, col=1)        

        # Net PnL
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_cum"]*100, name='Cumulative Return (%)'), row=2, col=1)        
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_cum"], name='Cumulative PnL (Fiat)', visible='legendonly'), row=2, col=1)        
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl"], name='PnL (Fiat)', visible='legendonly'), row=2, col=1)        

        # Gross PnL                
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_gross_cum"]*100, name='Cumulative Gross Return (%)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_gross_cum"], name='Cumulative Gross PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["trading_fee_cum"], name='Cumulative Trading Fee (Fiat)', visible='legendonly'), row=2, col=1)        

        # Position
        fig.add_trace(go.Scatter(x=df_pos['Date'], y=df_pos['position_fiat'], name='Position (Fiat)'),row=3, col=1)
        fig.add_trace(go.Scatter(x=df_pos['Date'], y=df_pos['position'], name='Position', visible='legendonly'),row=3, col=1)
        fig.add_trace(go.Scatter(x=df_pos['Date'], y=df_pos['position_per_size'], name='Position per Size', visible='legendonly'),row=3, col=1)

        # Vol
        fig.add_trace(go.Scatter(x=df['Date'], y=df['atr'], name='Average True Range'), row=4,col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['close_std'], name='Close Std'), row=4,col=1)        
        fig.add_trace(go.Scatter(x=df['Date'], y=df['hurst_exponent'], showlegend=False), row=5,col=1)        

        if len(save_jpg_path) == 0:
            fig.show()
        else:                        
            fig.write_image(save_jpg_path, format='png')
   

