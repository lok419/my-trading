from plotly.subplots import make_subplots
from pandas.core.frame import DataFrame
from utils.performance import get_latest_risk_free_rate, maximum_drawdown
from utils.data_helper import title_case
import plotly.graph_objects as go
import plotly
import pandas as pd
import numpy as np
from IPython.display import display

'''
    This is an extension class which consolidates all the pnl or performance related functions (e.g. plots, netting pnl)
'''
class StrategyPerformance(object):

    def compute_pnl(self, orders:DataFrame) -> DataFrame:
        '''
            Compute the pnl of the grid strategy since it was started. 
            it checks all filled orders and follows FIFO logic to pairs up all buy and close orders
        '''           
        orders = orders.copy()
        orders = orders[orders['status'] == 'FILLED']        
        orders = orders.sort_values(['updateTime', 'clientOrderId'])     

        # calculate trading fees separately
        df_fee = orders.copy()        
        df_fee['Date'] = df_fee['updateTime'].dt.round(self.interval_round)        
        df_fee = df_fee.groupby(['Date']).sum(numeric_only=True)[['trading_fee']].reset_index()

        pnl = []
        open_buy = []
        open_sell = []

        for _, row in orders.iterrows():
            date, side, qty, price, grid_id = row['updateTime'], row['side'], row['executedQty'], row['fill_price'], row['grid_id']
            opposite_side, same_side = (open_sell, open_buy) if side == 'BUY' else (open_buy, open_sell)            

            # we have nothing to net
            if len(opposite_side) == 0:
                same_side.append(row.to_dict())
                continue

            # start netting the oppsite side trades using FIFO
            netted_pnl = 0
            while len(opposite_side) > 0:        
                netted_qty = min(opposite_side[0]['executedQty'], qty)
                opposite_side[0]['executedQty'] -= netted_qty
                qty -= netted_qty

                if side == 'BUY':
                    netted_pnl += netted_qty * (opposite_side[0]['fill_price'] - price)            
                else:
                    netted_pnl += netted_qty * (price - opposite_side[0]['fill_price'])                            
                
                # pop the trade if this is completely netted
                if opposite_side[0]['executedQty'] == 0:
                    opposite_side.pop(0)

                # exit the netting when all qtys are netted
                if qty == 0:
                    break

            # for non-netted portion, add to open_buy or open_sell
            if qty > 0:
                row['executedQty'] = qty
                same_side.append(row.to_dict())

            pnl.append((date, netted_pnl, grid_id))             

        close_px = self.df.iloc[-1]['Close']
        close_dt = self.df.iloc[-1]['Date']

        # for all open positions, use close px
        for row in open_buy:
            pnl.append((close_dt, row['executedQty'] * (close_px - row['fill_price']), row['grid_id']))

        for row in open_sell:
            pnl.append((close_dt, row['executedQty'] * (row['fill_price'] - close_px), row['grid_id']))
        
        df_pnl = pd.DataFrame(pnl, columns=['Date', 'pnl_gross', 'grid_id'])       
        # in case of no data, we need to cast column 'Date' to datetime64
        df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
        df_pnl['Date'] = df_pnl['Date'].dt.round(self.interval_round)        
        df_pnl = df_pnl.groupby(['Date']).aggregate({'pnl_gross': 'sum', 'grid_id': 'first'}).reset_index()

        df_pnl = pd.merge(self.df[['Date']], df_pnl, how='left', on='Date', validate='1:1')
        df_pnl = pd.merge(df_pnl, df_fee, how='left', on=['Date'], validate='1:1')
        df_pnl = df_pnl.fillna(0)

        # all pnl columns
        # Net PnL                
        df_pnl['pnl'] = df_pnl['pnl_gross'] - df_pnl['trading_fee']                
        df_pnl['pnl_cum'] = df_pnl['pnl'].cumsum()
        df_pnl['return'] = df_pnl['pnl'] / (self.position_size * self.grid_size * 2)
        df_pnl['return_cum'] = df_pnl['pnl_cum'] / (self.position_size * self.grid_size * 2)

        # Gross PnL
        df_pnl['pnl_gross_cum'] = df_pnl['pnl_gross'].cumsum()
        df_pnl['return_gross'] = df_pnl['pnl_gross'] / (self.position_size * self.grid_size * 2)
        df_pnl['return_gross_cum'] = df_pnl['pnl_gross_cum'] / (self.position_size * self.grid_size * 2)        
        df_pnl['trading_fee_cum'] = df_pnl['trading_fee'].cumsum()        

        # sanity check by re-calculating the pnl in different ways - net cash flow + net positon value
        gross_pnl1 = (-1 * orders['NetExecutedQty'] * orders['fill_price']).sum() + orders['NetExecutedQty'].sum() * close_px
        gross_pnl2 = df_pnl['pnl_gross'].sum()
        diff = abs(gross_pnl1 - gross_pnl2)        

        assert diff < 1e-5, f'Gross PNL is {gross_pnl1}, but computed FIFO netting gross pnl is {gross_pnl2}.'

        return df_pnl
    
    def compute_pnl_metrics(self, df_pnl:DataFrame) -> DataFrame:
        interval = int(self.interval.replace('m', ''))
        rf = get_latest_risk_free_rate()
        rf_ = rf / (24*60/interval)
        ret_ts = df_pnl['return'].values
        ret_mean = np.mean(ret_ts)
        ret_std = np.std(ret_ts)

        sr = (ret_mean - rf_)/ret_std * np.sqrt(360*24*60/interval)        
        ret_cum = np.cumprod(ret_ts+1)[-1]
        ret_mean_ann = ret_mean * 360*24*60/interval
        ret_std_ann = ret_std  * np.sqrt(360*24*60/interval)

        perf = {}            
        perf['cumulative_return'] = ret_cum 
        perf['annualized_return'] = ret_mean_ann
        perf['annualized_volatility'] = ret_std_ann
        perf['annualized_sharpe_ratio'] = sr
        perf['maximum_drawdown'] = maximum_drawdown(ret_ts)

        measures = [title_case(x) for x in list(perf.keys())]
        perf = pd.DataFrame({'Measure': measures, self.__str__(): list(perf.values())})  
        return perf

    def summary(self, 
                plot_orders:bool=False, 
                lastn: int=-1):
        '''
            Summary the performance given the load periods
            plot_orders:    if plot orders on the graphs
            lastn:          only plot last n grid orders
        '''
        df = self.df
        df_orders = self.get_all_orders(query_all=True, trade_details=True)
        df_orders = self.executor.add_trading_fee(self.instrument, df_orders)
        df_orders = df_orders[df_orders['updateTime'] >= df['Date'].min()]
        df_orders = df_orders[df_orders['updateTime'] <= df['Date'].max()]        

        df_pnl = self.compute_pnl(df_orders)        
        df_pnl_metric = self.compute_pnl_metrics(df_pnl)

        # just save the data to object property
        self.df_pnl = df_pnl
        self.df_pnl_metric  = df_pnl_metric

        display(df_pnl_metric)

        fig = make_subplots(
            rows=4, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,             
            subplot_titles=[
                'Price OHLC',     
                'Cumulated PnL / Return (%)',        
                'Half Life',
                'Hurst Exponent',                
            ],         
            row_heights=[0.8, 0.2, 0.2, 0.2],  
        )
        fig.update_layout(
            title=self.instrument,
            width=1500, height=1300,        
            hovermode='x',
        )        

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS
        sma_name = f'SMA {self.vol_lookback}d'
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", legendgroup='OHCL'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df["Date"], y=df['Close_sma'], name=sma_name, legendgroup=sma_name, marker=dict(color=colors[0]), visible='legendonly'), row=1, col=1)
        fig.update(layout_xaxis_rangeslider_visible=False)

        if plot_orders:

            grid_orders = df_orders[df_orders['grid_tt'] == 'grid']

            # Sometime the grid id could be duplicated because the strategy (same id) is re-init such that the grid_id resets to 1
            # We need a way to distinguish they are different grids, so we use 15-mins rounded time to group the grid orders together (grid_id + rounded time).
            # This is because we believe all grids orders must be placed immediately and they must fall within the same 15-mins interval
            grid_orders['grid_start_period'] = grid_orders['time'].dt.round('15min')            

            for _, temp in grid_orders.groupby(['grid_id', 'grid_start_period']):            

                # if nothing got filled, then skip it (this is usually the momentum orders)
                if len(temp[temp['status'] == 'FILLED']) == 0:
                    continue

                grid_min_dt = temp['updateTime'].min()
                grid_max_dt = temp['updateTime'].max()    

                # when we draw the grid line, use the order price instead of fill price
                grid_prices = sorted(temp[temp['type'] == 'LIMIT']['price'].values)

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
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_cum"], name='Cumulative PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl"], name='PnL (Fiat)', visible='legendonly'), row=2, col=1)

        # Gross PnL                
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_gross_cum"]*100, name='Cumulative Gross Return (%)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_gross_cum"], name='Cumulative Gross PnL (Fiat)', visible='legendonly'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["trading_fee_cum"], name='Cumulative Trading Fee (Fiat)', visible='legendonly'), row=2, col=1)


        fig.add_trace(go.Scatter(x=df['Date'], y=df['half_life'], showlegend=False), row=3,col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['hurst_exponent'], showlegend=False), row=4,col=1)        
        fig.show()
