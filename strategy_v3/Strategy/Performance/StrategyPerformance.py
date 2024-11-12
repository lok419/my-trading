from pandas.core.frame import DataFrame
from plotly.subplots import make_subplots
from IPython.display import display
from utils.performance import get_latest_risk_free_rate, maximum_drawdown
from utils.data_helper import title_case
from datetime import timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly

'''
    This is an extension class which consolidates all the pnl or performance related functions (e.g. plots, netting pnl)
'''
class StrategyPerformance(object):

    def compute_pnl(self, orders:DataFrame, capital: float) -> DataFrame:
        '''
            Compute the pnl of the grid strategy since it was started. 
            it checks all filled orders and follows FIFO logic to pairs up all buy and close orders

            orders:     DataFrame of all orders
            capital:    capital allocated to the strategy, used to compute the pnl%
        '''           
        orders = orders.copy()

        # orders could be partially filled and cancelled later, so the executedQty could be zero for a cancelled orders.
        orders = orders[orders['NetExecutedQty'] != 0]                       

        # Some edge cases that there is no order filled price
        null_orders = orders[orders['fill_price'].isnull()]
        if len(null_orders):
            self.logger.error('Below orders are having null fill price, using order limit price instead')                        
            display(null_orders)

        orders['fill_price'] = orders['fill_price'].fillna(orders['price'])        
        orders = orders.sort_values(['updateTime', 'clientOrderId'])     

        # calculate trading fees separately
        df_fee = orders.copy()        
        df_fee['Date'] = df_fee['updateTime'].dt.round(self.interval_round)                
        df_fee = df_fee.groupby(['Date']).sum(numeric_only=True)[['trading_fee']].reset_index()

        pnl = []
        open_buy = []
        open_sell = []

        for _, row in orders.iterrows():
            date, side, qty, price = row['updateTime'], row['side'], row['executedQty'], row['fill_price']
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

            pnl.append((date, netted_pnl))             

        close_px = self.df.iloc[-1]['Close']
        close_dt = self.df.iloc[-1]['Date']

        # for all open positions, use close px
        for row in open_buy:
            pnl.append((close_dt, row['executedQty'] * (close_px - row['fill_price'])))

        for row in open_sell:
            pnl.append((close_dt, row['executedQty'] * (row['fill_price'] - close_px)))
        
        df_pnl = pd.DataFrame(pnl, columns=['Date', 'pnl_gross'])

        # in case of no data, we need to cast column 'Date' to datetime64
        df_pnl['Date'] = pd.to_datetime(df_pnl['Date'])
        df_pnl['Date'] = df_pnl['Date'].dt.round(self.interval_round)        
        df_pnl = df_pnl.groupby(['Date']).aggregate({'pnl_gross': 'sum'}).reset_index()

        df_pnl = pd.merge(self.df[['Date']], df_pnl, how='outer', on='Date', validate='1:1')
        df_pnl = pd.merge(df_pnl, df_fee, how='left', on=['Date'], validate='1:1')
        df_pnl = df_pnl.fillna(0)        

        # all pnl columns
        # Net PnL                
        df_pnl['pnl'] = df_pnl['pnl_gross'] - df_pnl['trading_fee']                
        df_pnl['pnl_cum'] = df_pnl['pnl'].cumsum()
        df_pnl['return'] = df_pnl['pnl'] / capital
        df_pnl['return_cum'] = df_pnl['pnl_cum'] / capital

        # Gross PnL
        df_pnl['pnl_gross_cum'] = df_pnl['pnl_gross'].cumsum()
        df_pnl['return_gross'] = df_pnl['pnl_gross'] / capital
        df_pnl['return_gross_cum'] = df_pnl['pnl_gross_cum'] / capital
        df_pnl['trading_fee_cum'] = df_pnl['trading_fee'].cumsum()        

        # sanity check by re-calculating the pnl in different ways - net cash flow + net positon value
        gross_pnl1 = (-1 * orders['NetExecutedQty'] * orders['fill_price']).sum() + orders['NetExecutedQty'].sum() * close_px
        gross_pnl2 = df_pnl['pnl_gross'].sum()
        diff = abs(gross_pnl1 - gross_pnl2)        

        assert diff < 1e-5, f'Gross PNL is {gross_pnl1}, but computed FIFO netting gross pnl is {gross_pnl2}.'

        return df_pnl
    
    def compute_pnl_metrics(self, 
                            df_pnl:DataFrame,
                            df_orders:DataFrame,
                            rename:bool=True,
                            ) -> DataFrame:
        '''
            Compute pnl metrics after the strategy given series of pnl and orders
        '''
        
        interval = self.interval_min

        rf = get_latest_risk_free_rate()
        rf_ = rf / (24*60/interval)
        ret_ts = df_pnl['return'].values
        ret_mean = np.mean(ret_ts)
        ret_std = np.std(ret_ts)

        pnl = df_pnl['pnl'].sum()
        pnl_gross = df_pnl['pnl_gross'].sum()
        trading_fee = df_pnl['trading_fee'].sum()        

        sr = (ret_mean - rf_)/ret_std * np.sqrt(360*24*60/interval)        
        ret_cum = 1 + df_pnl['return_cum'].iloc[-1]
        ret_mean_ann = ret_mean * 360*24*60/interval
        ret_std_ann = ret_std  * np.sqrt(360*24*60/interval)

        perf = {}   
        perf['pnl'] = pnl
        perf['trading_fee'] = -trading_fee   
        perf['pnl_gross'] = pnl_gross
        perf['cumulative_return'] = ret_cum 
        perf['annualized_return'] = ret_mean_ann
        perf['annualized_volatility'] = ret_std_ann
        perf['annualized_sharpe_ratio'] = sr
        perf['maximum_drawdown'] = maximum_drawdown(ret_ts)
        
        measures = [title_case(x) if rename else x for x in list(perf.keys())]        
        perf = pd.DataFrame({'Measure': measures, self.__str__(): list(perf.values())})  
        return perf
    
    def summary_table(self, rename:bool=True) -> DataFrame:
        '''
            Generate Sumamry Table of the performance given loaded data 
            i.e. time horizonal depends on date for load_data()            
        '''
        df = self.df
        start_date = df['Date'].min()
        end_date = df['Date'].max() + timedelta(minutes=self.interval_min)

        df_orders = self.get_all_orders(trade_details=True, start_date=start_date, end_date=end_date)
        df_orders = self.executor.add_trading_fee(self.instrument, df_orders)
        df_orders = df_orders[df_orders['updateTime'] >= df['Date'].min()]
        df_orders = df_orders[df_orders['updateTime'] <= df['Date'].max() + timedelta(minutes=self.interval_min)]        
        df_pnl = self.compute_pnl(df_orders)        
        df_pnl_metric = self.compute_pnl_metrics(df_pnl, df_orders, rename=rename)
        return df_pnl_metric     
    
    def summary(self, 
                plot_orders:bool=False,                 
                show_pnl_metrics: bool = True,
                save_jpg_path: str = '',                
                ):
        '''
            Genreal Summary the performance given the load periods (not design for strategy specific)
            plot_orders:        if plot orders on the graphs            
            show_pnl_metrics    True to show the performance metrics table
            save_jpg_path       file path to save the figure. default to not save
        '''
        df = self.df
        start_date = df['Date'].min()
        end_date = df['Date'].max() + timedelta(minutes=self.interval_min)

        df_orders = self.get_all_orders(trade_details=True, start_date=start_date, end_date=end_date)
        df_orders = self.executor.add_trading_fee(self.instrument, df_orders)
        if len(df_orders):
            df_orders = df_orders[df_orders['updateTime'] >= start_date]
            df_orders = df_orders[df_orders['updateTime'] <= end_date]        

        df_pnl = self.compute_pnl(df_orders, self.position_size)                
        df_pnl_metrics = self.compute_pnl_metrics(df_pnl, df_orders)
            
        # just save the data to object property
        self.df_orders = df_orders
        self.df_pnl = df_pnl
        self.df_pnl_metrics = df_pnl_metrics

        if show_pnl_metrics:
            display(df_pnl_metrics)

        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,             
            subplot_titles=[
                'Price OHLC',
                'Volume',
                'Cumulated PnL / Return (%)',                                 
            ],         
            row_heights=[0.8, 0.3, 0.3],  
        )
        fig.update_layout(
            title=self.instrument,
            width=1500, height=1000,        
            hovermode='x',
        )        

        colors = plotly.colors.DEFAULT_PLOTLY_COLORS        
        fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC", legendgroup='OHCL'), row=1, col=1)        

        # Volume
        df_up = df[df['Close'] > df['Open']]
        df_down = df[df['Close'] <= df['Open']]
        fig.add_trace(go.Bar(x=df_up['Date'], y=df_up['Volume'], showlegend=False, marker=dict(color=colors[2])), row=2, col=1)
        fig.add_trace(go.Bar(x=df_down['Date'], y=df_down['Volume'], showlegend=False, marker=dict(color=colors[3])), row=2, col=1)    
        fig.update(layout_xaxis_rangeslider_visible=False)

        if plot_orders:                            
            # limit orders
            filled_grid_orders = df_orders[(df_orders['status'] == 'FILLED')&(df_orders['trd_tt'] != 'cls')]
            filled_grid_buy = filled_grid_orders[filled_grid_orders['side'] == 'BUY']
            filled_grid_sell = filled_grid_orders[filled_grid_orders['side'] == 'SELL']

            # close orders 
            filled_close_orders = df_orders[(df_orders['status'] == 'FILLED')&(df_orders['trd_tt'] == 'cls')]
            filled_close_buy = filled_close_orders[filled_close_orders['side'] == 'BUY']
            filled_close_sell = filled_close_orders[filled_close_orders['side'] == 'SELL']

            fig.add_trace(go.Scatter(x=filled_grid_buy['updateTime'], y=filled_grid_buy['fill_price'], marker=dict(color='green',size=15), mode='markers', marker_symbol=5, name='Buy'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_grid_sell['updateTime'], y=filled_grid_sell['fill_price'], marker=dict(color='red',size=15), mode='markers', marker_symbol=6, name='Sell'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_close_buy['updateTime'], y=filled_close_buy['fill_price'], marker=dict(color='green',size=15), mode='markers', marker_symbol='x', name='Buy to Close'),row=1, col=1)
            fig.add_trace(go.Scatter(x=filled_close_sell['updateTime'], y=filled_close_sell['fill_price'], marker=dict(color='red',size=15), mode='markers', marker_symbol='x', name='Sell to Close'),row=1, col=1)

        # Net PnL
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_cum"]*100, name='Cumulative Return (%)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_cum"], name='Cumulative PnL (Fiat)', visible='legendonly'), row=3, col=1)    
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl"], name='PnL (Fiat)', visible='legendonly'), row=2, col=1)
        
        # Gross PnL                
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["return_gross_cum"]*100, name='Cumulative Gross Return (%)', visible='legendonly'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["pnl_gross_cum"], name='Cumulative Gross PnL (Fiat)', visible='legendonly'), row=3, col=1)
        fig.add_trace(go.Scatter(x=df_pnl["Date"], y=df_pnl["trading_fee_cum"], name='Cumulative Trading Fee (Fiat)', visible='legendonly'), row=3, col=1)
        
        if len(save_jpg_path) == 0:
            fig.show()
        else:                        
            fig.write_image(save_jpg_path, format='png')