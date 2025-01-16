from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
import warnings
import cvxpy as cp
import os
import random
from utils.db import duck
from utils.logging import get_logger
from utils.data_helper import *
from strategy_v3.Strategy import StrategyModel
from account import Binance
from tabulate import tabulate

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = "{:,.4f}".format
warnings.filterwarnings('ignore')

class ExchangeArbitrageStrategy(StrategyModel):    

    def __init__(self,                 
                zero_fees = False, 
                interval = 30,
                trades_num = -1,                
        ):

        # strategy_id is used to identify the strategy from list of orders
        self.strategy_id = str(round(random.random() * 1e6))
        self.interval = interval        
        self.logger = get_logger(self.__str__())

        # init all symbols fundemental data (only need to load once)        
        self.client = Binance().get_client()
        self.exch_info = self.client.get_exchange_info()
        self.fees = Binance().get_trading_fee()        

        # strategy setup        
        self.zero_fees = zero_fees
        self.trades_num = trades_num
        assert self.trades_num == -1 or self.trades_num >= 2, "trades_num must be >= 2"
        self.trade_currency = {
            'USDT': 50,
            'ETH':  0.01539456264,
            'BTC': 0.00052979531,
        }
        self.logger.info(self.trade_currency)

    def __str__(self):
        return self.strategy_id
    
    def set_strategy_id(self, strategy_id: str, **kargs):
        self.strategy_id = strategy_id
        self.logger.name = self.__str__()

        # path for logging and pnl records                  
        self.db = duck('{}'.format(self.__str__()))            
        
    def set_data_loder(self, *args, **kwargs):
        pass
    
    def set_executor(self, *args, **kwargs):
        pass      

    def load_data(self):
        '''
            Load up all prices data and create quote matrix for optimization
        '''

        price = self.client.get_orderbook_tickers()
        self.price_time = pd.to_datetime(datetime.now(tz=ZoneInfo('HongKong')))


        df_symbols = pd.DataFrame(self.exch_info['symbols'])
        df_symbols = df_symbols[['symbol', 'status', 'baseAsset', 'quoteAsset']]

        df_lot_sizes = pd.DataFrame([dict(symbol=y['symbol'], **next(filter(lambda x: x['filterType'] == 'LOT_SIZE', y['filters']), {})) for y in self.exch_info['symbols']])[['stepSize']]
        df_px_sizes = pd.DataFrame([dict(symbol=y['symbol'], **next(filter(lambda x: x['filterType'] == 'PRICE_FILTER', y['filters']), {})) for y in self.exch_info['symbols']])[['tickSize']]

        df_symbols = pd.concat([df_symbols, df_lot_sizes, df_px_sizes], axis=1)
        df_symbols = df_symbols[df_symbols['status'] == 'TRADING']

        df_symbols['qty_decimal'] = df_symbols['stepSize'].apply(count_digit)
        df_symbols['price_decimal'] = df_symbols['tickSize'].apply(count_digit)

        df_price = pd.DataFrame(price)
        df_symbols = pd.merge(df_symbols, df_price, how='left', on='symbol', validate='1:1')
        df_symbols = pd.merge(df_symbols, self.fees, how='left', on='symbol', validate='1:1')
        df_symbols['makerCommission'] = df_symbols['makerCommission'].fillna(0)
        df_symbols['takerCommission'] = df_symbols['takerCommission'].fillna(0)
        self.df_symbols = df_symbols

        # Create quote matrix for optimization
        # Quote matrix: matrix[x,y] => convert currency x to y at market price
        self.assets = sorted(list(set(self.df_symbols['baseAsset'].to_list() + self.df_symbols['quoteAsset'].to_list())))
        self.quote_matrix = np.ones((len(self.assets), len(self.assets)))
        self.fee_matrix = np.zeros((len(self.assets), len(self.assets)))

        for _, row in self.df_symbols.iterrows():
            baseAsset = row['baseAsset']
            quoteAsset = row['quoteAsset']

            bidPrice = float(row['bidPrice'])
            askPrice = float(row['askPrice'])

            # market orders follows taker fee    
            takerFee = float(row['takerCommission'])

            base_idx = self.assets.index(baseAsset)
            quote_idx = self.assets.index(quoteAsset)

            # sell one unit of base currency to buy quote currency
            self.quote_matrix[base_idx][quote_idx] = bidPrice
                        
            # sell one unti of quote currency to buy base currency
            self.quote_matrix[quote_idx][base_idx] = 1/askPrice
            self.fee_matrix[quote_idx][base_idx] = takerFee
            self.fee_matrix[base_idx][quote_idx] = takerFee

        if self.zero_fees:
            self.quote_matrix_w_fee = self.quote_matrix
        else:
            self.quote_matrix_w_fee = self.quote_matrix * (1 - self.fee_matrix)

        self.quote_matrix_ln = -1 * np.log(self.quote_matrix_w_fee)
        self.quote_matrix_ln[self.quote_matrix_ln == 0] = 100

        np.fill_diagonal(self.quote_matrix_ln, 0)      

    def optimize(self) -> bool: 
        ''' 
            Generate optimal arbitrage trades
        '''        
        '''
            1. OPTIMIZATION PROBLEM via CVXPY (simpified TSP problem)
        '''
        X = cp.Variable((len(self.quote_matrix_ln),len(self.quote_matrix_ln)), boolean=True)
        ones = np.ones((len(self.quote_matrix_ln),1))

        constraints = [
            X <= 1,
            X >= 0,
            X @ ones == ones,
            X.T @ ones == ones,                    
        ]

        if self.trades_num > 0:
            constraints.append(cp.sum(X) - cp.sum(cp.diag(X)) <= self.trades_num)

        # Form objective.
        obj = cp.Minimize(cp.sum(cp.multiply(X, self.quote_matrix_ln)))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(verbose=True)
        self.logger.info(f"CVXPY - Status: {prob.status}")
        self.logger.info(f"CVSPY - Optimal value: {prob.value}")

        # avoid some weird rounding
        self.opt_path = X.value.round(2)

        '''
            2. Generate arbitrage trades based on optimization output
               a. identify all arbitrage trades based on opt_path matrix
               b. seggregate all arbitrage trades per closed loops (optimization can returns more than one closed loops)
        '''

        total_pnl = self.opt_path * self.quote_matrix * (1 - self.fee_matrix)
        total_pnl = np.prod(total_pnl[total_pnl != 0]) - 1
        self.logger.info(f"Total PNL: {100*(total_pnl):.4f}%")        

        # 2a. First identify all arbitrage trades
        trades = {}
        for i, row in enumerate(self.opt_path):
            from_idx, to_idx = i, list(row).index(1)    
            from_asset, to_asset = self.assets[from_idx], self.assets[to_idx]         
            if from_asset != to_asset:        
                trades[from_asset] = {
                    'from_asset': from_asset,
                    'to_asset': to_asset,
                    'mkt_price': self.quote_matrix[from_idx][to_idx],
                    'fee': self.fee_matrix[from_idx][to_idx]
                }

        # in case no optimal trades, return False
        if len(trades) == 0:
            self.logger.info("No optimal trades found, end here.") 
            return False                

        # 2b. seggregate the arbitrage trades into groups (optimization can returns more than one closed loops)
        visited = set()
        group_num = 1

        for from_asset in trades:        
            if from_asset in visited:
                continue

            node = from_asset
            trade_order = 1

            while trades[node]['to_asset'] != from_asset:
                trades[node]['group'] = group_num
                trades[node]['order'] = trade_order

                # move to next node
                node = trades[node]['to_asset']

                # update status
                visited.add(node)
                trade_order += 1

            else:
                trades[node]['group'] = group_num
                trades[node]['order'] = trade_order

            group_num += 1
            
        df_trades = pd.DataFrame(trades.values())
        df_trades = df_trades.sort_values(by=['group', 'order'])
        df_trades['mkt_price_w_fee'] = df_trades['mkt_price'] * (1-df_trades['fee'])
        df_trades['symbol'] = df_trades.apply(lambda x: x['from_asset'] + x['to_asset'] if x['from_asset'] + x['to_asset'] in self.df_symbols['symbol'].unique() else x['to_asset'] + x['from_asset'], axis=1)
        df_trades = pd.merge(df_trades, self.df_symbols, how='left', on='symbol', validate='1:1')
        df_trades['side'] = np.where(df_trades['to_asset'] == df_trades['baseAsset'], 'BUY', 'SELL')
        df_trades['count'] = 1        
        df_trades['price_time'] = self.price_time
        df_trades['zero_fees'] = self.zero_fees        
        df_trades = Binance.format_output(df_trades)
        self.df_trades = df_trades        

        # save the trades for reference        
        self.db.insert('trades', df_trades, append_new_column=True)

        df_pnl = df_trades.groupby(['group']).agg({'mkt_price': 'prod', 'mkt_price_w_fee': 'prod', 'count': 'sum'})
        df_pnl.columns = ['gross_pnl%', 'net_pnl%', 'count']
        df_pnl['gross_pnl%'] = (df_pnl['gross_pnl%'] - 1)*100
        df_pnl['net_pnl%'] = (df_pnl['net_pnl%'] - 1)*100
        df_pnl = df_pnl.sort_values('net_pnl%', ascending=False)
        self.df_pnl = df_pnl
        self.logger.info(f"\n{tabulate(df_pnl, headers='keys', tablefmt='psql')}")        

        if 100 * total_pnl < 0.001:
            self.logger.info("Net pnl is too small, end here.")
            return False        

        return True
    
    def execute(self, force=False, *args, **kwargs):        
        if self.zero_fees and not force:
            self.logger.info('Do not execute the trades given zero_fees assumptions.')
            return
        
        # Trade the First Group For Now
        df_trades = self.df_trades
        df_pnl = self.df_pnl
        trades = df_trades[df_trades['group'] == df_pnl.index[0]]

        # reorder the trades to start with one of the trade currency
        start_ccys = [x for x in self.trade_currency if x in trades['from_asset'].to_list()]
        if len(start_ccys) == 0:
            self.logger.info('No trade currency found')
            return

        start_ccy = start_ccys[0]
        start_order = trades[trades['from_asset'] == start_ccy].iloc[0]['order']
        trades['order'] = (trades['order'] - start_order) % len(trades) + 1
        trades = trades.sort_values('order').reset_index(drop=True)

        # sanity check
        assert trades.iloc[0]['from_asset'] == start_ccy
        assert trades.iloc[-1]['to_asset'] == start_ccy

        trade_orders = []

        # execute the trades via market orders
        for i, row in trades.iterrows():
            symbol = row['symbol']
            side = row['side']
            from_asset = row['from_asset']
            to_asset = row['to_asset']    

            baseAsset = row['baseAsset']
            baseAsset_decimal = row['qty_decimal']
            quoteAsset_decimal = row['price_decimal']            

            # Rules:
            # 1. First trade are based on pre-set amounts under self.trade_currency
            # 2. qty is based on from_asset
            if i == 0:            
                from_asset_qty = self.trade_currency[from_asset]        
            
            order_params = {
                'symbol': symbol,
                'side': side,
                'type':'MARKET'
            }

            # round_down to make sure we don't sell more than holding balances
            if from_asset == baseAsset:
                order_params['quantity'] = np.format_float_positional(round_down(from_asset_qty, baseAsset_decimal), trim='-')
            else:
                order_params['quoteOrderQty'] = np.format_float_positional(round_down(from_asset_qty, quoteAsset_decimal), trim='-')
            
            order = self.client.create_order(**order_params)                    
            self.logger.info(f'Created Market Order: {order_params}')
            
            if order['status'] == 'FILLED':   
                # find net received asset given current fills, therefore, from_asset_qty will be used for trade in next iteration
                from_asset_qty = 0
                for fill in order['fills']:
                    if to_asset == baseAsset:
                        from_asset_qty += float(fill['qty'])
                    else:
                        from_asset_qty += float(fill['qty']) * float(fill['price'])

                    # subtract out comission to get net received quantity, comission is always quoted in received asset
                    if 'commission' in fill and to_asset == fill['commissionAsset']:
                        from_asset_qty -= float(fill['commission'])

                trade_orders.append(order)                
            else:
                raise Exception('order not filled yet!')   

        df_orders = pd.DataFrame(trade_orders).drop(columns=['fills'])
        df_orders = df_orders.rename(columns={'side': 'order_side'})

        df_fills = pd.DataFrame([dict({'symbol': x['symbol']},**y) for x in trade_orders for y in x['fills']])
        df_fills = df_fills.rename(columns={'price': 'fill_price', 'qty': 'fill_qty'})
        df_fills['fill_price'] = df_fills['fill_price'].astype(float)
        df_fills['fill_qty'] = df_fills['fill_qty'].astype(float)
        df_fills['commission'] = df_fills['commission'].astype(float)

        df_orders = pd.merge(df_orders, df_fills, how='left', on=['symbol'], validate='1:m')
        df_orders = df_orders.rename(columns={'status': 'fill_status'})
        df_orders = pd.merge(trades, df_orders, how='left', on=['symbol'], validate='1:m')

        df_orders['from_asset_qty'] = df_orders.apply(lambda x: x['fill_qty'] if x['from_asset'] == x['baseAsset'] else x['fill_qty'] * x['fill_price'], axis=1)
        df_orders['to_asset_gross_qty'] = df_orders.apply(lambda x: x['fill_qty'] if x['to_asset'] == x['baseAsset'] else x['fill_qty'] * x['fill_price'], axis=1)
        df_orders['to_asset_comms_qty'] = df_orders.apply(lambda x: x['commission'] if x['to_asset'] == x['commissionAsset'] else 0, axis=1)
        df_orders['to_asset_qty'] = df_orders['to_asset_gross_qty'] - df_orders['to_asset_comms_qty']
        df_orders = Binance.format_output(df_orders)
        self.df_orders = df_orders

        # save the trades for reference        
        self.db.insert('orders', df_orders, append_new_column=True)        

        df_orders_agg = df_orders.groupby(['group', 'order', 'from_asset']).sum(numeric_only=True).reset_index()
        df_orders_agg = df_orders_agg.groupby(['group']).agg({'from_asset': 'first', 'from_asset_qty': 'first', 'to_asset_qty': 'last'})
        df_orders_agg['net_qty'] = df_orders_agg['to_asset_qty'] - df_orders_agg['from_asset_qty']
        self.df_orders_agg = df_orders_agg        
        self.logger.info(f"\n{tabulate(df_orders_agg, headers='keys', tablefmt='psql')}")                
    
    def run(self, *args, **kwargs):
        pass
    
    def run_once(self, *args, **kwargs):
        self.load_data()                        
        if self.optimize():
            self.execute()
    
    def cancel_all_orders(self):
        pass
    
    def is_delta_neutral(self):
        pass
    
    def close_out_positions(self):
        pass
