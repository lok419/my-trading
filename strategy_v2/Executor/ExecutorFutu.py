from strategy_v2.Executor import ExecutorModel
from strategy_v2.Portfolio import PortfolioModel
from account import Futu
from futu import *
from utils.logging import get_logger
from utils.credentials import FUTU_TRADE_UNLOCK_PIN
from IPython.display import display
from pandas.core.frame import DataFrame
from utils.data_helper import add_bday, get_today

import yfinance as yf
import pandas as pd
import time

class ExecutorFutu(ExecutorModel):
    '''
        The is the Futu Executor for US Stocks. The class has two main features:        
        1.  Execute the trade according the target position and current position
        2.  Get the current position data / capital from Futu

        The class should focus on more business logic on executing the portfolio rather than general functionailty of Futu
    '''
    def __init__(
            self,   
            is_test:bool=True,         
        ):              
        # Init all accounts settings
        self.is_test = is_test        
        self.logger = get_logger('ExecutorFutu' if not is_test else 'ExecutorFutu (Test)' )        
        self.market = TrdMarket.US
        self.logger.info(f'market: {self.market}')
        self.portfolio = None
        self.account = Futu()
    
    def set_portfolio(self, 
                      portfolio:PortfolioModel, 
                      ):
        # set a portfolios        
        self.portfolio = portfolio                
        #self.logger.name = f'{self.logger.name} ({str(self.portfolio)})'        

    def calc_order_price(self,
                    instruments:list,
                    interval:str='15m') -> pd.Series:
        '''
            Get the nearest 15mins price from Yahoo to be used in placing limit orders
        '''                        
        order_px = yf.download(tickers=instruments, interval=interval ,auto_adjust=True, start=add_bday(datetime.now(), -1), end=add_bday(datetime.now(),1))
        order_px = order_px
        order_px = order_px.ffill()
        order_px = order_px.iloc[-1]

        self.logger.info('getting last {} prices since {} for order limit price'.format(interval, order_px.name.strftime('%Y-%m-%d %H:%M:%S')))
        return order_px            

    def execute(self, 
                position_date:datetime=None,
                px_interval:str='15m') -> DataFrame:
        '''
            Execute the portfolio based on specific position date
            Args:
                position_date:  position date to execute in portfolio. By default, it execute the latest position date
        '''       
        self.logger.info('Cancel all orders first before executing.....')
        self.account.cancel_all_orders()
        time.sleep(3)

        # get target position from portfolio
        port_position_shs = self.portfolio.get_position_for_trade()
        port_position_shs = port_position_shs.tail(1) if position_date is None else port_position_shs.loc[position_date]
        position_target = pd.DataFrame(port_position_shs.iloc[0].values, index=port_position_shs.columns, columns=['target'])
        position_target.index.name = 'instrument'   

        position_date = port_position_shs.index[0]
        self.logger.info('Execute {} position based on {}'.format(str(self.portfolio), position_date.strftime('%Y-%m-%d')))     
        
        # get current position from Futu
        position_current = self.get_position()
        symbols = [self.account.symbol_converter(x) for x in position_current['code'].values]
        shares = position_current['qty'].values        
        position_current = pd.DataFrame(shares, index=symbols, columns=['current'])
        position_current.index.name = 'instrument'
        
        # compute position turnover
        position_turnover = pd.merge(position_target, position_current, on='instrument', how='left', validate='1:1')
        position_turnover = position_turnover.fillna(0)
        position_turnover['turnover'] = position_turnover['target'] - position_turnover['current']
        position_turnover = position_turnover.reset_index()
        display(position_turnover)

        order_px = self.calc_order_price(position_turnover['instrument'].to_list(), interval=px_interval)

        # start execute the position via Futu API        
        outputs = []

        for _, row in position_turnover.iterrows():
            if abs(row['turnover']) < 1:
                continue

            symbol = row['instrument']
            order_symbol = self.account.symbol_converter(symbol, 'futu')
            shares = abs(row['turnover'])

            if self.is_test:
                side = TrdSide.BUY
                trd_env = TrdEnv.SIMULATE
            else:
                side = TrdSide.BUY if row['turnover'] > 0 else TrdSide.SELL
                trd_env = TrdEnv.REAL

            if side == TrdSide.BUY:
                px = round(order_px['Low'][symbol], 2)
            else:
                px = round(order_px['High'][symbol], 2)

            order = {
                'code': order_symbol,
                'price': px,
                'qty': shares,
                'trd_side': side,
                'order_type': OrderType.NORMAL,
                'market': TrdMarket.US,     
                'trd_env': trd_env,
                'remark': str(self.portfolio)
            }

            response = self.account.place_order(**order)
            if response is not None:
                outputs.append(response)
            time.sleep(3)

        # store the output orders
        if len(outputs) == 0:
            return pd.DataFrame()

        outputs = pd.concat(outputs)        
        outputs['portfolio'] = str(self.portfolio)
        outputs['date'] = get_today()

        return outputs
    
    def get_position(self) -> DataFrame:
        return self.account.get_position()
    
    def get_order_history(self) -> DataFrame:
        order_hist = self.account.get_order_history()
        if self.portfolio is not None:
            order_hist = order_hist[order_hist['remark'] == str(self.portfolio)]
        return order_hist

        

