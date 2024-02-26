import time
import schedule
from account.Futu import Futu
from strategy_v1.StrategyBase import StrategyBase
from utils.performance import *
from utils.data import *
from futu import TrdEnv, TrdSide, OrderType, TrdMarket

class IntradayStrategyBase(StrategyBase):         

    def __init__(self, strategy_name):        
        super().__init__(strategy_name)               

    def generate_backtest_return(self):
        """
            Compute the strategy return after generation_position()

            This is Close to Open return for intraday
        """

        px = self.px        
        ret = px['Close'] - px['Open']        
        ret = ret.fillna(0)
        ret = ret[self.start_date:self.end_date]
        self.ret = ret

        # Commission for round trip, assume all position are liquidate intraday
        # Actal commission on Futu is around $2.5 per $4000 notional
        self.comms = (-1 * np.abs(self.position * px['Open'][self.start_date:self.end_date]) * 2.5/4000 * 2).sum(axis=1)        

        # Each return        
        port_ret_symbols = self.position * ret                
        self.port_ret = (port_ret_symbols.sum(axis=1) + self.comms) / self.capital        

        return self 

    def enter_strategy(self, is_test=True): 
        """
            Function to execute the intraday strategy
        """        
        acc = Futu()        
        orders = []

        # make sure this is for one date noly
        assert(self.start_date == self.end_date)
        assert(not self.is_backtest)

        today_orders = self.df_trade_stats[self.df_trade_stats['position'] != 0]
        today_orders = today_orders[today_orders['shares'] != 0]

        for _, row in today_orders.iterrows():
            order_symbol = 'US.' + row['symbol']
            shares = abs(row['shares'])

            if is_test:
                side = TrdSide.BUY
                trd_env = TrdEnv.SIMULATE
            else:
                side = TrdSide.BUY if row['shares'] > 0 else TrdSide.SELL
                trd_env = TrdEnv.REAL    

            orders.append({
                'code': order_symbol,
                'price': 1,
                'qty': shares,
                'trd_side': side,
                'order_type': OrderType.MARKET,
                'market': TrdMarket.US,        
            })

        self.execute_orders = acc.place_many_orders(orders, self.end_date, strategy=self.strategy_name, trd_env=trd_env, no_duplicate=True)
        
        return self
    
    def exit_strategy(self, exit_time='', is_test=True):
        """
            Function to exit the intraday strategy by end of day
        """       

        acc = Futu()
        account_pos = acc.get_position()
        account_pos = account_pos[~account_pos['code'].isin(acc.position_lock)]
        account_pos = account_pos[account_pos['qty'] != 0]

        # same name might be included by another strategy
        today_strategy_order = acc.get_orders_by_strategy(self.strategy_name)
        today_strategy_order = today_strategy_order[today_strategy_order['date'] == self.end_date]
        today_strategy_order = today_strategy_order['code'].unique()

        #today_order_symbols = self.df_trade_stats[self.df_trade_stats['shares'] != 0]['symbol'].unique()
        #today_order_symbols = list(map(lambda x: f'US.{x}', today_order_symbols))
        #today_order_symbols = np.intersect1d(today_order_symbols, today_strategy_order)            

        today_order_symbols = np.intersect1d(today_strategy_order, account_pos['code'].unique())

        trd_env = TrdEnv.REAL if not is_test else TrdEnv.SIMULATE

        self.logger.info('Total {} stocks to close....'.format(len(today_order_symbols)))
        self.logger.info(today_order_symbols)

        def liquidate_position(today_order_symbols):
            acc = Futu()
            self.exit_orders = acc.liquidate_position(today_order_symbols, self.end_date, self.strategy_name, trd_env=trd_env)    
            return schedule.CancelJob
        
        if exit_time == '':
            liquidate_position(today_order_symbols)
        else:    
            schedule.every().day.at(exit_time).do(liquidate_position, today_order_symbols=today_order_symbols)

            try:
                while True:
                    schedule.run_pending()
                    time.sleep(1)
                    if len(schedule.jobs) == 0:
                        break
            except Exception as e:
                self.logger.error(e)
            finally:
                [schedule.cancel_job(job) for job in schedule.jobs]  

        return self
        

        
    