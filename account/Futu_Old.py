from futu import *
from utils.logging import get_logger
from utils.credentials import FUTU_TRADE_UNLOCK_PIN
import time

logger = get_logger('Futu')

class Futu():
    '''
        The code here are designed for strategy_v1, some the function is a specific for those strategy
        we can should better refactor this librray so that it can be used in multuple strategy        
    '''
    
    def __init__(self) -> None:                        
        self.logger = logger
        self.pwd_unlock = FUTU_TRADE_UNLOCK_PIN

        # setup used to store the orders
        self.source_key = 'futu'
        self.source_file = os.path.dirname(os.path.realpath(__file__)) + '/strategy_orders.h5'

        # position that we don't liquidate
        self.position_lock = ['US.VOE','US.VBR','US.TSLA','US.SPY','US.QQQ','US.NVDA','US.BRK.B','US.AAPL', 'US.EWY']            

    def store_orders_output(self, orders, date, strategy, trd_env):        
        """
            save the orders to h5df files
            Args:
                df_output:      array of output of place_order()
                strategy:       strategy name
                date:           date of the orders placed
        """ 

        if len(orders) == 0:
            return orders

        orders = pd.concat(orders)        
        orders['strategy'] = strategy
        orders['date'] = date    

        if trd_env == TrdEnv.REAL:
            logger.info('Saving {} orders for strategy {}....'.format(date.strftime('%Y-%m-%d'), strategy))        
            orders.to_hdf(self.source_file, key=self.source_key, format='table', append=True)        

        return orders

    def get_position(self, market=TrdMarket.US):        
        """
            Obtain existing positions
            Args:                
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN
                
        """    
        try:
            with OpenSecTradeContext(filter_trdmarket=self.market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
                ret, data = trd_ctx.position_list_query()
                if ret != RET_OK:
                    self.logger.error(data)
                    raise Exception('Error getting positions')

                l = len(data)
                codes = data['code'].unique()
                self.logger.info('{} Positions: {}'.format(l, ', '.join(codes)))                
                return data                
                    
        except Exception as e:
            self.logger.error(e)

    def place_many_orders(self, orders, date, strategy, trd_env=TrdEnv.REAL, no_duplicate=True):
        """
            place an array of orders, each order is a structure of inputs to place_order()            
        """ 
        outputs = []
        account_pos = self.get_position()
        account_pos = account_pos[account_pos['qty'] != 0]
        symbols_now = account_pos[~account_pos['code'].isin(self.position_lock)]['code'].unique()

        for order in orders:              
            if no_duplicate and order['code'] in symbols_now:
                logger.info(order['code'] + ' Already Traded: skip')
                continue

            response = self.place_order(order['code'], order['price'], order['qty'], order['trd_side'], order['order_type'], market=order['market'], trd_env=trd_env)
            if response is not None:
                outputs.append(response)            
            time.sleep(3)
        
        print(outputs)
        df_output = self.store_orders_output(outputs, date, strategy, trd_env)                            
        return df_output

    def place_order(self, code, price, qty, trd_side, order_type, market=TrdMarket.US, trd_env=TrdEnv.REAL):        
        """
            Max 15 orders over 30 seconds, each consequence order shuold be more than 3s
            Args:
                code:       stock codes
                price:      order prices, arbitrary numbers of order type is market
                qty :       order qty
                trd_side:   TrdSide.BUY | TrdSide.SELL (If we want short selling, we still need to submit SELL)
                order_type: OrderType.MARKET.....
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN
                trd_env:    TrdEnv.REAL | TrdEnv.SIMULATE
        """    

        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
                # unlock for trades
                ret, data = trd_ctx.unlock_trade(self.pwd_unlock)

                # for simulatuion, order type has to be normal
                if trd_env == TrdEnv.SIMULATE:
                    order_type = OrderType.NORMAL

                if ret == RET_OK:
                    ret, data = trd_ctx.place_order(price=price, qty=qty, code=code, order_type=order_type, trd_side=trd_side, trd_env=trd_env)
                    if ret == RET_OK:
                        order = {
                            'code': code,
                            'price': price,
                            'qty': qty,
                            'trd_side': trd_side,
                            'order_type': order_type,
                            'market': market,
                            'trd_env': trd_env,
                        }
                        self.logger.info('Placed Order: {}'.format(order))
                        return data
                    else:
                        self.logger.error('place_order_error: {}'.format(data))
                else:
                    self.logger.error('place_order_error: {}'.format(data))

        except Exception as e:
            self.logger.error(e)

    def order_history(self, start_date=None, end_date=None, market=TrdMarket.US):
        """
            Get all historical orders
            Args:
                start_date: Based on local time (e.g. US Time for US Market)
                end_date:   Based on local time (e.g. US Time for US Market)
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN
                
        """   
        start = start_date.strftime('%Y-%m-%d %H:%M:%S') if start_date is not None else ''
        end = end_date.strftime('%Y-%m-%d %H:%M:%S') if end_date is not None else ''

        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
                ret, data = trd_ctx.history_order_list_query(start=start, end=end)

            if ret == RET_OK:
                return data                
            else:
                self.logger.error('history_order_list_query: ', data)                
        except Exception as e:
            self.logger.error(e)   

    def order_incomplete(self, start_date=None, end_date=None, market=TrdMarket.US):
        """
            Get all incomplete orders
            Args:
                start_date: Based on local time (e.g. US Time for US Market)
                end_date:   Based on local time (e.g. US Time for US Market)
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN
                
        """   
        start = start_date.strftime('%Y-%m-%d %H:%M:%S') if start_date is not None else ''
        end = end_date.strftime('%Y-%m-%d %H:%M:%S') if end_date is not None else ''

        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
                ret, data = trd_ctx.order_list_query(start=start, end=end)

            if ret == RET_OK:
                return data                
            else:
                self.logger.error('history_order_list_query: ', data)                
        except Exception as e:
            self.logger.error(e)

    def liquidate_position(self, codes, date, strategy, all=False, market=TrdMarket.US, trd_env=TrdEnv.REAL):
        """
            Close out all position, this is used for intraday trading strategy
            But function will excluded a predefined lists that's currently holding for long terms            
            Args:
                codes:      Symbols to liqidate
                all:        True if liquidate all positions
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN                
        """          

        positions = self.get_position(market=market)
        positions = positions[~positions['code'].isin(self.position_lock)]

        if not all:
            positions = positions[positions['code'].isin(codes)]

        if not len(positions):
            self.logger.info('no position to liquidate.')

        outputs = []

        for _, row in positions.iterrows():
            code = row['code']
            qty = abs(row['qty'])
            side = row['position_side']
            trd_side = TrdSide.SELL if side == 'LONG' else TrdSide.BUY            

            response = self.place_order(code, 1, qty, trd_side, order_type=OrderType.MARKET, market=market, trd_env=trd_env)
            outputs.append(response)            

            time.sleep(3)

        df_output = self.store_orders_output(outputs, date, strategy, trd_env)

        return df_output
    
    def get_orders_by_strategy(self, strategy=''):
        orders = pd.read_hdf(self.source_file, key=self.source_key)      
        if strategy != '':
            orders = orders[orders['strategy'] == strategy]

        return orders 


if __name__ == "__main__":    
    f = Futu()
    print(f.get_position())
    x = f.place_order('US.AAPL', 1, 1, TrdSide.BUY, OrderType.MARKET, trd_env=TrdEnv.SIMULATE) 
    orders_hist = f.order_history()
    print(orders_hist)    
    f.liquidate_position(all=True, trd_env=TrdEnv.SIMULATE)


    
    



    