from futu import *
from utils.credentials import FUTU_TRADE_UNLOCK_PIN
from utils.logging import get_logger
from pandas.core.frame import DataFrame
from account.AccountModel import AccountModel

class Futu(AccountModel):
    '''
        Futu Utility class which consolidates all native functions here (try to make less strategy specific logic here so that it could be used anywhere)
    '''
    
    def __init__(self) -> None:                                
        self.pwd_unlock = FUTU_TRADE_UNLOCK_PIN            
        self.logger = get_logger('Futu')

    def get_position(self, market=TrdMarket.US) -> DataFrame:        
        """
            Obtain existing positions
            Args:                
                market:     TrdMarket.US | TrdMarket.HK | TrdMarket.CN                
        """    
        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
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

    def place_order(self, 
                    code:str, 
                    price:float, 
                    qty:float, 
                    trd_side:str, 
                    order_type:str, 
                    market:str=TrdMarket.US, 
                    trd_env:str=TrdEnv.SIMULATE,
                    remark:str='',
        ):        
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
                    ret, data = trd_ctx.place_order(price=price, qty=qty, code=code, order_type=order_type, trd_side=trd_side, trd_env=trd_env, remark=remark)
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
                        self.logger.error('place_order_error ({}): {}'.format(code, data))
                else:
                    self.logger.error('place_order_error ({}): {}'.format(code, data))

        except Exception as e:
            self.logger.error(e)

    def get_order_history(self, start_date=None, end_date=None, market=TrdMarket.US):
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

    def get_order_incomplete(self, start_date=None, end_date=None, market=TrdMarket.US):
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

    def get_capital(self, market=TrdMarket.US) -> DataFrame:
        """
            Get Account Capital
        """
        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:
                ret, data = trd_ctx.accinfo_query()
                if ret == RET_OK:
                    return data
                else:
                    print('accinfo_query error: ', data)
                trd_ctx.close() 

        except Exception as e:
            self.logger.error(e)

    def cancel_all_orders(self, market=TrdMarket.US) -> DataFrame:
        """
            Cancel all outstanding unfilled orders
        """
        try:
            with OpenSecTradeContext(filter_trdmarket=market, host='127.0.0.1', port=11111, security_firm=SecurityFirm.FUTUSECURITIES) as trd_ctx:

                ret, data = trd_ctx.unlock_trade(self.pwd_unlock)  
                if ret == RET_OK:
                    ret, data = trd_ctx.cancel_all_order()
                    if ret == RET_OK:
                        return data
                    else:
                        self.logger.error('cancel_all_order error: ', data)
                else:
                    self.logger.error('unlock_trade failed: ', data)
        except Exception as e:
            self.logger.error(e)


    def symbol_converter(self, 
                         symbol: str, 
                         output_type: str=''):
        '''
            Function to convert the general symbol (symbol used in this infra) to Futu specific symbol
        '''
        if output_type == '':
            new_symbol = symbol.replace('US.', '').replace('BRK.B', 'BRK-B')
        elif output_type == 'futu':
            new_symbol = 'US.' + symbol.replace('BRK-B', 'BRK.B')
        else:
            raise 'Unrecognized symbol type'
        return new_symbol