from datetime import datetime
from strategy_v3.Strategy import STATUS, StrategyBase

class SimpleMarketMakingStrategy(StrategyBase):

    def __init__(self, 
                 instrument:str, 
                 interval:str,                 
                 price_decimal: int = 2,
                 qty_decimal: int = 5,                             
                 status: str = STATUS.RUN,
                 verbose: bool = True,                 
        ):
        '''
            instrument:             The instrument to trade
            interval:               time interval to trade            
            price_decimal:          rounding decimal of price
            qty_decimal:            rounding decimal of quantity
            status:                 user can set status to control the strategy behavior
            verbose:                True to print the log message
        '''
        super().__init__(
            instrument=instrument,
            interval=interval,
            price_decimal=price_decimal,
            qty_decimal=qty_decimal,
            status=status,
            verbose=verbose,
        )    

    def execute(self, data):
        pass

    def close_out_positions(self, 
                            type: str = 'close', 
                            price: float = None, 
                            date: datetime = None):        
        return super().close_out_positions(type, price, date)
    
    