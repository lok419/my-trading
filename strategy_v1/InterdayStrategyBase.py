import time
from strategy_v1.StrategyBase import StrategyBase
from utils.performance import *
from utils.data import *

class InterdayStrategyBase(StrategyBase):         

    def __init__(self, strategy_name):        
        super().__init__(strategy_name)     
        self.capital = 10000

    def generate_backtest_return(self):
        """
            Compute the strategy return after generation_position()

            This is Close to Close return for interday WITHOUT daily rebalance

            e.g. 
                Price @ T = [100,100]
                Price @ T+1 = [120, 80]

                Portfolio Weights @ T = [0.5, 0.5]
                Portoflio Shares  @ T = [1, 1]
                Portfolio Values  @ T = [100, 100]

                # We assume no daily rebalance
                Portfolio Values  @ T+1 = [120, 80]
                Portfolio Shares  @ T+1 = [1,1]
                Portfolio Weights @ T+1 = [0.6, 0.4]

                So when we calculate the return, we uses 
                    "Portfolio Shares * Dollar Return per Shares" 
                instead of 
                    "Portfolio weights * Pct Return of stock"
        """  
                        
        close_px = self.px['Close'].loc[self.position.index]

        self.ret = close_px / close_px.shift(1) - 1

        # Calculate the commission
        # Actal commission on Futu is around $2.5 per $4000 notional
        self.turnover = self.position.diff().fillna(0)
        self.turnover_ntl = self.turnover * close_px
        self.comms = -1 * (np.abs(self.turnover_ntl) * 2.5/4000).sum(axis=1)

        # the first day of entry should have no return        
        ret_dollar = (self.position * close_px.diff().shift(-1))

        self.port_ret_dollar = ret_dollar.sum(axis=1)

        # we want to convert dollar return as all of the performance function is based on dollar return
        self.captial_ts = (self.capital + self.port_ret_dollar.cumsum() + self.comms.cumsum())
        self.port_ret = self.captial_ts.pct_change().fillna(0)

        return self