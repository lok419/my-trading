import pandas as pd
import numpy as np
from pandas.core.api import DataFrame as DataFrame, Series
from strategy_v2.TradingSubSystem import TradingSubSystemBase
from utils.performance import annualized_volatility_ts

class TradingSubSystemSingle(TradingSubSystemBase):
    '''
        Base Trading SubSystem which includes basic function of getting data, backtest, performance

        This SubSystem Class is designed for basic Buy and Hold strategy or simple algorithm rule based strategy, given
            - Only support single instrument            
            - Equal weighted weights from strategy variations
            - Position sizing is based on instrument's volatility and your volatility target
    '''

    def __init__(self, *args, **kwargs):   
        super().__init__(*args, **kwargs)                        

    def optimize(self):
        '''
            Optimize the capital weights among strategy variations
            1.  Combine different strategy weights from different strategy
            2.  Scale the combined weights to match your volatility target
            2a. We use the backtest returns to derive the historical volatility and use that to scale to the volatility target
        '''

        # Combine and produce the final weights
        # combnie_weights() transforms position from (strategy-symbol) to symbol dataframe
        self.combined_position = self.combnie_weights(self.ret, self.position)
        self.combined_ret = self.generate_backtest_return(self.combined_position)        

        self.scale_factor, self.px_vol = self.position_sizing(self.combined_ret, self.vol_target, px_vol_windows=20)
        self.scaled_combined_position = np.minimum(self.combined_position.mul(self.scale_factor, axis=0), self.max_leverage)
        self.scaled_combined_ret = self.generate_backtest_return(self.scaled_combined_position)
        self.logger.info('Volatility Target = {:.1f}% | Price Volatility = {:.1f}% | Last Scale Factor = {:.2f}'.format(self.vol_target*100, self.px_vol*100 ,self.scale_factor[-1]))

    def combnie_weights(self, 
                        ret: pd.Series, 
                        position: pd.DataFrame
        ) -> pd.DataFrame:
        """
            Combine the weights from different strategy variations
            Simplest way is to use simple average.
            We can also adjust with the correlation of the strategies, because the combined ways are likely to be samller 
            when they're not perfectly correlated            
        """

        strategies = [x[0] for x in position.columns]
        for i, s in enumerate(strategies):
            combined_position = position[s] if i == 0 else combined_position + position[s]

        combined_position /= len(strategies)
        return combined_position

    def position_sizing(self, 
                        px_ret: pd.Series, 
                        vol_target: float, 
                        px_vol_windows: float=20,
        ) -> (Series, float):         # type: ignore
        """
            Derive the position sizing based on your volatiltiy target and instrument volatility
            We use Simple volatility for now

            Need to shift the scale factor by 1 days because we only know the volatility on T-1 when we sizing the position on T
        """        
        close_vol = annualized_volatility_ts(px_ret, windows=px_vol_windows)        
        return (vol_target / close_vol).shift(1).bfill(), close_vol[-1]

    def get_position(self) -> DataFrame:
        return self.scaled_combined_position
    
    def get_return(self) -> Series:
        return self.scaled_combined_ret
    

    