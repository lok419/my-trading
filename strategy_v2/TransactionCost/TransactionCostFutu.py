from pandas import DataFrame
from pandas.core.api import Series as Series
from strategy_v2.TransactionCost import TransactionCostModel
import numpy as np

class TransactionCostFutu(TransactionCostModel):
    '''
        Stock Only
        Commission	                    $0.0049/Share, minimum $0.99/order

        Fixed platform fees	            $0.005/Share, minimum $1/order

        Tiered platform fees (Accumlated monthly shares)	Fees per share (minimum $1/order)        
        1-500th shares	                $0.0100
        501-1,000th shares	            $0.0080
        1,001-5,000th shares	        $0.0070
        5,001-10,000th shares	        $0.0060
        10,001-50,000th shares	        $0.0055
        50,001-200,000th shares	        $0.0050
        200,001-500,000th shares	    $0.0045
        500,001-1,000,000th shares	    $0.0040
        1,000,001-5,000,000th shares	$0.0035
        5,000,001st shares and above	$0.0030

        Settlement Fees	                $0.003/Share
        SEC fees	                    $0.000008 * transaction amount, min $0.01/trade (sells only)
        Trading Activity Fees	        $0.000166/Share, min $0.01/trade, max $8.30/trade (sells only)
    '''

    def __init__(self):
        pass

    def compute(self, position_shs: DataFrame, px: DataFrame) -> Series:
        # Assume we use fixed platform fees
        turnover = position_shs.diff()        
        turnover.iloc[0] = position_shs.iloc[0]

        self.commission = np.where(turnover != 0, np.maximum(turnover * 0.0049, 0.99), 0)
        self.platform_fees = np.where(turnover != 0, np.maximum(turnover * 0.005, 1), 0)
        self.settlement_fees = turnover * 0.003        
        self.sec_fees = np.where(turnover < 0, np.maximum(turnover * px * 0.000008, 0.01), 0)
        self.trading_act_fees = np.where(turnover < 0, np.clip(0.000166 * turnover, 0.01, 8.3), 0)

        self.total_cost = self.commission + self.platform_fees + self.settlement_fees + self.sec_fees + self.trading_act_fees
        self.total_cost_ts = self.total_cost.sum(axis=1)        
        
        return self.total_cost_ts