"""
Momentum Rebalance Strategy

A strategy that allocates capital based on momentum signals.
Assets with positive momentum receive more capital allocation.
"""

import pandas as pd
import numpy as np
from strategy_v5.lib.strategy.strategy import Strategy

class MomentumRebalance(Strategy):
    """
    Strategy that allocates capital based on momentum signals.
    
    Assets with positive momentum get more capital.
    Currently falls back to equal weight - extend this class to implement
    momentum calculation using historical price data.
    
    Example:
    --------
    >>> strategy = MomentumRebalance(lookback_days=20)
    >>> # In a real implementation, you would calculate momentum from historical prices
    >>> # and weight the allocation accordingly
    """
    
    def __init__(self, lookback_days: int = 20):
        """
        Initialize MomentumRebalance strategy.
        
        Parameters:
        -----------
        lookback_days : int, default=20
            Number of days to look back for momentum calculation.
            Higher values use longer-term trends, lower values use recent trends.
        """
        super().__init__("MomentumRebalance")
        self.lookback_days = lookback_days
    
    def rebalance(self, date: pd.Timestamp, prices: np.ndarray, capital: float, current_positions: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Allocate capital based on momentum signals.
        
        Currently falls back to equal weight allocation. To implement true momentum:
        1. Pass historical price data to this method (extend the signature)
        2. Calculate momentum for each asset (e.g., return over lookback_days)
        3. Normalize momentum scores to weights
        4. Allocate capital proportionally to momentum weights
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date (useful for time-based strategies)
        
        capital : float
            Current portfolio value
        
        current_positions : np.ndarray
            Current holdings (not used in current implementation)
        
        Returns:
        --------
        np.ndarray
            Target positions based on momentum (currently equal-weighted)
        """
        # TODO: Implement momentum calculation from price history
        # For now, fallback to equal weight
        n_assets = len(prices)
        capital_per_asset = capital / n_assets
        target_shares = capital_per_asset / prices        
        return target_shares
    
    def set_lookback_days(self, days: int):
        """
        Update lookback period for momentum calculation.
        
        Parameters:
        -----------
        days : int
            New lookback period in days
        """
        self.lookback_days = days
