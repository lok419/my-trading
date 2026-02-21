"""
Monthly Even Rebalance Strategy

A simple strategy that allocates capital evenly across all instruments.
This strategy rebalances monthly and divides available capital equally among all assets.
"""

import numpy as np
import pandas as pd
from strategy_v5.lib.strategy.strategy import Strategy

class WeightRebalance(Strategy):
    """
    Simple strategy: allocate capital evenly across all instruments.
    
    This is the monthly rebalancing example - it divides capital equally among all assets.
    Each asset receives an equal dollar amount to invest, regardless of market conditions.
    
    Example:
    --------
    >>> strategy = WeightRebalance()
    >>> prices = np.array([100, 200, 150])  # 3 assets
    >>> capital = 30000  # $30,000
    >>> positions = strategy.rebalance(prices, capital, np.array([0, 0, 0]))
    >>> positions
    array([100., 50., 66.66...])  # 100 shares @ $100, 50 @ $200, ~67 @ $150
    """
    
    def __init__(self, weights: np.ndarray = None):
        """Initialize WeightRebalance strategy"""
        super().__init__("WeightRebalance")
        self.weights = np.array(weights) if weights is not None else None

        if self.weights is not None:
            if not np.isclose(self.weights.sum(), 1.0):
                raise ValueError("Weights must sum to 1.0")
            if np.any(self.weights < 0):
                raise ValueError("Weights cannot be negative")
    
    def rebalance(self, date: pd.Timestamp, prices: np.ndarray, capital: float, current_positions: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Allocate capital equally across all instruments.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date (useful for time-based strategies)
        
        prices : np.ndarray
            Current prices for each instrument
        
        capital : float
            Current portfolio value (will be divided equally)
        
        current_positions : np.ndarray
            Not used in this strategy (purely price-based allocation)
        
        Returns:
        --------
        np.ndarray
            Equal-weighted positions (number of shares for each asset)
        """
        if self.weights is not None:
            target_shares = capital * self.weights / prices
            return target_shares
        else:
            n_assets = len(prices)
            capital_per_asset = capital / n_assets
            target_shares = capital_per_asset / prices    
            return target_shares
