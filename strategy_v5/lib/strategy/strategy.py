"""
Strategy Module

Defines abstract Strategy class and concrete strategy implementations.
Users extend Strategy class to create custom rebalancing strategies.
"""

import numpy as np
from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    """
    Abstract base class for rebalancing strategies.
    
    A strategy defines how to allocate capital across instruments given:
    - Current prices
    - Current capital
    - Current positions
    
    Users should extend this class to implement custom rebalancing logic.
    """
    
    def __init__(self, name: str):
        """
        Initialize strategy.
        
        Parameters:
        -----------
        name : str
            Name of the strategy
        """
        self.name = name
    
    @abstractmethod
    def rebalance(self, date: pd.Timestamp, prices: np.ndarray, capital: float, current_positions: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Calculate target positions (number of shares) for each instrument.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date (useful for time-based strategies)
            
        prices : np.ndarray
            Current price for each instrument (array of length n_assets)
        
        capital : float
            Current total capital available
        
        current_positions : np.ndarray
            Current number of shares for each instrument (array of length n_assets)
        
        Returns:
        --------
        np.ndarray
            Target number of shares for each instrument
        """
        pass


# Concrete implementations have been moved to separate files:
# - monthly_even_rebalance.py: MonthlyEvenRebalance
# - momentum_rebalance.py: MomentumRebalance
# Import them as: from monthly_even_rebalance import MonthlyEvenRebalance
