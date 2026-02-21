"""
Portfolio Module

Manages portfolio state, positions, and rebalancing triggers.
"""

import numpy as np
import pandas as pd
from strategy_v5.lib.strategy import Strategy
from enum import Enum

class RebalanceFrequency(Enum):
    """Enum for rebalance frequencies"""
    DAILY = 'D'
    WEEKLY = 'W'
    BI_WEEKLY = 'BW'
    MONTHLY = 'MS'
    QUARTERLY = 'QS'
    YEARLY = 'YS'

class Portfolio:
    """
    Manages a portfolio with a given strategy.
    
    Responsibilities:
    - Track current positions and portfolio value
    - Store instruments list
    - Handle rebalance frequency (only rebalance when appropriate)
    - Call strategy.rebalance() when needed
    - Keep historical records
    """
    
    def __init__(
        self,                
        instruments: list,
        initial_capital: float,
        strategy: Strategy,
        rebalance_freq: RebalanceFrequency = RebalanceFrequency.MONTHLY,
        rebalance_day: int = 1,
        name: str = None
    ):
        """
        Initialize portfolio.
        
        Parameters:
        -----------
        instruments : list
            List of stock tickers
        
        initial_capital : float
            Starting capital
        
        strategy : Strategy
            Strategy object that determines rebalancing
        
        rebalance_freq : RebalanceFrequency
            How often to rebalance (default: MONTHLY)
            - DAILY: Rebalance every day
            - WEEKLY: Rebalance on specified weekday (rebalance_day: 0=Monday, 6=Sunday)
            - BI_WEEKLY: Rebalance every other week on specified weekday
            - MONTHLY: Rebalance on specified day of month (1-31)
        
        rebalance_day : int
            Day parameter for rebalancing:
            - For WEEKLY/BI_WEEKLY: day of week (0=Monday, 1=Tuesday, ... 6=Sunday)
            - For MONTHLY: day of month (1-31)
            - For DAILY: ignored
        """
        self.instruments = instruments
        self.n_assets = len(instruments)
        self.initial_capital = initial_capital
        self.strategy = strategy
        self.rebalance_freq = rebalance_freq
        self.rebalance_day = rebalance_day
        
        # Current state
        self.current_positions = np.zeros(self.n_assets)  # shares held
        self.current_weights = np.zeros(self.n_assets)  # weights of each asset in portfolio
        self.current_capital = initial_capital
        self.current_portfolio_value = None
        self.current_date = None
        self.name = name if name else strategy.name
        
        # Historical tracking
        self.history = {
            'dates': [],
            'positions': [],
            'weights': [],
            'portfolio_values': [],
            'capitals': [],
            'rebalance_events': []
        }
        
        # Results (populated by Executor)
        self.positions_df = None
        self.values_series = None
    
    def should_rebalance(self, current_date: pd.Timestamp, last_rebalance_date: pd.Timestamp = None) -> bool:
        """
        Check if it's time to rebalance based on frequency and day.
        
        Parameters:
        -----------
        current_date : pd.Timestamp
            Current trading date
        
        last_rebalance_date : pd.Timestamp
            Date of last rebalance (None for first check)
        
        Returns:
        --------
        bool
            True if should rebalance, False otherwise
        """

        # Handle daily rebalance (simplest case)
        if self.rebalance_freq == RebalanceFrequency.DAILY:
            return True
        
        if last_rebalance_date is None:
            # First rebalance - check if we meet the frequency criteria
            if self.rebalance_freq == RebalanceFrequency.WEEKLY:
                # Rebalance on specified weekday (rebalance_day as day of week: 0=Monday, 6=Sunday)
                return current_date.weekday() == self.rebalance_day
            elif self.rebalance_freq == RebalanceFrequency.BI_WEEKLY:
                # Rebalance every other week (rebalance on specific weekday)
                return current_date.weekday() == self.rebalance_day
            elif self.rebalance_freq == RebalanceFrequency.MONTHLY:
                return current_date.day >= self.rebalance_day
            else:
                return True
        
        # Check if we've crossed into a new period
        if self.rebalance_freq == RebalanceFrequency.WEEKLY:
            # Rebalance if we're on the specified weekday (0=Monday, 6=Sunday)
            # and at least 7 days have passed
            days_since_last = (current_date - last_rebalance_date).days
            return current_date.weekday() == self.rebalance_day and days_since_last >= 7
        
        elif self.rebalance_freq == RebalanceFrequency.BI_WEEKLY:
            # Rebalance if we're on the specified weekday AND at least 14 days have passed
            days_since_last = (current_date - last_rebalance_date).days
            return current_date.weekday() == self.rebalance_day and days_since_last >= 14
        
        elif self.rebalance_freq == RebalanceFrequency.MONTHLY:
            # Rebalance if we're in a new month and past rebalance_day
            is_new_month = current_date.month != last_rebalance_date.month
            is_past_day = current_date.day >= self.rebalance_day            
            return is_new_month and is_past_day
        
        return False
    
    def rebalance(self, current_date: pd.Timestamp, current_prices: np.ndarray, force: bool = False, *args, **kwargs) -> bool:
        """
        Execute rebalance if appropriate.
        
        Parameters:
        -----------
        current_date : pd.Timestamp
            Current trading date
        
        current_prices : np.ndarray
            Current prices for all instruments
        
        force : bool
            If True, force rebalance regardless of frequency. This is used to force rebalance on the first trading day to initialize positions.
        
        Returns:
        --------
        bool
            True if rebalance occurred, False otherwise
        """
        # Check if we should rebalance
        last_rebal = self.history['rebalance_events'][-1]['date'] if self.history['rebalance_events'] else None
        
        if not force and not self.should_rebalance(current_date, last_rebal):
            return False
        
        # Calculate current portfolio value
        if self.current_portfolio_value is None:
            # First rebalance - use initial capital
            self.current_portfolio_value = self.initial_capital
        else:
            self.current_portfolio_value = (self.current_positions * current_prices).sum()
        
        # Call strategy to get target positions
        target_positions = self.strategy.rebalance(
            date=current_date,
            prices=current_prices,
            capital=self.current_portfolio_value,
            current_positions=self.current_positions,
            *args,
            **kwargs,
        )        
        
        # Update positions
        old_positions = self.current_positions.copy()
        old_weights = self.current_weights.copy()
        self.current_positions = target_positions
        self.current_weights = self.current_positions * current_prices / self.current_portfolio_value
        
        # Record rebalance event
        rebal_event = {
            'date': current_date,
            'portfolio_value': self.current_portfolio_value,
            'old_positions': old_positions,
            'new_positions': target_positions,
            'old_weights': old_weights,
            'new_weights': self.current_weights,
            'prices': current_prices.copy(),
            'strategy': self.strategy.name
        }        
        self.history['rebalance_events'].append(rebal_event)                
        return True
    
    def update_daily(self, current_date: pd.Timestamp, current_prices: np.ndarray):
        """
        Update portfolio values for current date (without rebalancing).
        
        Parameters:
        -----------
        current_date : pd.Timestamp
            Current trading date
        
        current_prices : np.ndarray
            Current prices for all instruments
        """
        self.current_date = current_date
        self.current_portfolio_value = (self.current_positions * current_prices).sum()
        self.current_weights = self.current_positions * current_prices / self.current_portfolio_value
        
        # Record history
        self.history['dates'].append(current_date)
        self.history['positions'].append(self.current_positions.copy())        
        self.history['weights'].append(self.current_weights.copy())
        self.history['portfolio_values'].append(self.current_portfolio_value)
        self.history['capitals'].append(self.current_portfolio_value)

    @property
    def positions_history(self) -> pd.DataFrame:
        """
        Get historical positions as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Position dataframe with index=dates, columns=instruments
            Values are the number of shares held for each instrument
        """
        if not self.history['dates']:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            self.history['positions'],
            index=self.history['dates'],
            columns=self.instruments
        )
        df.index.name = 'Date'
        return df
    
    @property
    def weights_history(self) -> pd.DataFrame:
        """
        Get historical portfolio weights as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Weights dataframe with index=dates, columns=instruments
            Values are the portfolio weight (0 to 1) for each instrument
        """
        if not self.history['dates']:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            self.history['weights'],
            index=self.history['dates'],
            columns=self.instruments
        )
        df.index.name = 'Date'
        return df
    
    @property
    def portfolio_values_history(self) -> pd.Series:
        """
        Get historical portfolio values as a Series.
        
        Returns:
        --------
        pd.Series
            Portfolio value series with index=dates
            Values are total portfolio value in dollars
        """
        if not self.history['dates']:
            return pd.Series()
        
        series = pd.Series(
            self.history['portfolio_values'],
            index=self.history['dates']
        )
        series.index.name = 'Date'
        series.name = 'PortfolioValue'
        return series
    
    @property
    def capital_history(self) -> pd.Series:
        """
        Get historical capital (total portfolio value) as a Series.
        
        This is equivalent to portfolio_values_history but with a different name
        to emphasize that it represents the total capital available.
        
        Returns:
        --------
        pd.Series
            Capital series with index=dates
            Values are total capital in dollars
        """
        if not self.history['dates']:
            return pd.Series()
        
        series = pd.Series(
            self.history['capitals'],
            index=self.history['dates']
        )
        series.index.name = 'Date'
        series.name = 'Capital'
        return series

    @property
    def rebalance_events(self) -> pd.DataFrame:
        """
        Get historical rebalance events as a DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            Rebalance events dataframe with columns:
            - date: Date of rebalance
            - portfolio_value: Portfolio value at time of rebalance
            - old_positions: Positions before rebalance (array)
            - new_positions: Positions after rebalance (array)
            - prices: Prices at time of rebalance (array)
            - strategy: Strategy name used for rebalance
        """
        if not self.history['rebalance_events']:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history['rebalance_events'])
        df.set_index('date', inplace=True)
        return df
