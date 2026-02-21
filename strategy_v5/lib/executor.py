"""
Executor Module

Executes backtest of a portfolio over a date range.
Handles data fetching and daily portfolio updates.
"""

import pandas as pd
from strategy_v5.lib.portfolio import Portfolio
from utils.data import get_yahoo_data_formatted
from pandas.tseries.offsets import BDay

class Executor:
    """
    Executes backtest of a portfolio over a date range.
    
    Responsibilities:
    - Fetch OHLC data
    - Iterate through dates
    - Call portfolio.rebalance() at appropriate times
    - Track daily positions and values
    - Store results back to portfolio
    """
    
    def __init__(self, portfolio: Portfolio):
        """
        Initialize executor.
        
        Parameters:
        -----------
        portfolio : Portfolio
            Portfolio object to backtest
        """
        self.portfolio = portfolio
        self.df_ohlc = None
    
    def fetch_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp, lookback=120) -> pd.DataFrame:
        """
        Fetch OHLC data for all instruments using Yahoo Finance.
        
        Parameters:
        -----------
        start_date : pd.Timestamp
            Start date
        
        end_date : pd.Timestamp
            End date
        
        Returns:
        --------
        pd.DataFrame
            Close prices (indexed by date, columns are tickers)
        """
        self.df_ohlc = get_yahoo_data_formatted(
            instruments=self.portfolio.instruments,
            start_date=start_date - BDay(lookback),  # Fetch extra data for lookback period
            end_date=end_date
        )

        return self.df_ohlc
    
    def run(self, start_date: pd.Timestamp, end_date: pd.Timestamp, verbose: bool = True):
        """
        Run backtest over date range using Yahoo Finance data.
        
        Parameters:
        -----------
        start_date : pd.Timestamp
            Backtest start date
        
        end_date : pd.Timestamp
            Backtest end date
        
        verbose : bool
            If True, print progress updates (default: True)
        
        Returns:
        --------
        pd.DataFrame
            Summary results with dates and portfolio values
        """

        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)

        if verbose:
            print(f"\n{'='*80}")
            print(f"EXECUTING BACKTEST: {self.portfolio.strategy.name}")
            print(f"{'='*80}")
            print(f"Instruments: {self.portfolio.instruments}")
            print(f"Initial Capital: ${self.portfolio.initial_capital:,.2f}")
            print(f"Rebalance Frequency: {self.portfolio.rebalance_freq.value}")
            print(f"Date Range: {start_date.date()} to {end_date.date()}")
            print(f"Data Source: Yahoo Finance")
            print(f"{'='*80}\n")
            print("Rebalance Events:")
        
        # Fetch data from Yahoo Finance
        df_ohlc = self.fetch_data(start_date, end_date)
        close_prices = df_ohlc['Close'].loc[start_date:]
        
        # Initialize portfolio (first rebalance on first trading date)        
        first_date = close_prices.index[0]                
        first_prices = close_prices.loc[first_date].values
        first_ohlc = df_ohlc.loc[:first_date - BDay(1)]  # Pass lookback data up to day before first rebalance to avoid lookahead bias

        self.portfolio.rebalance(first_date, first_prices, force=True, ohlc=first_ohlc)        
        
        # Iterate through dates
        for idx, (date, prices_row) in enumerate(close_prices.iterrows()):
            prices = prices_row.values      
            ohlc = df_ohlc.loc[:date - BDay(1)]  # Pass all data up to day before current date to avoid lookahead bias
            
            # Check if should rebalance
            is_rebalanced = self.portfolio.rebalance(date, prices, ohlc=ohlc)
            
            # Update daily values
            self.portfolio.update_daily(date, prices)
            
            if verbose and is_rebalanced:
                pv = self.portfolio.current_portfolio_value
                ret = ((pv / self.portfolio.initial_capital) - 1) * 100
                print(f"  {date.date()}: Portfolio Value = ${pv:,.2f} | Return = {ret:+.2f}%")
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"BACKTEST COMPLETE")
            print(f"{'='*80}\n")
