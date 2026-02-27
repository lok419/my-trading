"""
Executor Module

Executes backtest of a portfolio over a date range.
Handles data fetching and daily portfolio updates.
Also supports live trading execution via Futu broker.
"""

import pandas as pd
import numpy as np
from strategy_v5.lib.portfolio import Portfolio
from utils.data import get_yahoo_data_formatted
from pandas.tseries.offsets import BDay
from utils.logging import get_logger
from account.Futu import Futu
from futu import TrdSide, TrdEnv, OrderType, TrdMarket
import time
from IPython.display import display

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

        # this define the lookback period for fetching data to ensure we have enough history for the first rebalance
        self.lookback = self.portfolio.strategy.lookback_days * 2 if hasattr(self.portfolio.strategy, 'lookback_days') else 120
        self.logger = get_logger(f"Executor-{self.portfolio.name}")
    
    def fetch_data(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
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
            start_date=start_date - BDay(self.lookback),  # Fetch extra data for lookback period
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
        self.portfolio.reset()  # Reset portfolio to initial state before backtest

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

    def _convert_ticker(self, ticker: str, to_futu: bool = True) -> str:
        """
        Convert between Yahoo and Futu ticker formats.
        
        Yahoo format: 'BRK.B', 'AAPL'
        Futu format: 'US.BRK-B', 'US.AAPL'
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol to convert
        to_futu : bool
            If True, convert Yahoo → Futu. If False, convert Futu → Yahoo.
        
        Returns:
        --------
        str
            Converted ticker
        """
        if to_futu:
            # Yahoo to Futu: BRK-B → BRK.B, add US prefix
            ticker = ticker.replace('BRK-B', 'BRK.B')
            ticker = f'US.{ticker}' if not ticker.startswith('US.') else ticker
        else:
            # Futu to Yahoo: remove US prefix, BRK.B → BRK-B
            ticker = ticker.replace('US.', '')
            ticker = ticker.replace('BRK.B', 'BRK-B')
        return ticker

    def _display_rebalance_summary(self, df_rebal: pd.DataFrame):
        """
        Display a formatted rebalance summary table.
        
        Parameters:
        -----------
        df_rebal : pd.DataFrame
            Rebalance dataframe with columns: price, current, current_mv, target, target_mv, shares, mv, buy_mv, sell_mv
        """        
        df = df_rebal.copy()
        df['mv'] = df['shares'] * df['price']
        df['current_mv'] = df['current'] * df['price']
        df['target_mv'] = df['target'] * df['price']
        df['mv'] = (df['shares'] * df['price'])
        df['buy_mv'] = df.apply(lambda row: row['mv'] if row['shares'] > 0 else 0, axis=1)
        df['sell_mv'] = df.apply(lambda row: -1 * row['mv'] if row['shares'] < 0 else 0, axis=1)
        df = df[['price', 'current', 'current_mv', 'target', 'target_mv', 'shares', 'mv', 'buy_mv', 'sell_mv']]

        df_total = df[['current_mv', 'target_mv', 'mv', 'buy_mv', 'sell_mv']]
        df_total = df_total.sum()
        df_total.name = 'TOTAL'
        df = pd.concat([df, df_total.to_frame().T])

        df.columns = pd.MultiIndex.from_tuples([            
            ('Price', ''), 
            ('Current', 'Shares'), 
            ('Current', 'MV'), 
            ('Target', 'Shares'), 
            ('Target', 'MV'), 
            ('Rebalance', 'Shares'), 
            ('Rebalance', 'MV'), 
            ('Rebalance', 'Buy MV'), 
            ('Rebalance', 'Sell MV')
        ])   

        style = df.style.set_caption("Rebalance Summary").set_table_styles([
            {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('color', '#e0e0e0'), ('caption-side', 'top'), ('padding', '10px')]},
            {'selector': 'th', 'props': [('background-color', '#2d2d2d'), ('color', '#e0e0e0'), ('font-size', '12px'), ('border', '1px solid #444'), ('text-align', 'center'), ('vertical-align', 'middle')]},
            {'selector': 'td', 'props': [('padding', '8px'), ('text-align', 'center'), ('font-size', '12px'), ('border', '1px solid #444'), ('color', '#e0e0e0')]},
            {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#1e1e1e')]},
        ])  
        style = style.format({
            ('Price', ''): '${:,.2f}'.format,
            ('Current', 'Shares'): '{:,.0f}'.format,
            ('Current', 'MV'): '${:,.0f}'.format,
            ('Target', 'Shares'): '{:,.0f}'.format,
            ('Target', 'MV'): '${:,.0f}'.format,
            ('Rebalance', 'Shares'): '{:,.0f}'.format,
            ('Rebalance', 'MV'): '${:,.0f}'.format,
            ('Rebalance', 'Buy MV'): '${:,.0f}'.format,
            ('Rebalance', 'Sell MV'): '${:,.0f}'.format,
        }, na_rep='')

        display(style)

    def execute(self, date: pd.Timestamp, is_live: bool = False, force: bool = False, use_spot_mv: bool = False) -> pd.DataFrame:
        """
        Execute live trading orders via Futu broker.
        
        Parameters:
        -----------
        date : pd.Timestamp
            The date to check for rebalancing and execute orders
        is_live : bool
            If True, execute real orders (TrdEnv.REAL). If False, simulate only (TrdEnv.SIMULATE).
        force : bool
            If True, force execution regardless of rebalance schedule
        use_spot_mv : bool
            If True, derive target shares based on current realized total MV from spot (Futu).
            Target weights come from the strategy, quantities calculated from actual holdings.
            Default is False (uses portfolio target positions).
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing placed orders
        """
        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        self.run(date, date, verbose=False)  # Run backtest for the single date to get target positions updated

        self.logger.info(f"\nExecuting live orders - {date.date()}")
        market = TrdMarket.US
        
        # for now the executor don't handle the rebalance frequency, completely depends on users to call the execute function at the right time.
        # if not force and not self.portfolio.should_rebalance(date):
        #     self.logger.info(f"No rebalance scheduled for {date.date()}")
        #     return pd.DataFrame()
        
        futu_account = Futu()
        
        # Get current positions from Futu
        current_data = futu_account.get_position(market=market)
        if current_data is None or current_data.empty:
            raise Exception("No positions returned from Futu API - possible connection issue")
        
        current = current_data.copy()
        current['instrument'] = current['code'].apply(lambda x: self._convert_ticker(x, to_futu=False))
        current = current[['instrument', 'qty']].set_index('instrument')
        current.columns = ['current']
        
        # Get target from portfolio
        target = pd.DataFrame(self.portfolio.positions_history.loc[date]).drop('CASH', errors='ignore')
        target.columns = ['target']
        target = target[target['target'] > 0]
        
        # Build rebalance dataframe
        df_rebal = target.join(current, how='left').fillna(0)
        
        if df_rebal.empty:
            self.logger.info("No trades needed")
            return pd.DataFrame()
        
        # Get prices for target instruments
        prices = get_yahoo_data_formatted(
            instruments=df_rebal.index.tolist(),
            interval='15m',
            period='1d',
            use_cache=False
        )
        prices = prices.ffill()
        price_last = prices['Close'].iloc[-1]
        df_rebal['price'] = df_rebal.index.map(price_last)
        
        # Calculate target shares based on rebalance mode
        if use_spot_mv:
            # Derive target from weights and current spot MV
            target_weights = pd.DataFrame(self.portfolio.weights_history.loc[date]).drop('CASH', errors='ignore')
            target_weights.columns = ['target_weight']
            target_weights = target_weights[target_weights['target_weight'] > 0]
            
            current_mv = (df_rebal['current'] * df_rebal['price']).sum()
            df_rebal['target'] = np.floor(target_weights.loc[df_rebal.index, 'target_weight'] * current_mv / df_rebal['price']).fillna(0).astype(int)
        
        df_rebal['shares'] = (df_rebal['target'] - df_rebal['current']).astype(int)        

        # Display rebalance summary
        self._display_rebalance_summary(df_rebal)
        
        # Prompt user for confirmation
        user_input = input("\nProceed with execution? (Y/N): ").strip().upper()
        if user_input != 'Y':
            self.logger.info("Execution cancelled by user")
            return pd.DataFrame()
        
        # Cancel all existing orders before executing
        self.logger.info("Cancelling all existing orders")
        futu_account.cancel_all_orders(market=market)
        time.sleep(3)
        
        # Execute orders
        trd_env = TrdEnv.REAL if is_live else TrdEnv.SIMULATE
        executed_orders = []
        
        for instrument, row in df_rebal.iterrows():
            shares = abs(row['shares'])
            side = TrdSide.BUY if row['shares'] > 0 else TrdSide.SELL
            price = round(float(row['price']), 2)
            futu_code = self._convert_ticker(instrument, to_futu=True)
            
            response = futu_account.place_order(
                code=futu_code,
                price=price,
                qty=shares,
                trd_side=side,
                order_type=OrderType.NORMAL,
                market=market,
                trd_env=trd_env,
                remark=f"{self.portfolio.name} | {date.date()}"
            )
            
            if response is not None:
                executed_orders.append({
                    'instrument': instrument,
                    'side': 'BUY' if side == TrdSide.BUY else 'SELL',
                    'qty': shares,
                    'price': price,
                    'date': date
                })
            
            time.sleep(2)
        
        self.logger.info(f"Executed {len(executed_orders)} orders")
        return pd.DataFrame(executed_orders) if executed_orders else pd.DataFrame()
