"""
MVO Mean Reversion Strategy

Mean-Variance Optimization portfolio using cvxpy with mean reversion signals.
Maximizes expected return - risk_aversion_coefficient/2 * portfolio_variance
subject to constraints on weights (fully invested, no short selling, max concentration).

Mean reversion strategy: allocate more to assets below their lookback mean price
(undervalued, expected to revert up) and less to assets above mean (overvalued,
expected to revert down).
"""

import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from strategy_v5.lib.strategy.strategy import Strategy
from utils.logging import get_logger


class MVOMeanRevertRebalance(Strategy):
    """
    Mean-Variance Optimization (MVO) strategy with mean reversion signals using cvxpy.
    
    Optimizes portfolio weights to maximize:
        Expected Return - (lambda / 2) * Portfolio Variance
    
    where expected returns are based on mean reversion signals:
    - Assets trading below their lookback mean are expected to revert UP (positive return)
    - Assets trading above their lookback mean are expected to revert DOWN (negative return)
    - Strength of signal is proportional to distance from mean
    
    Constraints:
    - Weights sum to 1 (fully invested)
    - Weights >= 0 (no short selling)
    - Weights <= max_weight (no concentration)
    
    Parameters can be customized for different risk profiles and concentration limits.
    
    Example:
    --------
    >>> strategy = MVOMeanRevertRebalance(
    ...     lookback_days=60,    
    ...     risk_aversion=1.0,
    ...     max_weight=0.3,
    ...     use_shrinkage=True
    ... )
    """
    
    def __init__(
        self,
        lookback_days: int = 60,        
        risk_aversion: float = 1.0,
        max_weight: float = 0.3,
        use_shrinkage: bool = True,        
    ):
        """
        Initialize MVOMeanRevertRebalance strategy.
        
        Parameters:
        -----------
        lookback_days : int, default=60
            Number of trading days to use for mean and covariance estimation.                
        
        risk_aversion : float, default=1.0
            Risk aversion coefficient (lambda).
            Higher values => more risk-averse (lower risk portfolio)
            Lower values => more aggressive (higher risk for higher return)
        
        max_weight : float, default=0.3
            Maximum weight allowed for any single asset (0 to 1).
            Use to prevent concentration risk.
            e.g., 0.3 means no asset can be >30% of portfolio
        
        use_shrinkage : bool, default=True
            Whether to apply covariance matrix shrinkage (Ledoit-Wolf).
            Recommended for small lookback periods or high-dimensional data.        
        """
        super().__init__(f"MVOMeanRevert(lookback_days={lookback_days}, risk_aversion={risk_aversion}, max_weight={max_weight})")
        self.lookback_days = lookback_days        
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.use_shrinkage = use_shrinkage        
        self.logger = get_logger(self.name)
    
    def _estimate_expected_returns(self, price_history: pd.DataFrame) -> np.ndarray:
        """
        Estimate expected returns using mean reversion signals.
        
        Assets below their mean are expected to revert UP (positive return).
        Assets above their mean are expected to revert DOWN (negative return).
        The strength is proportional to the distance from the mean, annualized.
        
        Parameters:
        -----------
        price_history : pd.DataFrame
            Historical close prices. Shape: (n_periods, n_assets)
        
        Returns:
        --------
        np.ndarray
            Expected returns for each asset (annualized mean reversion signal)
        """
        # Get current prices (last row) and historical mean
        current_prices = price_history.iloc[-1].values
        mean_prices = price_history.mean().values
        
        # Calculate deviation from mean as a percentage
        deviation_from_mean = (current_prices - mean_prices) / mean_prices
        
        # Mean reversion expected return: negative of deviation
        # If price is 10% below mean, we expect +10% reversion return
        # If price is 10% above mean, we expect -10% reversion return
        mean_revert_returns = -deviation_from_mean
        
        # Annualize the mean reversion signal (252 trading days per year)
        annualized_returns = mean_revert_returns * 252
        
        return annualized_returns
    
    def _estimate_covariance(self, price_history: pd.DataFrame) -> np.ndarray:
        """
        Estimate covariance matrix from historical returns.
        Optionally applies Ledoit-Wolf shrinkage (via sklearn) to improve robustness.
        
        Parameters:
        -----------
        price_history : pd.DataFrame
            Historical close prices. Shape: (n_periods, n_assets)
        
        Returns:
        --------
        np.ndarray
            Annualized covariance matrix
        """
        # Calculate daily returns
        daily_returns = price_history.pct_change().dropna()
        returns_array = daily_returns.values
        
        # Apply Ledoit-Wolf shrinkage if requested
        if self.use_shrinkage:
            lw = LedoitWolf()
            cov_matrix, _ = lw.fit(returns_array).covariance_, lw.shrinkage_
        else:
            cov_matrix = np.cov(returns_array.T)
        
        # Annualize covariance (252 trading days per year)
        annualized_cov = cov_matrix * 252
        
        return annualized_cov
    
    def _optimize_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        n_assets: int
    ) -> np.ndarray:
        """
        Optimize portfolio weights using cvxpy.
        
        Maximizes: expected_returns.T @ w - (lambda / 2) * w.T @ cov_matrix @ w
        
        Subject to:
        - sum(w) == 1
        - w >= 0
        - w <= max_weight
        
        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns for each asset (from mean reversion signals)
        
        cov_matrix : np.ndarray
            Covariance matrix of returns
        
        n_assets : int
            Number of assets
        
        Returns:
        --------
        np.ndarray
            Optimal portfolio weights
        """
        # Decision variable: portfolio weights
        w = cp.Variable(n_assets)
        
        # Objective: maximize return - (lambda/2) * variance
        # Note: cvxpy minimizes, so negate for maximization
        portfolio_return = expected_returns @ w
        portfolio_variance = cp.quad_form(w, cov_matrix)
        objective = portfolio_return - (self.risk_aversion / 2) * portfolio_variance
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,              # Fully invested
            w >= 0,                      # No short selling
            w <= self.max_weight         # Max weight constraint
        ]
        
        # Solve
        problem = cp.Problem(cp.Maximize(objective), constraints)
        
        try:
            problem.solve(solver=cp.CVXOPT, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                return w.value
            else:
                # Fallback to equal weight if optimization fails
                self.logger.warning(f"Warning: Optimization failed with status {problem.status}. Using equal weights.")
                return np.ones(n_assets) / n_assets
        except Exception as e:
            self.logger.warning(f"Warning: Optimization error: {e}. Using equal weights.")
            return np.ones(n_assets) / n_assets
    
    def rebalance(
        self,
        date: pd.Timestamp,
        prices: np.ndarray,
        capital: float,
        current_positions: np.ndarray,
        *args,
        **kwargs
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights using MVO with mean reversion signals.
        Convert to target shares.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date
        
        prices : np.ndarray
            Current prices for each asset
        
        capital : float
            Current portfolio value
        
        current_positions : np.ndarray
            Current holdings (not used)
        
        **kwargs :
            Expects 'ohlc' key containing historical OHLC data as DataFrame
        
        Returns:
        --------
        np.ndarray
            Target number of shares for each asset
        """
        # Get historical OHLC data
        ohlc = kwargs.get('ohlc', None)
        assert ohlc is not None, "OHLC data must be provided in kwargs for MVO optimization"                        
        assert ohlc.index.max() < date, "OHLC data should only contain historical data"

        # Extract lookback period data
        ohlc = ohlc.loc[:date].tail(self.lookback_days)                
        assert len(ohlc) >= self.lookback_days, "Not enough historical data for MVO optimization"
        
        # Extract close prices for each asset
        close_prices = ohlc['Close']                                    
        n_assets = len(prices)
        
        # Estimate expected returns (mean reversion signals) and covariance
        expected_returns = self._estimate_expected_returns(close_prices)
        cov_matrix = self._estimate_covariance(close_prices)
        
        # Ensure covariance matrix is positive definite
        try:
            np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Add small regularization to diagonal if not PD
            self.logger.warning("Covariance matrix not positive definite. Adding regularization.")
            cov_matrix += np.eye(n_assets) * 1e-6
        
        # Optimize weights
        optimal_weights = self._optimize_weights(expected_returns, cov_matrix, n_assets)
        
        # Convert weights to target shares
        target_shares = (optimal_weights * capital) / prices
        return target_shares