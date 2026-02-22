# Strategy V5 Backtesting Framework

A modular, production-ready framework for building, backtesting, and evaluating portfolio rebalancing strategies.

## Architecture

The framework follows a four-tier architecture:

### 1. **strategy/strategy.py** - Strategy Base Class

Abstract base class `Strategy` that users extend to implement custom rebalancing logic.

**Key Features:**
- Abstract `rebalance()` method that calculates target positions
- `RebalanceFrequency` enum supporting DAILY, WEEKLY, BI_WEEKLY, MONTHLY, QUARTERLY, YEARLY
- Located in `lib/strategy/` for modular organization

**Creating Custom Strategies:**
```python
from strategy_v5.lib.strategy import Strategy
import numpy as np

class MyCustomStrategy(Strategy):
    def __init__(self):
        super().__init__("MyCustomStrategy")
    
    def rebalance(self, prices, capital, current_positions):
        """
        Calculate target positions (number of shares).
        
        Args:
            prices: np.ndarray of current prices for each asset
            capital: float, current portfolio value
            current_positions: np.ndarray, current shares held
        
        Returns:
            np.ndarray, target shares for each asset
        """
        n_assets = len(prices)
        capital_per_asset = capital / n_assets
        target_shares = capital_per_asset / prices
        return target_shares
```

**Available Strategy Implementations:**
- `strategy/strategy.py` - Base abstract class
- `strategy/buy_and_hold.py` - Buy and hold (no rebalancing)
- `strategy/weight.py` - Equal-weight rebalancing
- `strategy/momentum.py` - Momentum-based allocation
- `strategy/mvo_momentum.py` - Mean-variance optimization with momentum
- `strategy/mvo_mean_revert.py` - Mean-variance optimization with mean reversion

## Available Strategy Implementations

The framework includes several pre-built strategies in `strategy_v5/lib/strategy/`:

### 1. **BuyAndHoldRebalance** - `buy_and_hold.py`
Passive buy-and-hold strategy with no active rebalancing.
```python
from strategy_v5.lib.strategy import BuyAndHoldRebalance

strategy = BuyAndHoldRebalance()
portfolio = Portfolio(
    name='Buy & Hold',
    instruments=['AAPL', 'TSLA', 'NVDA'],
    initial_capital=100_000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY
)
```

### 2. **WeightRebalance** - `weight.py`
Equal-weight rebalancing - allocates capital equally across all assets.
```python
from strategy_v5.lib.strategy import WeightRebalance

strategy = WeightRebalance()
portfolio = Portfolio(
    name='Equal Weight',
    instruments=['AAPL', 'TSLA', 'NVDA', 'META'],
    initial_capital=100_000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1
)
```

### 3. **MVOMomentumRebalance** - `mvo_momentum.py`
Mean-Variance Optimization (MVO) with momentum signals.
- Allocates more capital to assets with positive momentum (recent uptrends)
- Uses lookback period mean returns as expected return signal
- Optimizes weights to maximize: `Return - (λ/2) × Variance`
```python
from strategy_v5.lib.strategy import MVOMomentumRebalance

strategy = MVOMomentumRebalance(
    lookback_days=20,       # Period for momentum calculation
    risk_aversion=1.0,      # λ coefficient (higher = more conservative)
    max_weight=0.3,         # Max 30% per asset (concentration limit)
    use_shrinkage=True      # Ledoit-Wolf covariance shrinkage
)
portfolio = Portfolio(
    name='MVO Momentum',
    instruments=['AAPL', 'TSLA', 'NVDA', 'META', 'BRK-B'],
    initial_capital=100_000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1
)
```

### 4. **MVOMeanRevertRebalance** - `mvo_mean_revert.py`
Mean-Variance Optimization with mean reversion signals.
- Allocates more capital to assets below their historical mean (undervalued)
- Expected to revert UP if trading below mean, DOWN if above mean
- Uses deviation from mean as the expected return signal
- Same MVO optimization framework as momentum strategy
```python
from strategy_v5.lib.strategy import MVOMeanRevertRebalance

strategy = MVOMeanRevertRebalance(
    lookback_days=20,        # Period for calculating mean    
    risk_aversion=1.0,       # Risk aversion in MVO
    max_weight=0.3,          # Max 30% per asset
    use_shrinkage=True       # Ledoit-Wolf shrinkage
)
portfolio = Portfolio(
    name='MVO Mean Reversion',
    instruments=['AAPL', 'TSLA', 'NVDA', 'META', 'BRK-B'],
    initial_capital=100_000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1
)
```

### 2. **portfolio.py** - Portfolio Management

Manages portfolio state, positions, and rebalancing triggers. Handles dynamic capital adjustment based on portfolio performance.

**Key Features:**
- Automatic rebalance frequency management with intelligent trigger logic
- Dynamic capital adjustment (portfolio gains/losses affect per-asset allocation)
- Complete historical tracking of positions, weights, and portfolio values
- Rebalance event logging with full event details
- Property-based access to historical data as DataFrames

**RebalanceFrequency Options:**
```python
class RebalanceFrequency(Enum):
    DAILY = 'D'
    WEEKLY = 'W'          # Rebalances on specified weekday (0=Monday, 6=Sunday)
    BI_WEEKLY = 'BW'      # Rebalances every 2 weeks on specified weekday
    MONTHLY = 'MS'        # Rebalances on specified day of month
    QUARTERLY = 'QS'
    YEARLY = 'YS'
```

**Key Properties:**
```python
portfolio.positions_history      # DataFrame: positions by date
portfolio.weights_history        # DataFrame: weights by date
portfolio.portfolio_values_history  # Series: portfolio values over time
portfolio.capital_history        # Series: capital over time
```

**Creating a Portfolio:**
```python
from strategy_v5.lib.portfolio import Portfolio
from strategy_v5.lib.strategy import RebalanceFrequency
from strategy_v5.lib.strategy.weight import WeightRebalance

strategy = WeightRebalance()
portfolio = Portfolio(
    instruments=['AAPL', 'NVDA', 'MSFT', 'TSLA'],
    initial_capital=100_000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1,           # 1st of each month
    name="Equal-Weight Portfolio"
)
```

**Rebalance Frequency Examples:**
```python
# Monthly on 15th
Portfolio(..., rebalance_freq=RebalanceFrequency.MONTHLY, rebalance_day=15)

# Weekly on Wednesday (2=Wednesday)
Portfolio(..., rebalance_freq=RebalanceFrequency.WEEKLY, rebalance_day=2)

# Bi-weekly on Monday (0=Monday)
Portfolio(..., rebalance_freq=RebalanceFrequency.BI_WEEKLY, rebalance_day=0)

# Daily
Portfolio(..., rebalance_freq=RebalanceFrequency.DAILY)
```

### 3. **executor.py** - Backtest Engine

Executes backtest over a date range. Handles data fetching from Yahoo Finance and daily portfolio updates.

**Key Features:**
- Automatic price data fetching via Yahoo Finance
- Daily portfolio rebalancing and value tracking
- Progress monitoring with verbose output
- Results stored in DataFrame format

**Running a Backtest:**
```python
from strategy_v5.lib.executor import Executor
import pandas as pd

executor = Executor(portfolio)
results = executor.run(
    start_date=pd.Timestamp('2024-01-01'),
    end_date=pd.Timestamp('2025-12-31'),
    verbose=True  # Show progress
)
# Returns DataFrame with dates and portfolio values
```

### 4. **evaluator.py** - Performance Analysis

Calculates metrics and compares multiple portfolios with comprehensive visualizations.

**Key Classes:**
- `PerformanceMetrics` - 12+ static methods for metric calculations
- `PortfolioEvaluator` - Compares multiple portfolios side-by-side

**Metrics Calculated:**
- Total Return
- Annualized Return  
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Average Daily Returns
- Max Daily Returns
- Min Daily Returns

**4-Subplot Visualization:**
1. Portfolio Value Over Time
2. Cumulative Returns %
3. 20-Day Rolling Annualized Volatility
4. Drawdown Over Time

**Using the Evaluator:**
```python
from strategy_v5.lib.evaluator import PortfolioEvaluator

evaluator = PortfolioEvaluator([portfolio1, portfolio2, portfolio3])

# Display metrics table
evaluator.print_comparison()

# Plot comparison (show_rebalance_markers=True to see rebalance events)
evaluator.plot_comparison(show_rebalance_markers=False)

# Access metrics DataFrame
metrics_df = evaluator.metrics_df
```
- Total Return
- Annualized Return
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio

```python
from evaluator import PortfolioEvaluator

evaluator = PortfolioEvaluator([portfolio1, portfolio2, portfolio3])
metrics = evaluator.calculate_all_metrics()
evaluator.print_comparison()
evaluator.plot_comparison()
```

## Complete Workflow Example

```python
from strategy_v5.lib.strategy import (
    WeightRebalance, 
    BuyAndHoldRebalance, 
    MVOMomentumRebalance, 
    MVOMeanRevertRebalance
)
from strategy_v5.lib.portfolio import Portfolio, RebalanceFrequency
from strategy_v5.lib.executor import Executor
from strategy_v5.lib.evaluator import PortfolioEvaluator
from datetime import datetime
from pandas.tseries.offsets import BDay
import pandas as pd

# Define instruments and dates
instruments = ['META', 'TSLA', 'NVDA', 'AAPL', 'BRK-B', 'SPY', 'QQQ']
initial_capital = 100_000
end_date = datetime.today() - BDay(1)
start_date = end_date - BDay(250)

# Create multiple portfolios with different strategies
portfolios = [
    Portfolio(
        name='EQUAL WEIGHT',
        instruments=instruments,
        initial_capital=initial_capital,
        strategy=WeightRebalance(),
        rebalance_freq=RebalanceFrequency.MONTHLY,
        rebalance_day=1
    ),
    
    Portfolio(
        name='BUY & HOLD',
        instruments=instruments,
        initial_capital=initial_capital,
        strategy=BuyAndHoldRebalance(),
        rebalance_freq=RebalanceFrequency.MONTHLY,
        rebalance_day=1
    ),
    
    Portfolio(
        name='MVO MOMENTUM',
        instruments=instruments,
        initial_capital=initial_capital,
        strategy=MVOMomentumRebalance(
            lookback_days=20, 
            max_weight=0.3, 
            risk_aversion=1, 
            use_shrinkage=True
        ),
        rebalance_freq=RebalanceFrequency.MONTHLY,
        rebalance_day=1
    ),
    
    Portfolio(
        name='MVO MEAN REVERSION',
        instruments=instruments,
        initial_capital=initial_capital,
        strategy=MVOMeanRevertRebalance(
            lookback_days=20, 
            max_weight=0.3, 
            risk_aversion=1, 
            use_shrinkage=True
        ),
        rebalance_freq=RebalanceFrequency.MONTHLY,
        rebalance_day=1
    ),
    
    Portfolio(
        name='SPY Benchmark',
        instruments=['SPY'],
        initial_capital=initial_capital,
        strategy=BuyAndHoldRebalance()
    )
]

# Execute backtests
for portfolio in portfolios:
    Executor(portfolio).run(start_date, end_date, verbose=False)

# Evaluate and compare
evaluator = PortfolioEvaluator(portfolios)
metrics = evaluator.calculate_all_metrics()

# Display results
evaluator.print_comparison()
evaluator.plot_comparison(figsize=(14, 10), show_rebalance_markers=False)

# Access detailed results
for portfolio in portfolios:
    print(f"\n{portfolio.name}:")
    print(f"  Final Value: ${portfolio.portfolio_values_history.iloc[-1]:,.0f}")
    print(f"  Total Return: {(portfolio.portfolio_values_history.iloc[-1] / portfolio.initial_capital - 1) * 100:.2f}%")
    print(f"  Number of Rebalances: {len(portfolio.history['rebalance_events'])}")
```

## Key Features

✅ **Modular Architecture** - Four-tier design (Strategy → Portfolio → Executor → Evaluator)
✅ **Easy Strategy Development** - Extend `Strategy` class and implement `rebalance()`
✅ **Multiple Rebalance Frequencies** - Daily, Weekly, Bi-weekly, Monthly, Quarterly, Yearly
✅ **Dynamic Capital Adjustment** - Portfolio P&L automatically affects per-asset allocation
✅ **Intelligent Rebalance Scheduling** - Weekday-based for weekly/bi-weekly, day-of-month for monthly
✅ **Comprehensive Metrics** - 9+ performance metrics including Sharpe, Calmar, volatility
✅ **Rich Visualizations** - 4-subplot comparison with optional rebalance event markers
✅ **Full Historical Tracking** - Complete audit trail of all positions, weights, and rebalance events
✅ **Property-Based Data Access** - Clean DataFrame/Series access to historical data
✅ **Yahoo Finance Integration** - Automatic data fetching from Yahoo Finance
✅ **Portfolio Comparison** - Side-by-side analysis of multiple strategies

## File Structure

```
strategy_v5/
├── lib/
│   ├── __init__.py                    # Package initialization
│   ├── strategy/
│   │   ├── __init__.py               # Strategy exports
│   │   ├── strategy.py               # Base Strategy class & RebalanceFrequency enum
│   │   ├── buy_and_hold.py           # Buy and hold (no rebalancing)
│   │   ├── weight.py                 # Equal-weight rebalancing
│   │   ├── mvo_momentum.py           # MVO with momentum signals
│   │   └── mvo_mean_revert.py        # MVO with mean reversion signals
│   ├── portfolio.py                  # Portfolio management & state tracking
│   ├── executor.py                   # Backtest execution engine
│   └── evaluator.py                  # Performance metrics & visualization
├── Portfolio Optimization - Framework.ipynb  # Complete backtest example
├── Portfolio Optimization - Mean Reversion.ipynb  # Mean reversion research
└── Readme.md                          # This file
```

## Key Design Patterns

### Dynamic Capital Adjustment
The framework recognizes that portfolio gains/losses should affect per-asset capital allocation:
```python
# Capital adjusts each day based on returns
new_capital = current_portfolio_value
# When rebalancing, allocation is based on current_portfolio_value, not initial_capital
```

### Flexible Rebalance Scheduling
```python
# Monthly on 15th day
Portfolio(..., rebalance_freq=RebalanceFrequency.MONTHLY, rebalance_day=15)

# Weekly every Wednesday
Portfolio(..., rebalance_freq=RebalanceFrequency.WEEKLY, rebalance_day=2)

# Bi-weekly, minimum 14 days between rebalances
Portfolio(..., rebalance_freq=RebalanceFrequency.BI_WEEKLY, rebalance_day=0)
```

### Visualization with Rebalance Context
The evaluator creates 4-subplot comparisons with optional rebalance event markers:
- Tap `show_rebalance_markers=True` to see when each strategy rebalanced
- Markers use the same color as the strategy line for easy correlation
- Three plots show rebalance events (Value, Returns, Volatility), one is a distribution

## Performance Metrics Reference

| Metric | Calculation | Interpretation |
|--------|-------------|-----------------|
| **Total Return** | (final - initial) / initial × 100 | Overall % gain/loss |
| **Annualized Return** | (final/initial)^(252/days) - 1 × 100 | Average annual % return |
| **Annualized Volatility** | Std(daily_returns) × √252 × 100 | Risk (how volatile) |
| **Sharpe Ratio** | (return - risk_free) / volatility | Risk-adjusted return |
| **Max Drawdown** | (peak - trough) / peak × 100 | Largest peak-to-trough loss |
| **Calmar Ratio** | annualized_return / max_drawdown | Return per unit of drawdown |

## Notes & Assumptions

- **Daily Close Prices** - All calculations use daily close prices from Yahoo Finance
- **Trading Days** - 252 trading days per year (standard convention)
- **Capital Adjustment** - Capital adjusts daily; rebalancing uses current portfolio value
- **Rebalance Timing** - Rebalances execute at market close using prices from that day
- **Transaction Costs** - Currently assumes frictionless execution (no commissions/slippage)
- **Risk-Free Rate** - Used for Sharpe ratio; fetched from `utils.data.get_latest_risk_free_rate()`

## Next Steps / Future Enhancements

- [ ] Transaction cost modeling (fixed/variable commissions, slippage)
- [ ] Tax-aware rebalancing
- [ ] Risk management features (drawdown limits, volatility-based position sizing)
- [ ] Advanced portfolio optimization (Black-Litterman, etc.)
- [ ] ML-based signal generation
- [ ] Multi-asset class support (bonds, commodities, crypto)

## Notes

- Prices are assumed to be daily close prices
- Capital is adjusted dynamically based on portfolio P&L
- Rebalancing happens on specified frequency with intelligent trigger logic
- All metrics are annualized using 252 trading days/year convention
