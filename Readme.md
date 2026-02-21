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
from strategy_v5.lib.strategy.weight import WeightRebalance
from strategy_v5.lib.strategy import RebalanceFrequency
from strategy_v5.lib.portfolio import Portfolio
from strategy_v5.lib.executor import Executor
from strategy_v5.lib.evaluator import PortfolioEvaluator
from strategy_v5.lib.strategy.momentum import MomentumRebalance
import pandas as pd

# 1. Define strategies
strategy_equal = WeightRebalance()
strategy_momentum = MomentumRebalance()

# 2. Create portfolios
portfolio_equal = Portfolio(
    instruments=['AAPL', 'NVDA', 'MSFT', 'TSLA', 'META'],
    initial_capital=100_000,
    strategy=strategy_equal,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1,
    name="Equal Weight"
)

portfolio_momentum = Portfolio(
    instruments=['AAPL', 'NVDA', 'MSFT', 'TSLA', 'META'],
    initial_capital=100_000,
    strategy=strategy_momentum,
    rebalance_freq=RebalanceFrequency.MONTHLY,
    rebalance_day=1,
    name="Momentum"
)

# 3. Run backtests
executor_equal = Executor(portfolio_equal)
executor_equal.run(
    start_date=pd.Timestamp('2024-01-01'),
    end_date=pd.Timestamp('2025-12-31'),
    verbose=True
)

executor_momentum = Executor(portfolio_momentum)
executor_momentum.run(
    start_date=pd.Timestamp('2024-01-01'),
    end_date=pd.Timestamp('2025-12-31'),
    verbose=True
)

# 4. Evaluate and compare
evaluator = PortfolioEvaluator([portfolio_equal, portfolio_momentum])
evaluator.print_comparison()
evaluator.plot_comparison(show_rebalance_markers=True)

# 5. Access detailed results
print(portfolio_equal.portfolio_values_history)  # Portfolio values over time
print(portfolio_equal.positions_history)         # Share holdings over time
print(portfolio_equal.weights_history)           # Portfolio weights over time
print(portfolio_equal.history['rebalance_events'])  # All rebalance events
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
│   │   ├── weight.py                 # Equal-weight rebalancing
│   │   ├── momentum.py               # Momentum-based strategy
│   │   ├── buy_and_hold.py           # Buy and hold strategy
│   │   └── mvo_momentum.py           # MVO with momentum
│   ├── portfolio.py                  # Portfolio management & state tracking
│   ├── executor.py                   # Backtest execution engine
│   └── evaluator.py                  # Performance metrics & visualization
├── Portfolio Optimization - Mean Reversion.ipynb  # Example notebook
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
