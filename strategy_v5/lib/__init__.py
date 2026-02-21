"""
Strategy V5 Backtesting Framework

A modular framework for building, backtesting, and evaluating trading strategies.

Components:
-----------
1. Strategy (strategy.py): Define rebalancing logic
2. Portfolio (portfolio.py): Manage positions and portfolio state
3. Executor (executor.py): Run backtest over date range
4. Evaluator (evaluator.py): Compare and analyze results

Example Usage:
--------------
from strategy_v5.strategy import MonthlyEvenRebalance, RebalanceFrequency
from strategy_v5.portfolio import Portfolio
from strategy_v5.executor import Executor
from strategy_v5.evaluator import PortfolioEvaluator

# Create strategy
strategy = MonthlyEvenRebalance()

# Create portfolio
portfolio = Portfolio(
    instruments=['AAPL', 'NVDA', 'MSFT'],
    initial_capital=100000,
    strategy=strategy,
    rebalance_freq=RebalanceFrequency.MONTHLY
)

# Run backtest
executor = Executor(portfolio)
results = executor.run(start_date, end_date)

# Evaluate
evaluator = PortfolioEvaluator([portfolio])
evaluator.print_comparison()
evaluator.plot_comparison()
"""

from strategy_v5.lib.strategy import *
from strategy_v5.lib.portfolio import Portfolio
from strategy_v5.lib.executor import Executor
from strategy_v5.lib.evaluator import PerformanceMetrics, PortfolioEvaluator