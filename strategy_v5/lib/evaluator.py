"""
Evaluator Module

Calculates performance metrics and compares multiple backtested portfolios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy_v5.lib.portfolio import Portfolio
from utils.data import get_latest_risk_free_rate
from IPython.display import display

class PerformanceMetrics:
    """
    Static class for calculating performance metrics.
    
    Provides methods to compute:
    - Returns (daily, total, annualized)
    - Sharpe Ratio
    - Maximum Drawdown
    - Calmar Ratio
    """
    
    @staticmethod
    def calculate_returns(values_series: pd.Series) -> pd.Series:
        """
        Calculate daily returns.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        Returns:
        --------
        pd.Series
            Daily returns (as decimals)
        """
        return values_series.pct_change()
    
    @staticmethod
    def total_return(initial_value: float, final_value: float) -> float:
        """
        Calculate total return percentage.
        
        Parameters:
        -----------
        initial_value : float
            Starting portfolio value
        
        final_value : float
            Ending portfolio value
        
        Returns:
        --------
        float
            Total return in percentage
        """
        return ((final_value / initial_value) - 1) * 100
    
    @staticmethod
    def annualized_return(values_series: pd.Series, trading_days: int = 252) -> float:
        """
        Calculate annualized return.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        trading_days : int
            Trading days per year (default 252)
        
        Returns:
        --------
        float
            Annualized return in percentage
        """
        n_days = len(values_series) - 1
        total_ret = (values_series.iloc[-1] / values_series.iloc[0]) - 1
        annualized = (1 + total_ret) ** (trading_days / n_days) - 1
        return annualized * 100
    
    @staticmethod
    def sharpe_ratio(values_series: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252) -> float:
        """
        Calculate Sharpe Ratio.
        
        Measures excess return per unit of volatility.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        risk_free_rate : float
            Annual risk-free rate (default 2%)
        
        trading_days : int
            Trading days per year (default 252)
        
        Returns:
        --------
        float
            Sharpe Ratio
        """
        daily_returns = values_series.pct_change().dropna()
        excess_return = daily_returns.mean() * trading_days - risk_free_rate
        volatility = daily_returns.std() * np.sqrt(trading_days)
        
        if volatility == 0:
            return 0
        return excess_return / volatility
    
    @staticmethod
    def max_drawdown(values_series: pd.Series) -> float:
        """
        Calculate maximum drawdown percentage.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        Returns:
        --------
        float
            Maximum drawdown in percentage (negative value)
        """
        cummax = values_series.cummax()
        drawdown = (values_series - cummax) / cummax
        return drawdown.min() * 100
    
    @staticmethod
    def calmar_ratio(values_series: pd.Series, trading_days: int = 252) -> float:
        """
        Calculate Calmar Ratio.
        
        Ratio of annualized return to maximum drawdown.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        trading_days : int
            Trading days per year (default 252)
        
        Returns:
        --------
        float
            Calmar Ratio
        """
        ann_return = PerformanceMetrics.annualized_return(values_series, trading_days)
        max_dd = abs(PerformanceMetrics.max_drawdown(values_series))
        
        if max_dd == 0:
            return 0
        return ann_return / max_dd
    
    @staticmethod
    def avg_daily_returns(values_series: pd.Series) -> float:
        """
        Calculate average daily return.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        Returns:
        --------
        float
            Average daily return in percentage
        """
        daily_ret = values_series.pct_change().dropna()
        return daily_ret.mean() * 100
    
    @staticmethod
    def max_daily_returns(values_series: pd.Series) -> float:
        """
        Calculate maximum daily return.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        Returns:
        --------
        float
            Maximum daily return in percentage
        """
        daily_ret = values_series.pct_change().dropna()
        return daily_ret.max() * 100
    
    @staticmethod
    def min_daily_returns(values_series: pd.Series) -> float:
        """
        Calculate minimum daily return.
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        Returns:
        --------
        float
            Minimum daily return in percentage
        """
        daily_ret = values_series.pct_change().dropna()
        return daily_ret.min() * 100
    
    @staticmethod
    def annualized_volatility(values_series: pd.Series, trading_days: int = 252) -> float:
        """
        Calculate annualized volatility (standard deviation of returns).
        
        Parameters:
        -----------
        values_series : pd.Series
            Portfolio values over time
        
        trading_days : int
            Trading days per year (default 252)
        
        Returns:
        --------
        float
            Annualized volatility in percentage
        """
        daily_ret = values_series.pct_change().dropna()
        daily_std = daily_ret.std()
        annualized_vol = daily_std * np.sqrt(trading_days)
        return annualized_vol * 100


class PortfolioEvaluator:
    """
    Evaluates and compares multiple backtested portfolios.
    
    Responsibilities:
    - Calculate performance metrics for each portfolio
    - Generate comparison tables
    - Create comparative visualizations
    """
    
    def __init__(self, portfolios: list[Portfolio]|Portfolio):
        """
        Initialize evaluator with list of portfolios.
        
        Parameters:
        -----------
        portfolios : list
            List of Portfolio objects with completed backtests
        """

        if isinstance(portfolios, Portfolio):
            self.portfolios = [portfolios]
        else:
            self.portfolios = portfolios  
                      
        self.metrics_df = None
    
    def calculate_all_metrics(self) -> pd.DataFrame:
        """
        Calculate performance metrics for all portfolios.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate for Sharpe/Calmar calculations (default 2%)
        
        Returns:
        --------
        pd.DataFrame
            Performance metrics for each portfolio
        """

        risk_free_rate = (1+get_latest_risk_free_rate())**252-1
        metrics_list = []
        
        for portfolio in self.portfolios:                        
            values = portfolio.capital_history
            initial = portfolio.initial_capital
            final = values.iloc[-1]            
            
            metrics = {
                'Strategy': portfolio.name,
                'Initial Capital': f"${initial:,.0f}",
                'Final Value': f"${final:,.0f}",
                'Total Return %': f"{PerformanceMetrics.total_return(initial, final):.2f}%",
                'Ann. Return %': f"{PerformanceMetrics.annualized_return(values):.2f}%",
                'Ann. Volatility %': f"{PerformanceMetrics.annualized_volatility(values):.2f}%",
                'Return %': f"{PerformanceMetrics.avg_daily_returns(values):.4f}%",
                'Max Return %': f"{PerformanceMetrics.max_daily_returns(values):.2f}%",
                'Min Return %': f"{PerformanceMetrics.min_daily_returns(values):.2f}%",                
                'Sharpe Ratio': f"{PerformanceMetrics.sharpe_ratio(values, risk_free_rate):.3f}",
                'Max Drawdown %': f"{PerformanceMetrics.max_drawdown(values):.2f}%",
                'Calmar Ratio': f"{PerformanceMetrics.calmar_ratio(values):.3f}",
                'Num Rebalances': len(portfolio.history['rebalance_events']),
                'Avg Turnover $': f"${np.mean([event['turnover_mv'] for event in portfolio.history['rebalance_events']]):,.0f}",                
                'Total Turnover $': f"${np.sum([event['turnover_mv'] for event in portfolio.history['rebalance_events']]):,.0f}",                
                'Trading Days': len(values)
            }
            metrics_list.append(metrics)
                
        self.metrics_df = pd.DataFrame(metrics_list)        
        self.metrics_df.columns = pd.MultiIndex.from_tuples([            
            ('', 'Strategy'),
            ('Performance', 'Initial Capital'),
            ('Performance', 'Final Value'),
            ('Performance', 'Total Return %'),
            ('Performance', 'Ann. Return %'),
            ('Performance', 'Ann. Volatility %'),
            ('Performance', 'Return %'),
            ('Performance', 'Max Return %'),
            ('Performance', 'Min Return %'),
            ('Risk-Adjusted', 'Sharpe Ratio'),
            ('Risk-Adjusted', 'Max Drawdown %'),
            ('Risk-Adjusted', 'Calmar Ratio'),
            ('Rebalance', 'Num Rebalances'),
            ('Rebalance', 'Avg Turnover $'),
            ('Rebalance', 'Total Turnover $'),
            ('Rebalance', 'Trading Days'),
        ])

        return self.metrics_df
    
    def print_comparison(self, format=True):
        """
        Print comparison table of all portfolios with styling.
        Displays a formatted HTML table with grouped columns.
        """
        if self.metrics_df is None:
            self.calculate_all_metrics()
        
        # Apply styling for display
        if format:
            style = self.metrics_df.style.set_caption("Portfolio Performance Comparison").set_table_styles([
                {'selector': 'caption', 'props': [('font-size', '16px'), ('font-weight', 'bold'), ('color', '#e0e0e0'), ('caption-side', 'top'), ('padding', '10px')]},
                {'selector': 'th', 'props': [('background-color', '#2d2d2d'), ('color', '#e0e0e0'), ('font-size', '12px'), ('border', '1px solid #444'), ('text-align', 'center'), ('vertical-align', 'middle')]},
                {'selector': 'td', 'props': [('padding', '8px'), ('text-align', 'center'), ('font-size', '12px'), ('border', '1px solid #444'), ('color', '#e0e0e0')]},
                {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#1e1e1e')]},
            ])
        else:
            style = self.metrics_df.style
        
        display(style)        
    
    def _mark_rebalance_events(self, ax: plt.Axes, portfolio: Portfolio, data_series: pd.Series, color: str) -> None:
        """
        Mark rebalance events on a plot with scatter markers.
        
        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axis to draw on
        
        portfolio : Portfolio
            Portfolio object containing rebalance events
        
        data_series : pd.Series
            Data series being plotted (portfolio values, returns, volatility, etc.)
        
        color : str
            Color for the scatter markers (should match the line color)
        """
        rebal_dates = []
        rebal_values = []
        
        for rebal_event in portfolio.history['rebalance_events']:
            rebal_date = rebal_event['date']
            if rebal_date in data_series.index:
                rebal_dates.append(rebal_date)
                rebal_values.append(data_series[rebal_date])
        
        if rebal_dates:
            ax.scatter(rebal_dates, rebal_values, marker='o', s=25, color=color, linewidths=1.5, zorder=5)
    
    def plot_comparison(self, figsize: tuple = (14, 8), show_rebalance_markers: bool = False):
        """
        Plot comparison of multiple portfolios.
        
        Creates 4 subplots:
        1. Portfolio values over time
        2. Cumulative returns %
        3. 20-day rolling annualized volatility
        4. Drawdown over time
        
        Rebalance events are marked with 'x' markers using the same color as the strategy line (if show_rebalance_markers=True).
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        show_rebalance_markers : bool
            Whether to show rebalance event markers on plots. Default is False.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Get default matplotlib colors (blue, orange, green, red, purple, etc.)
        # Cycle colors if we have more portfolios than default colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']        
        colors = [colors[i % len(colors)] for i in range(len(self.portfolios))]
        
        # Plot 1: Portfolio values over time
        ax = axes[0]
        for idx, portfolio in enumerate(self.portfolios):            
            ax.plot(portfolio.portfolio_values_history.index, portfolio.portfolio_values_history.values, label=portfolio.name, linewidth=2, color=colors[idx])
            
            if show_rebalance_markers:
                self._mark_rebalance_events(ax, portfolio, portfolio.portfolio_values_history, colors[idx])

        ax.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Plot 2: Cumulative Log Returns
        ax = axes[1]
        for idx, portfolio in enumerate(self.portfolios):                        
            cum_ret = np.log(portfolio.portfolio_values_history.pct_change().add(1)).cumsum() * 100
            ax.plot(cum_ret.index, cum_ret.values, label=portfolio.name, linewidth=2, color=colors[idx])
            
            if show_rebalance_markers:
                self._mark_rebalance_events(ax, portfolio, cum_ret, colors[idx])

        ax.set_title('Cumulative Log Returns %', fontsize=12, fontweight='bold')
        ax.set_ylabel('Return %', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: 20-day rolling annualized volatility
        ax = axes[2]
        for idx, portfolio in enumerate(self.portfolios):
            daily_ret = portfolio.portfolio_values_history.pct_change().dropna()
            rolling_vol = daily_ret.rolling(window=20, min_periods=1).std() * np.sqrt(252) * 100  # Annualized vol in %
            ax.plot(rolling_vol.index, rolling_vol.values, label=portfolio.name, linewidth=2, color=colors[idx])
            
            if show_rebalance_markers:
                self._mark_rebalance_events(ax, portfolio, rolling_vol, colors[idx])
        
        ax.set_title('20-Day Rolling Annualized Volatility', fontsize=12, fontweight='bold')
        ax.set_ylabel('Volatility (%)', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)        
        
        # Plot 4: Drawdown
        ax = axes[3]
        for idx, portfolio in enumerate(self.portfolios):
            values = portfolio.portfolio_values_history
            cummax = values.cummax()
            drawdown = ((values - cummax) / cummax) * 100
            ax.plot(drawdown.index, drawdown.values, label=portfolio.name, linewidth=2, color=colors[idx])
            
            if show_rebalance_markers:
                self._mark_rebalance_events(ax, portfolio, drawdown, colors[idx])
        
        ax.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown %', fontsize=11)
        ax.set_xlabel('Date', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_weights_history(self, portfolio_idx: int = 0, portfolio_name: str = None):
        """
        Plot historical weights distribution for a portfolio using Plotly.
        
        Shows how the portfolio allocation to each asset evolves over time.
        Allows interactive selection/deselection of assets via legend clicks.
        
        Parameters:
        -----------
        portfolio_idx : int, default=0
            Index of portfolio in self.portfolios list (if portfolio_name not provided)
        
        portfolio_name : str, optional
            Name of portfolio to plot. If provided, overrides portfolio_idx.
            If neither exists, defaults to first portfolio.
        Returns:
        --------
        None (displays Plotly chart)
        """
        # Select portfolio by name or index
        portfolio = None
        
        if portfolio_name:
            for p in self.portfolios:
                if p.name == portfolio_name:
                    portfolio = p
                    break
            if portfolio is None:
                print(f"Warning: Portfolio '{portfolio_name}' not found. Using first portfolio.")
                portfolio = self.portfolios[0]
        else:
            if portfolio_idx >= len(self.portfolios):
                print(f"Warning: Portfolio index {portfolio_idx} out of range. Using first portfolio.")
                portfolio_idx = 0
            portfolio = self.portfolios[portfolio_idx]
        
        # Get weights history
        weights_df = portfolio.weights_history
        
        if weights_df is None or weights_df.empty:
            print(f"Error: No weights history available for portfolio '{portfolio.name}'")
            return
        
        # Create figure(s) based on chart_type                
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Portfolio Weights (Lines)', 'Portfolio Weights (Stacked)'),
            specs=[[{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Add line chart traces
        for asset in weights_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=weights_df.index,
                    y=weights_df[asset],
                    mode='lines',
                    name=asset,
                    hovertemplate='%{fullData.name}: %{y:.2%}<extra></extra>',
                    line=dict(width=2),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add stacked area traces
        for asset in weights_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=weights_df.index,
                    y=weights_df[asset],
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    fillcolor=None,
                    hovertemplate='%{fullData.name}: %{y:.2%}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text=f'Portfolio Weights History - {portfolio.name}',
            height=600,
            width=1400,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='left',
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='gray',
                borderwidth=1
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text='Date', row=1, col=1)
        fig.update_xaxes(title_text='Date', row=1, col=2)
        fig.update_yaxes(title_text='Weight (%)', tickformat='.0%', range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text='Weight (%)', tickformat='.0%', range=[0, 1], row=1, col=2)                 
        fig.show()
