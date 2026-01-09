"""
Backtesting module for portfolio management system
Tests the rebalancing strategy on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class Backtester:
    """
    Class for backtesting portfolio strategies on historical data
    """
    
    def __init__(self, config: dict):
        """
        Initialize backtester with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str) -> Dict:
        """
        Run backtest of the portfolio strategy
        
        Args:
            symbols: List of stock symbols to include in portfolio
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        print(f"Running backtest from {start_date} to {end_date}")
        
        # Load historical data
        from .data_loader import DataLoader
        data_loader = DataLoader()
        price_data = data_loader.get_historical_data(symbols, start_date, end_date)
        
        if price_data.empty:
            raise ValueError("No historical data available for backtesting")
        
        # Calculate daily returns
        returns_data = price_data.pct_change().dropna()
        
        # Initialize portfolio weights (equal weights)
        n_assets = len(symbols)
        initial_weights = {symbol: 1.0/n_assets for symbol in symbols}
        
        # Simulate portfolio performance over time
        portfolio_values = []
        portfolio_weights_history = []
        rebalance_dates = []
        
        # Starting value
        current_weights = initial_weights.copy()
        initial_value = 100000  # $100k starting capital
        current_value = initial_value
        
        # Iterate through each day
        for date_idx in range(len(returns_data)):
            current_date = returns_data.index[date_idx]
            
            # Calculate portfolio return for this day based on yesterday's weights
            if date_idx == 0:
                # Use initial weights for first day
                day_return = sum(
                    current_weights[symbol] * returns_data.iloc[date_idx][symbol] 
                    for symbol in symbols if symbol in returns_data.columns
                )
            else:
                # Calculate return based on previous day's weights
                day_return = sum(
                    current_weights[symbol] * returns_data.iloc[date_idx][symbol] 
                    for symbol in symbols if symbol in returns_data.columns
                )
            
            # Update portfolio value
            current_value *= (1 + day_return)
            portfolio_values.append({
                'date': current_date,
                'value': current_value,
                'return': day_return
            })
            
            # Check if rebalancing is needed (based on threshold)
            # Calculate current weights based on performance drift
            if date_idx > 0:  # Skip first day
                current_prices = price_data.loc[current_date]
                
                # Calculate market values based on current prices and previous weights
                market_values = {}
                for symbol in symbols:
                    if symbol in current_prices:
                        # Get the proportion of portfolio value in this asset
                        prev_weight = current_weights.get(symbol, 0)
                        asset_value = current_value * prev_weight * (current_prices[symbol] / price_data.iloc[date_idx-1][symbol])
                        market_values[symbol] = asset_value
                
                total_value = sum(market_values.values())
                if total_value > 0:
                    actual_weights = {s: v/total_value for s, v in market_values.items()}
                else:
                    actual_weights = current_weights
                    
                # Check if any asset has drifted beyond threshold
                rebalance_threshold = self.config.get('rebalance_threshold', 0.05)  # 5%
                days_since_last_rebalance = (
                    (current_date - rebalance_dates[-1]).days 
                    if rebalance_dates else float('inf')
                )
                min_rebalance_interval = self.config.get('min_rebalance_interval_days', 7)
                
                should_rebalance = (
                    days_since_last_rebalance >= min_rebalance_interval and
                    any(abs(actual_weights.get(s, 0) - initial_weights.get(s, 0)) > rebalance_threshold 
                        for s in symbols)
                )
                
                if should_rebalance:
                    # Rebalance to target weights
                    current_weights = initial_weights.copy()
                    rebalance_dates.append(current_date)
            
            portfolio_weights_history.append(current_weights.copy())
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate benchmark (buy and hold without rebalancing)
        benchmark_returns = returns_data.mean(axis=1)  # Equal weighted buy and hold
        benchmark_values = [initial_value]
        for ret in benchmark_returns[1:]:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        
        benchmark_df = pd.DataFrame({
            'value': benchmark_values
        }, index=returns_data.index)
        
        # Calculate performance metrics
        self.results = self._calculate_metrics(portfolio_df, benchmark_df, rebalance_dates)
        
        return self.results
    
    def _calculate_metrics(self, portfolio_data: pd.DataFrame, benchmark_data: pd.DataFrame, 
                          rebalance_dates: List) -> Dict:
        """
        Calculate performance metrics from backtest results
        
        Args:
            portfolio_data: DataFrame with portfolio value history
            benchmark_data: DataFrame with benchmark value history
            rebalance_dates: List of dates when rebalancing occurred
            
        Returns:
            Dictionary with performance metrics
        """
        # Portfolio returns
        portfolio_returns = portfolio_data['value'].pct_change().dropna()
        benchmark_returns = benchmark_data['value'].pct_change().dropna()
        
        # Total return
        total_return = (portfolio_data['value'][-1] / portfolio_data['value'][0]) - 1
        benchmark_total_return = (benchmark_data['value'][-1] / benchmark_data['value'][0]) - 1
        
        # Annualized return (assuming 252 trading days per year)
        years = len(portfolio_data) / 252
        annualized_return = (portfolio_data['value'][-1] / portfolio_data['value'][0]) ** (1/years) - 1 if years > 0 else 0
        benchmark_annualized_return = (benchmark_data['value'][-1] / benchmark_data['value'][0]) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        
        # Maximum drawdown
        rolling_max = portfolio_data['value'].expanding().max()
        daily_drawdown = portfolio_data['value'] / rolling_max - 1
        max_drawdown = daily_drawdown.min()
        
        benchmark_rolling_max = benchmark_data['value'].expanding().max()
        benchmark_daily_drawdown = benchmark_data['value'] / benchmark_rolling_max - 1
        benchmark_max_drawdown = benchmark_daily_drawdown.min()
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        benchmark_sharpe = benchmark_annualized_return / benchmark_volatility if benchmark_volatility != 0 else 0
        
        # Number of rebalancing events
        num_rebalances = len(rebalance_dates)
        
        return {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'volatility': volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'num_rebalances': num_rebalances,
            'rebalance_dates': rebalance_dates
        }
    
    def plot_results(self):
        """
        Plot backtest results
        """
        if not self.results:
            print("No backtest results to plot. Run backtest first.")
            return
        
        # We would need to store portfolio and benchmark data during backtest to plot
        print("Plotting functionality would be implemented here.")
        # Example of what would be plotted:
        # - Portfolio value vs benchmark over time
        # - Drawdown chart
        # - Performance attribution by asset