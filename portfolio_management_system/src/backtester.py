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
import yfinance as yf


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
        self.portfolio_history = []
        self.benchmark_history = []
        self.rebalance_history = []
    
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
        
        # Initialize portfolio weights (equal weights or from config)
        if 'initial_weights' in self.config:
            initial_weights = self.config['initial_weights']
        else:
            n_assets = len(symbols)
            initial_weights = {symbol: 1.0/n_assets for symbol in symbols}
        
        # Initialize risk parameters
        max_weight_limit = self.config.get('max_weight_limit', 0.20)  # 20% max per asset
        critical_drop_threshold = self.config.get('critical_drop_threshold', -0.30)  # -30% threshold
        rebalance_threshold = self.config.get('rebalance_threshold', 0.05)  # 5% threshold
        min_rebalance_interval = self.config.get('min_rebalance_interval_days', 7)  # 7 days
        transaction_cost = self.config.get('transaction_cost', 0.001)  # 0.1% per transaction
        
        # Simulate portfolio performance over time
        portfolio_values = []
        portfolio_weights_history = []
        rebalance_dates = []
        
        # Starting value
        current_weights = initial_weights.copy()
        initial_value = 100000  # $100k starting capital
        current_value = initial_value
        last_rebalance_date = None
        
        # Iterate through each day
        for date_idx in range(len(returns_data)):
            current_date = returns_data.index[date_idx]
            
            # Calculate portfolio return for this day based on yesterday's weights
            day_return = 0
            for symbol in symbols:
                if symbol in returns_data.columns:
                    day_return += current_weights.get(symbol, 0) * returns_data.iloc[date_idx][symbol]
            
            # Update portfolio value
            current_value *= (1 + day_return)
            portfolio_values.append({
                'date': current_date,
                'value': current_value,
                'return': day_return
            })
            
            # Store portfolio history
            self.portfolio_history.append({
                'date': current_date,
                'value': current_value,
                'return': day_return,
                'weights': current_weights.copy()
            })
            
            # Calculate benchmark value (buy and hold without rebalancing)
            if date_idx == 0:
                benchmark_value = initial_value
            else:
                benchmark_return = returns_data.iloc[date_idx].mean()  # Equal weighted benchmark
                benchmark_value *= (1 + benchmark_return)
            
            self.benchmark_history.append({
                'date': current_date,
                'value': benchmark_value
            })
            
            # Check if rebalancing is needed (after the first day)
            if date_idx > 0:
                # Calculate current weights based on performance
                current_prices = price_data.loc[current_date]
                prev_prices = price_data.iloc[date_idx-1]
                
                # Calculate current market values and weights
                market_values = {}
                for symbol in symbols:
                    if symbol in current_prices and symbol in prev_prices:
                        # Calculate the value of each asset based on the portfolio value and returns
                        prev_weight = current_weights.get(symbol, initial_weights.get(symbol, 0))
                        price_return = (current_prices[symbol] / prev_prices[symbol]) - 1
                        new_value = current_value * prev_weight * (1 + price_return)
                        market_values[symbol] = new_value
                
                total_value = sum(market_values.values())
                if total_value > 0:
                    actual_weights = {s: v/total_value for s, v in market_values.items()}
                else:
                    actual_weights = current_weights.copy()
                
                # Check if any asset has dropped critically (more than -30% in a month)
                should_skip_rebalance = False
                if date_idx >= 22:  # At least a month of data
                    for symbol in symbols:
                        if symbol in price_data.columns:
                            # Calculate monthly return
                            month_ago_idx = date_idx - 22  # Approximate month
                            if month_ago_idx >= 0:
                                month_return = (current_prices[symbol] / price_data.iloc[month_ago_idx][symbol]) - 1
                                if month_return < critical_drop_threshold:
                                    print(f"Critical drop detected for {symbol}: {month_return:.2%}, skipping rebalance")
                                    should_skip_rebalance = True
                                    break
                
                # Check if rebalancing conditions are met
                days_since_last_rebalance = (
                    (current_date - last_rebalance_date).days 
                    if last_rebalance_date else float('inf')
                )
                
                weight_drift_exceeded = any(
                    abs(actual_weights.get(s, 0) - initial_weights.get(s, 0)) > rebalance_threshold 
                    for s in symbols
                )
                
                should_rebalance = (
                    not should_skip_rebalance and
                    days_since_last_rebalance >= min_rebalance_interval and
                    weight_drift_exceeded
                )
                
                if should_rebalance:
                    # Apply risk management constraints to target weights
                    adjusted_weights = self._apply_risk_constraints(actual_weights, initial_weights, max_weight_limit)
                    
                    # Calculate transaction costs
                    weight_changes = {s: abs(adjusted_weights.get(s, 0) - current_weights.get(s, 0)) 
                                      for s in symbols}
                    total_transaction_cost = sum(weight_changes.values()) * transaction_cost
                    
                    # Apply transaction costs to portfolio value
                    current_value *= (1 - total_transaction_cost)
                    
                    # Update weights after rebalancing
                    current_weights = adjusted_weights
                    last_rebalance_date = current_date
                    rebalance_dates.append(current_date)
                    
                    # Store rebalance event
                    self.rebalance_history.append({
                        'date': current_date,
                        'previous_weights': actual_weights.copy(),
                        'new_weights': current_weights.copy(),
                        'transaction_cost': total_transaction_cost
                    })
            
            portfolio_weights_history.append(current_weights.copy())
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        benchmark_df = pd.DataFrame(self.benchmark_history)
        benchmark_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        self.results = self._calculate_metrics(portfolio_df, benchmark_df, rebalance_dates)
        
        return self.results
    
    def _apply_risk_constraints(self, current_weights: Dict[str, float], target_weights: Dict[str, float], 
                               max_weight_limit: float) -> Dict[str, float]:
        """
        Apply risk management constraints to rebalancing weights
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights for rebalancing
            max_weight_limit: Maximum allowed weight for any single asset
            
        Returns:
            Adjusted weights that comply with risk constraints
        """
        adjusted_weights = target_weights.copy()
        
        # Ensure no asset exceeds maximum weight limit
        for symbol, weight in adjusted_weights.items():
            if weight > max_weight_limit:
                adjusted_weights[symbol] = max_weight_limit
        
        # Renormalize weights if they exceed 100%
        total_weight = sum(adjusted_weights.values())
        if total_weight > 1:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight
        
        # Ensure all weights are positive and sum to 1
        for symbol in adjusted_weights:
            if adjusted_weights[symbol] < 0:
                adjusted_weights[symbol] = 0
        
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for symbol in adjusted_weights:
                adjusted_weights[symbol] /= total_weight
        
        return adjusted_weights
    
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
        total_return = (portfolio_data['value'].iloc[-1] / portfolio_data['value'].iloc[0]) - 1
        benchmark_total_return = (benchmark_data['value'].iloc[-1] / benchmark_data['value'].iloc[0]) - 1
        
        # Annualized return (assuming 252 trading days per year)
        years = len(portfolio_data) / 252
        annualized_return = (portfolio_data['value'].iloc[-1] / portfolio_data['value'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        benchmark_annualized_return = (benchmark_data['value'].iloc[-1] / benchmark_data['value'].iloc[0]) ** (1/years) - 1 if years > 0 else 0
        
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
        
        # Alpha and Beta calculation
        # Covariance of portfolio and benchmark returns
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
        
        # Alpha (excess return over what CAPM predicts)
        alpha = annualized_return - (0 + beta * (benchmark_annualized_return - 0))  # Assuming 0% risk-free rate
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (annualized_return - benchmark_annualized_return) / tracking_error if tracking_error != 0 else 0
        
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
            'alpha': alpha,
            'beta': beta,
            'information_ratio': information_ratio,
            'num_rebalances': num_rebalances,
            'rebalance_dates': rebalance_dates
        }
    
    def plot_results(self, save_path: str = None):
        """
        Plot backtest results
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results or not self.portfolio_history:
            print("No backtest results to plot. Run backtest first.")
            return
        
        # Convert history to DataFrames for plotting
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df.set_index('date', inplace=True)
        
        benchmark_df = pd.DataFrame(self.benchmark_history)
        benchmark_df.set_index('date', inplace=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Backtesting Results', fontsize=16)
        
        # Plot 1: Portfolio vs Benchmark Performance
        axes[0, 0].plot(portfolio_df.index, portfolio_df['value'], label='Rebalanced Portfolio', linewidth=2)
        axes[0, 0].plot(benchmark_df.index, benchmark_df['value'], label='Buy & Hold Benchmark', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add rebalance markers if any occurred
        if self.rebalance_history:
            rebalance_dates = [r['date'] for r in self.rebalance_history]
            rebalance_values = [portfolio_df.loc[r['date']]['value'] for r in self.rebalance_history]
            axes[0, 0].scatter(rebalance_dates, rebalance_values, color='red', marker='^', s=50, label='Rebalance Points')
        
        # Plot 2: Drawdown chart
        portfolio_rolling_max = portfolio_df['value'].expanding().max()
        portfolio_drawdown = (portfolio_df['value'] - portfolio_rolling_max) / portfolio_rolling_max
        benchmark_rolling_max = benchmark_df['value'].expanding().max()
        benchmark_drawdown = (benchmark_df['value'] - benchmark_rolling_max) / benchmark_rolling_max
        
        axes[0, 1].fill_between(portfolio_df.index, portfolio_drawdown * 100, 0, alpha=0.3, color='blue', label='Portfolio Drawdown')
        axes[0, 1].fill_between(benchmark_df.index, benchmark_drawdown * 100, 0, alpha=0.3, color='orange', label='Benchmark Drawdown')
        axes[0, 1].set_title('Drawdown Comparison')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].invert_yaxis()  # Invert y-axis so drawdowns show as negative values below zero
        
        # Plot 3: Asset weights over time (if we have weight history)
        if len(self.portfolio_history) > 0 and 'weights' in self.portfolio_history[0]:
            # Get all unique symbols from weights
            all_symbols = set()
            for entry in self.portfolio_history:
                all_symbols.update(entry['weights'].keys())
            
            # Create a DataFrame for weights over time
            weight_data = {}
            for symbol in all_symbols:
                weight_data[symbol] = []
            
            for entry in self.portfolio_history:
                for symbol in all_symbols:
                    weight_data[symbol].append(entry['weights'].get(symbol, 0))
            
            weight_df = pd.DataFrame(weight_data, index=portfolio_df.index)
            
            # Plot weights over time
            for symbol in all_symbols:
                axes[1, 0].plot(weight_df.index, weight_df[symbol], label=symbol)
            axes[1, 0].set_title('Asset Weights Over Time')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Weight')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics comparison
        metrics_labels = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
        portfolio_metrics = [
            self.results['total_return'],
            self.results['annualized_return'],
            self.results['sharpe_ratio'],
            self.results['max_drawdown']
        ]
        benchmark_metrics = [
            self.results['benchmark_total_return'],
            self.results['benchmark_annualized_return'],
            self.results['benchmark_sharpe'],
            self.results['benchmark_max_drawdown']
        ]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, portfolio_metrics, width, label='Rebalanced Portfolio', alpha=0.8)
        axes[1, 1].bar(x + width/2, benchmark_metrics, width, label='Buy & Hold Benchmark', alpha=0.8)
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_labels, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self) -> str:
        """
        Generate a text report of backtesting results
        
        Returns:
            Formatted string with backtesting results
        """
        if not self.results:
            return "No backtest results available. Run backtest first."
        
        report = []
        report.append("="*60)
        report.append("PORTFOLIO BACKTESTING REPORT")
        report.append("="*60)
        
        report.append(f"Backtest Period: {self.portfolio_history[0]['date'].strftime('%Y-%m-%d')} to {self.portfolio_history[-1]['date'].strftime('%Y-%m-%d')}")
        report.append(f"Number of Trading Days: {len(self.portfolio_history)}")
        report.append(f"Number of Rebalances: {self.results['num_rebalances']}")
        report.append("")
        
        report.append("PERFORMANCE METRICS:")
        report.append("-" * 30)
        report.append(f"Rebalanced Portfolio Total Return:     {self.results['total_return']:.2%}")
        report.append(f"Buy & Hold Benchmark Total Return:     {self.results['benchmark_total_return']:.2%}")
        report.append(f"Excess Return:                         {self.results['total_return'] - self.results['benchmark_total_return']:.2%}")
        report.append("")
        report.append(f"Rebalanced Portfolio Annualized Return: {self.results['annualized_return']:.2%}")
        report.append(f"Buy & Hold Benchmark Annualized Return: {self.results['benchmark_annualized_return']:.2%}")
        report.append("")
        report.append(f"Rebalanced Portfolio Volatility:        {self.results['volatility']:.2%}")
        report.append(f"Buy & Hold Benchmark Volatility:        {self.results['benchmark_volatility']:.2%}")
        report.append("")
        report.append(f"Rebalanced Portfolio Sharpe Ratio:      {self.results['sharpe_ratio']:.3f}")
        report.append(f"Buy & Hold Benchmark Sharpe Ratio:      {self.results['benchmark_sharpe']:.3f}")
        report.append("")
        report.append(f"Rebalanced Portfolio Max Drawdown:      {self.results['max_drawdown']:.2%}")
        report.append(f"Buy & Hold Benchmark Max Drawdown:      {self.results['benchmark_max_drawdown']:.2%}")
        report.append("")
        report.append(f"Portfolio Alpha:                        {self.results['alpha']:.3f}")
        report.append(f"Portfolio Beta:                         {self.results['beta']:.3f}")
        report.append(f"Information Ratio:                      {self.results['information_ratio']:.3f}")
        report.append("")
        
        if self.rebalance_history:
            report.append("REBALANCING EVENTS:")
            report.append("-" * 30)
            for i, event in enumerate(self.rebalance_history[:5]):  # Show first 5 events
                report.append(f"Event {i+1}: {event['date'].strftime('%Y-%m-%d')}, Transaction Cost: {event['transaction_cost']:.3%}")
            if len(self.rebalance_history) > 5:
                report.append(f"... and {len(self.rebalance_history) - 5} more events")
        
        report.append("")
        report.append("="*60)
        
        return "\n".join(report)