#!/usr/bin/env python3
"""
Main entry point for the Automated Portfolio Management System
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.portfolio_manager import PortfolioManager
from src.backtester import Backtester
import json


def main():
    """
    Main function to run the portfolio management system
    """
    print("Automated Portfolio Management System")
    print("=" * 50)
    
    # Initialize portfolio manager
    config_file = 'config/default_config.json'
    manager = PortfolioManager(config_file=config_file)
    
    print(f"Portfolio initialized with assets: {list(manager.portfolio_weights.keys())}")
    print(f"Initial weights: {manager.portfolio_weights}")
    
    # Run the portfolio management system
    manager.run()
    
    # Show current portfolio status
    performance = manager.get_portfolio_performance()
    print(f"\nPortfolio Performance:")
    print(f"Total Value: ${performance['total_value']:,.2f}")
    print(f"Current Weights: {performance['current_weights']}")
    if performance['last_rebalance_date']:
        print(f"Last Rebalance: {performance['last_rebalance_date']}")
    
    # Run backtest
    print("\n" + "="*50)
    print("Running Backtest...")
    
    # Load config for backtester
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    backtester = Backtester(config)
    
    # Run backtest with the same assets
    backtest_results = backtester.run_backtest(
        symbols=config['assets'],
        start_date=config['backtest_period']['start_date'],
        end_date=config['backtest_period']['end_date']
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Benchmark Total Return: {backtest_results['benchmark_total_return']:.2%}")
    print(f"Annualized Return: {backtest_results['annualized_return']:.2%}")
    print(f"Benchmark Annualized Return: {backtest_results['benchmark_annualized_return']:.2%}")
    print(f"Volatility: {backtest_results['volatility']:.2%}")
    print(f"Benchmark Volatility: {backtest_results['benchmark_volatility']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Benchmark Sharpe Ratio: {backtest_results['benchmark_sharpe']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Benchmark Max Drawdown: {backtest_results['benchmark_max_drawdown']:.2%}")
    print(f"Number of Rebalances: {backtest_results['num_rebalances']}")


if __name__ == "__main__":
    main()