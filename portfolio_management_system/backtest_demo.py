#!/usr/bin/env python3
"""
Demonstration script for the backtesting module
Simulates portfolio management strategy over historical market data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.backtester import Backtester
from datetime import datetime, timedelta
import json


def main():
    print("Portfolio Management System - Backtesting Demo")
    print("="*60)
    
    # Define configuration for the backtest
    config = {
        'rebalance_threshold': 0.05,      # Rebalance when weight deviates by more than 5%
        'min_rebalance_interval_days': 7, # Minimum 7 days between rebalances
        'max_weight_limit': 0.25,         # Maximum 25% in any single asset
        'critical_drop_threshold': -0.30, # Don't invest in assets that dropped more than 30% in a month
        'transaction_cost': 0.001,        # 0.1% transaction cost
        'initial_weights': {              # Initial weights for the portfolio
            'AAPL': 0.2,
            'MSFT': 0.2,
            'GOOGL': 0.2,
            'AMZN': 0.2,
            'TSLA': 0.2
        }
    }
    
    # Create backtester instance
    backtester = Backtester(config)
    
    # Define the portfolio symbols and time period
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    print(f"Testing portfolio: {', '.join(symbols)}")
    print(f"Backtest period: {start_date} to {end_date}")
    print(f"Initial allocation: Equal weights (20% each)")
    print()
    
    try:
        # Run the backtest
        print("Running backtest...")
        results = backtester.run_backtest(symbols, start_date, end_date)
        
        # Display results
        print("\nBacktest Results:")
        print("-" * 40)
        print(f"Total Return (Rebalanced):      {results['total_return']:.2%}")
        print(f"Total Return (Benchmark):       {results['benchmark_total_return']:.2%}")
        print(f"Excess Return:                  {results['total_return'] - results['benchmark_total_return']:.2%}")
        print()
        print(f"Annualized Return (Rebalanced): {results['annualized_return']:.2%}")
        print(f"Annualized Return (Benchmark):  {results['benchmark_annualized_return']:.2%}")
        print()
        print(f"Volatility (Rebalanced):        {results['volatility']:.2%}")
        print(f"Volatility (Benchmark):         {results['benchmark_volatility']:.2%}")
        print()
        print(f"Sharpe Ratio (Rebalanced):      {results['sharpe_ratio']:.3f}")
        print(f"Sharpe Ratio (Benchmark):       {results['benchmark_sharpe']:.3f}")
        print()
        print(f"Max Drawdown (Rebalanced):      {results['max_drawdown']:.2%}")
        print(f"Max Drawdown (Benchmark):       {results['benchmark_max_drawdown']:.2%}")
        print()
        print(f"Alpha:                          {results['alpha']:.3f}")
        print(f"Beta:                           {results['beta']:.3f}")
        print(f"Information Ratio:              {results['information_ratio']:.3f}")
        print()
        print(f"Number of Rebalances:           {results['num_rebalances']}")
        
        # Generate detailed report
        print("\nDetailed Report:")
        print("=" * 60)
        report = backtester.generate_report()
        print(report)
        
        # Plot results (uncomment if you want to visualize)
        # backtester.plot_results()
        
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()


def simple_backtest_example():
    """
    A simplified example showing basic backtesting functionality
    """
    print("\nSimple Backtesting Example")
    print("="*40)
    
    # Simple equal-weight configuration
    config = {
        'rebalance_threshold': 0.05,
        'min_rebalance_interval_days': 7,
        'max_weight_limit': 0.30,
        'transaction_cost': 0.001
    }
    
    backtester = Backtester(config)
    
    # Test with a smaller portfolio for faster execution
    symbols = ['SPY', 'QQQ', 'TLT']  # ETFs for diversification
    start_date = '2021-01-01'
    end_date = '2022-01-01'
    
    print(f"Testing: {', '.join(symbols)} from {start_date} to {end_date}")
    
    try:
        results = backtester.run_backtest(symbols, start_date, end_date)
        
        print(f"\nResults:")
        print(f"Rebalanced Portfolio Return: {results['total_return']:.2%}")
        print(f"Benchmark Return:            {results['benchmark_total_return']:.2%}")
        print(f"Excess Return:               {results['total_return'] - results['benchmark_total_return']:.2%}")
        print(f"Number of Rebalances:        {results['num_rebalances']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
    simple_backtest_example()