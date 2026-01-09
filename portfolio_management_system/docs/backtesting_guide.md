# Backtesting Guide

This document explains the backtesting module of the Automated Portfolio Management System.

## Overview

The backtesting module allows you to test your portfolio management strategy against historical market data. It simulates how your portfolio would have performed over a specific time period, comparing the results with a benchmark strategy (typically buy-and-hold).

## Key Features

### 1. Historical Simulation
- Tests portfolio rebalancing strategies against real historical price data
- Supports any combination of stocks/ETFs with historical data available
- Simulates daily portfolio performance based on changing asset prices

### 2. Rebalancing Logic
- Implements threshold-based rebalancing (default: 5% deviation from target weights)
- Enforces minimum rebalancing intervals (default: 7 days)
- Applies risk management constraints during rebalancing

### 3. Risk Management
- Critical drop filter: Skips rebalancing when assets drop more than 30% in a month
- Maximum weight limits: Prevents concentration in single assets (default: 20%)
- Transaction cost consideration: Accounts for trading fees in performance calculations

### 4. Performance Metrics
The backtester calculates key performance indicators:

- **Total Return**: Overall portfolio growth over the period
- **Annualized Return**: Average yearly return
- **Volatility**: Standard deviation of returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Alpha**: Excess return relative to benchmark
- **Beta**: Sensitivity to market movements
- **Information Ratio**: Excess return per unit of tracking error

## Usage Examples

### Basic Usage
```python
from src.backtester import Backtester

# Configure your strategy
config = {
    'rebalance_threshold': 0.05,      # 5% deviation triggers rebalance
    'min_rebalance_interval_days': 7, # Wait at least 7 days between rebalances
    'max_weight_limit': 0.25,         # Max 25% in any single asset
    'critical_drop_threshold': -0.30, # Skip rebalancing if asset drops 30% in month
    'transaction_cost': 0.001         # 0.1% transaction cost
}

# Create backtester instance
backtester = Backtester(config)

# Run backtest
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
start_date = '2020-01-01'
end_date = '2023-01-01'

results = backtester.run_backtest(symbols, start_date, end_date)

# View results
print(f"Portfolio Return: {results['total_return']:.2%}")
print(f"Benchmark Return: {results['benchmark_total_return']:.2%}")
print(f"Number of Rebalances: {results['num_rebalances']}")
```

### Generate Detailed Reports
```python
# Generate a comprehensive text report
report = backtester.generate_report()
print(report)

# Create visualizations
backtester.plot_results()  # Shows interactive plots
backtester.plot_results(save_path='backtest_results.png')  # Saves to file
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rebalance_threshold` | Percentage deviation that triggers rebalancing | 0.05 (5%) |
| `min_rebalance_interval_days` | Minimum days between rebalancing events | 7 |
| `max_weight_limit` | Maximum percentage allowed in single asset | 0.20 (20%) |
| `critical_drop_threshold` | Monthly drop threshold to skip rebalancing | -0.30 (-30%) |
| `transaction_cost` | Cost per transaction (as decimal) | 0.001 (0.1%) |
| `initial_weights` | Starting weights for portfolio assets | Equal weights |

## How Rebalancing Works in Backtest

1. **Daily Monitoring**: Each day, the system calculates current asset weights based on price changes
2. **Threshold Check**: If any asset's weight deviates from target by more than the threshold:
   - Check if minimum interval since last rebalance has passed
   - Check if any assets have experienced critical drops (>30% monthly loss)
   - If all checks pass, initiate rebalancing
3. **Risk Adjustment**: Apply constraints (max weights) to rebalancing targets
4. **Transaction Costs**: Deduct costs based on amount of trading activity
5. **Weight Reset**: Return portfolio to target allocation

## Interpretation of Results

- **Positive Alpha**: Strategy outperformed benchmark on risk-adjusted basis
- **Lower Max Drawdown**: Strategy had less severe temporary losses
- **Higher Sharpe Ratio**: Better return per unit of risk taken
- **Fewer Rebalances**: More efficient strategy with lower transaction costs
- **Information Ratio**: Measures active management skill relative to benchmark

## Limitations

- Past performance doesn't guarantee future results
- Uses adjusted closing prices only (doesn't account for dividends unless reinvested)
- Transaction costs are estimates and may vary in practice
- Historical data availability depends on data source (Yahoo Finance)

## Integration with Main System

The backtesting module integrates with the main portfolio management system and uses the same configuration file and risk management principles, ensuring consistency between simulation and live management.