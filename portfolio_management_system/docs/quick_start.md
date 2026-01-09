# Quick Start Guide

## Prerequisites
- Python 3.7 or higher
- pip package manager

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/default_config.json` to customize:
- Assets in your portfolio
- Target weights for each asset
- Rebalancing threshold (default 5%)
- Minimum rebalance interval (default 7 days)
- Risk parameters

Example configuration:
```json
{
    "assets": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "target_weights": {
        "AAPL": 0.2,
        "MSFT": 0.2,
        "GOOGL": 0.2,
        "AMZN": 0.2,
        "TSLA": 0.2
    },
    "rebalance_threshold": 0.05,
    "min_rebalance_interval_days": 7,
    "max_asset_weight": 0.20,
    "critical_drop_threshold": -30.0
}
```

## Running the System

### Basic Execution
```bash
python main.py
```

### Using the API
```python
from src.portfolio_manager import PortfolioManager

# Initialize with custom config
manager = PortfolioManager(config_file='config/my_config.json')

# Run the system
manager.run()

# Get current performance
performance = manager.get_portfolio_performance()
```

### Running Backtest
```python
from src.backtester import Backtester
import json

with open('config/default_config.json', 'r') as f:
    config = json.load(f)

backtester = Backtester(config)
results = backtester.run_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2022-01-01',
    end_date='2023-01-01'
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## Understanding the Output

When you run `main.py`, you'll see:
1. Initial portfolio setup
2. Current weights vs target weights
3. Whether rebalancing was triggered
4. Final portfolio weights after rebalancing
5. Portfolio performance metrics
6. Backtest results comparing active vs passive strategy

## Key Metrics Explained

- **Total Return**: Overall gain/loss over the period
- **Annualized Return**: Average yearly return
- **Volatility**: Standard deviation of returns (risk measure)
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Max Drawdown**: Largest peak-to-trough decline
- **Number of Rebalances**: How many times the system adjusted the portfolio

## Customization Options

You can modify various aspects of the system:

1. **Assets**: Change the stock symbols in the config
2. **Weights**: Adjust target allocation percentages
3. **Thresholds**: Modify sensitivity to deviations
4. **Risk Limits**: Adjust maximum/minimum position sizes
5. **Strategy Logic**: Extend the rebalancing engine with custom rules