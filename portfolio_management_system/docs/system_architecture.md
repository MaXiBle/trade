# System Architecture Documentation

## Overview
The Automated Portfolio Management System is designed to manage a diversified investment portfolio with dynamic rebalancing and contracyclical capital reallocation strategies.

## Core Components

### 1. Portfolio Manager (`src/portfolio_manager.py`)
The main orchestrator that:
- Initializes portfolio with target weights
- Monitors current portfolio weights against targets
- Determines when rebalancing is needed
- Executes rebalancing decisions
- Manages overall portfolio state

### 2. Data Loader (`src/data_loader.py`)
Responsible for:
- Fetching current market prices
- Retrieving historical price data
- Calculating price changes over time periods
- Interface with external data sources (Yahoo Finance)

### 3. Rebalancing Engine (`src/rebalancing_engine.py`)
Handles:
- Calculation of rebalancing signals based on weight deviations
- Adjustment of signals for transaction costs
- Determination of buy/sell amounts

### 4. Risk Manager (`src/risk_manager.py`)
Implements:
- Critical drop filters (prevents buying assets that dropped significantly)
- Weight limits (max/min allocation per asset)
- Position size validation
- Safety checks before executing trades

### 5. Backtester (`src/backtester.py`)
Provides:
- Historical simulation of portfolio strategies
- Performance metrics calculation
- Benchmark comparison
- Rebalancing event tracking

## Configuration
The system uses a JSON configuration file (`config/default_config.json`) that defines:
- Assets in the portfolio
- Target weights for each asset
- Rebalancing thresholds
- Risk management parameters
- Transaction cost rates

## Key Features Implemented

### Portfolio Monitoring
- Real-time tracking of portfolio weights
- Comparison against target allocations
- Deviation detection

### Dynamic Rebalancing
- Threshold-based rebalancing triggers
- Minimum interval between rebalances
- Capital reallocation from over-performers to under-performers

### Risk Controls
- Maximum allocation limits per asset (default 20%)
- Critical drop filters (-30% threshold)
- Minimum allocation limits (default 1%)
- Transaction cost considerations

### Backtesting Capabilities
- Historical performance simulation
- Key metrics: total return, volatility, Sharpe ratio, max drawdown
- Comparison with passive strategies
- Rebalancing frequency tracking

## How It Works

1. **Initialization**: Portfolio is set up with target weights from config
2. **Monitoring**: Current market values determine actual weights
3. **Analysis**: Compare actual vs target weights to detect deviations
4. **Decision**: Trigger rebalancing if deviations exceed threshold AND minimum interval has passed
5. **Filtering**: Apply risk controls to rebalancing signals
6. **Execution**: Update portfolio weights based on filtered signals
7. **Tracking**: Record rebalance events and performance metrics

## Extensibility

The modular design allows for:
- Integration with real broker APIs
- Addition of fundamental analysis filters
- Implementation of machine learning models
- Custom risk management rules
- Alternative rebalancing strategies

## Usage Example

```python
from src.portfolio_manager import PortfolioManager

# Initialize with configuration
manager = PortfolioManager(config_file='config/default_config.json')

# Run the management system
manager.run()

# Get performance metrics
performance = manager.get_portfolio_performance()
```

For backtesting:
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
```