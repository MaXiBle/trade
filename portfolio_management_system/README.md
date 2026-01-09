# Automated Portfolio Management System

## Project Overview
An automated system for managing a diversified investment portfolio with dynamic rebalancing and contracyclical capital reallocation strategies.

## Features
- Portfolio monitoring with real-time market data
- Dynamic rebalancing based on target weights
- Risk management controls
- Backtesting module with performance metrics
- Configurable strategy parameters
- Modular architecture ready for broker API integration

## Architecture
The system consists of several modules:
- Data loading module
- Signal calculation module
- Operation execution module
- Backtesting module
- Configuration management

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from src.portfolio_manager import PortfolioManager

# Initialize with configuration
manager = PortfolioManager(config_file='config/default_config.json')
manager.run()
```