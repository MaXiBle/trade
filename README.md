# Quantitative Trading Agent with Reinforcement Learning

An advanced reinforcement learning system for autonomous stock trading that adapts to market conditions by dynamically selecting technical indicators.

## 🎯 Features

- **Adaptive Indicator Selection**: Automatically chooses relevant technical indicators (RSI, MACD, Bollinger Bands, etc.) based on current market conditions
- **Realistic Trading Environment**: Simulates slippage, commissions, and partial fills
- **Risk Management**: Controls drawdown and implements position sizing
- **Performance Metrics**: Tracks Sharpe ratio, Sortino ratio, win rate, and more
- **Modular Architecture**: Clean, extensible codebase with separate modules for data, indicators, environment, and agents

## 🏗️ Architecture

```
trading_agent/
├── data/                 # Data storage
├── models/               # RL agent implementations
│   └── ppo_agent.py      # PPO-based trading agent with attention mechanism
├── environments/         # Trading environment
│   └── trading_env.py    # Custom Gym environment with realistic market conditions
├── indicators/           # Adaptive indicator selection
│   └── indicator_selector.py  # Neural network for indicator importance prediction
├── utils/                # Utility functions
│   └── data_processor.py # Technical indicator calculation and data processing
├── train_agent.py        # Main training script
└── analyze_results.py    # Performance analysis and visualization
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training the Agent

```bash
python trading_agent/train_agent.py --symbol AAPL --train_period 1y --test_period 3mo --train_steps 50000
```

### Training Options

- `--symbol`: Stock symbol to trade (default: AAPL)
- `--train_period`: Training period (default: 1y)
- `--test_period`: Testing period (default: 3mo)
- `--interval`: Data interval (default: 1d)
- `--initial_balance`: Starting capital (default: 10000)
- `--train_steps`: Number of training steps (default: 50000)
- `--use_wandb`: Enable Weights & Biases logging
- `--save_model`: Save the trained model
- `--model_path`: Path to save/load model

### Example Usage

```python
from trading_agent import DataProcessor, TradingEnv, AdaptivePPOAgent, IndicatorSelector

# Initialize components
data_processor = DataProcessor()
indicator_selector = IndicatorSelector()

# Fetch and process data
data = data_processor.fetch_data('AAPL', period='1y')
data = data_processor.calculate_technical_indicators(data)

# Create environment
env = TradingEnv(data)

# Initialize agent
agent = AdaptivePPOAgent(env, indicator_selector)

# Train the agent
agent.train(total_timesteps=10000)

# Make predictions
obs, _ = env.reset()
action, _ = agent.predict(obs)
```

## 🔬 Key Components

### 1. Data Processor
- Fetches OHLCV data from Yahoo Finance
- Calculates technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Identifies market regimes (trending, volatile, calm)

### 2. Indicator Selector
- Neural network that predicts indicator importance based on market state
- Dynamically selects which indicators to focus on
- Adapts to changing market conditions

### 3. Trading Environment
- Realistic simulation with slippage and commissions
- Portfolio tracking and risk management
- Support for long/short positions

### 4. PPO Agent
- Policy gradient method with advantage estimation
- Custom CNN-based feature extractor for time series
- Attention mechanism for focusing on relevant information

## 📊 Performance Metrics

The agent optimizes for:
- **Portfolio Return**: Maximizing overall profitability
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Controlling downside risk
- **Win Rate**: Consistency of profitable trades
- **Sortino Ratio**: Downside deviation focused metric

## 🛠️ Customization

### Adding New Indicators
1. Add calculation logic to `data_processor.py`
2. Update feature selection in `indicator_selector.py`
3. Retrain the indicator selector

### Modifying Risk Management
Adjust parameters in `trading_env.py`:
- Transaction fees
- Slippage factors
- Position sizing limits
- Risk penalties

## 📈 Results Analysis

After training, analyze results with:

```bash
python trading_agent/analyze_results.py
```

This provides:
- Comprehensive performance reports
- Visualization of trading activity
- Indicator effectiveness analysis
- Risk-adjusted metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

*Built with ❤️ for quantitative finance enthusiasts*