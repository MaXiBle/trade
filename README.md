# Trading Agent v2.0

A sophisticated reinforcement learning-based trading system with the following improvements over v1.0:

## Key Features

- **Time-Based Data Split**: Non-overlapping train/test periods (2018-2022 train, 2023-2024 test)
- **Continuous Action Space**: Position sizing from -1 (full short) to +1 (full long)
- **Risk Management**: Stop-loss, take-profit, and max position size controls
- **Realistic Reward Function**: PnL-based with transaction costs and volatility normalization
- **Advanced Algorithms**: SAC (Soft Actor-Critic) with support for PPO
- **Comprehensive Evaluation**: Walk-forward analysis, Sharpe ratio, max drawdown, win rate
- **No Lookahead Bias**: Properly lagged technical indicators

## Architecture

```
trading_agent_v2/
├── config/
│   └── config.yaml          # System configuration
├── data/                    # Data directory
├── envs/
│   └── trading_env.py       # Custom trading environment
├── models/                  # Trained models
├── utils/
│   ├── data_utils.py        # Data preparation utilities
│   └── reward_utils.py      # Reward calculation utilities
├── scripts/
│   ├── train_agent.py       # Training script
│   └── evaluate_agent.py    # Evaluation script
├── results/                 # Evaluation results
├── requirements.txt         # Dependencies
└── run_pipeline.py          # Main pipeline runner
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
python run_pipeline.py
```

### Train Only
```bash
python run_pipeline.py --train-only
```

### Evaluate Only
```bash
python run_pipeline.py --eval-only --config /path/to/config.yaml
```

## Configuration

Key configuration options in `config/config.yaml`:

- `data.symbol`: Trading symbol (default: AAPL)
- `data.train_start_date` / `data.test_end_date`: Time periods
- `environment.transaction_cost`: Transaction cost (default: 0.001 for 0.1%)
- `model.algorithm`: RL algorithm (SAC or PPO)
- `evaluation.walk_forward_periods`: Number of periods for walk-forward analysis

## Environment Features

The trading environment includes:

- **Action Space**: Continuous position size [-1, 1]
- **Observation Space**: Technical indicators + account information
- **Risk Controls**: Stop-loss, take-profit, max position limits
- **Transaction Costs**: Applied to each trade
- **Performance Tracking**: Real-time metrics and trade logging

## Evaluation Metrics

- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Volatility
- Total Trades

## Technical Indicators

- Simple Moving Averages (SMA 5, SMA 20)
- Relative Strength Index (RSI 14)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Average True Range (ATR)
- Volatility measures

All indicators are properly lagged to prevent lookahead bias.

## Performance Improvements Over v1.0

1. **Fixed Data Leakage**: Time-based splits eliminate overlap
2. **Enhanced Reward Function**: Real PnL with transaction costs
3. **Continuous Actions**: More nuanced position sizing vs discrete actions
4. **Risk Management**: Built-in stop-loss and position limits
5. **Robust Evaluation**: Walk-forward analysis prevents overfitting
6. **Modern RL Algorithm**: SAC for better sample efficiency