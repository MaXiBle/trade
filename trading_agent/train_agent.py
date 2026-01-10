import os
import sys
import numpy as np
import pandas as pd
import torch
import wandb
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append('/workspace')

from utils.data_processor import DataProcessor
from indicators.indicator_selector import IndicatorSelector, AdaptiveFeatureExtractor
from environments.trading_env import TradingEnv
from models.ppo_agent import AdaptivePPOAgent, TrainingCallback, PortfolioMetrics

def setup_logging(use_wandb: bool = True, config: Dict = None):
    """Setup experiment logging"""
    if use_wandb:
        wandb.init(
            project="quant-trading-agent",
            config=config or {},
            name=f"trading_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

def main():
    parser = argparse.ArgumentParser(description='Train a reinforcement learning trading agent')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to trade')
    parser.add_argument('--train_period', type=str, default='1y', help='Training period')
    parser.add_argument('--test_period', type=str, default='3mo', help='Testing period')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--train_steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--model_path', type=str, default='./models/trained_agent', help='Path to save model')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'symbol': args.symbol,
        'train_period': args.train_period,
        'test_period': args.test_period,
        'interval': args.interval,
        'initial_balance': args.initial_balance,
        'train_steps': args.train_steps,
        'transaction_fee': 0.001,
        'slippage_factor': 0.0005
    }
    
    # Setup logging
    if args.use_wandb:
        setup_logging(use_wandb=True, config=config)
    
    print("Starting Quantitative Trading Agent Training...")
    print(f"Symbol: {args.symbol}")
    print(f"Training Period: {args.train_period}")
    print(f"Test Period: {args.test_period}")
    print(f"Training Steps: {args.train_steps}")
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    data_processor = DataProcessor()
    
    # Get training data
    train_data = data_processor.fetch_data(args.symbol, period=args.train_period, interval=args.interval)
    train_data = data_processor.calculate_technical_indicators(train_data)
    
    # Get test data
    test_data = data_processor.fetch_data(args.symbol, period=args.test_period, interval=args.interval)
    test_data = data_processor.calculate_technical_indicators(test_data)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # 2. Initialize adaptive indicator selector
    print("\n2. Initializing adaptive indicator selector...")
    indicator_selector = IndicatorSelector(num_market_features=5, num_indicators=10, hidden_size=64)
    
    # 3. Create trading environments
    print("\n3. Creating trading environments...")
    train_env = TradingEnv(
        data=train_data,
        initial_balance=args.initial_balance,
        transaction_fee=config['transaction_fee'],
        slippage_factor=config['slippage_factor']
    )
    
    test_env = TradingEnv(
        data=test_data,
        initial_balance=args.initial_balance,
        transaction_fee=config['transaction_fee'],
        slippage_factor=config['slippage_factor']
    )
    
    # 4. Initialize and train the agent
    print("\n4. Initializing and training the agent...")
    agent = AdaptivePPOAgent(
        env=train_env,
        indicator_selector=indicator_selector,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_epochs=10
    )
    
    # Training callback
    callback = TrainingCallback(verbose=1)
    
    print(f"Starting training for {args.train_steps} steps...")
    agent.train(total_timesteps=args.train_steps, callback=callback)
    
    # 5. Evaluate the agent
    print("\n5. Evaluating the agent on test data...")
    obs, _ = test_env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action, _states = agent.predict(obs)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        if step_count % 100 == 0:
            print(f"Step {step_count}, Portfolio Value: ${info['portfolio_value']:.2f}, "
                  f"Return: {(info['portfolio_value']/args.initial_balance - 1)*100:.2f}%")
    
    # Final evaluation metrics
    final_portfolio_value = info['portfolio_value']
    total_return = (final_portfolio_value / args.initial_balance - 1) * 100
    sharpe_ratio = test_env._calculate_sharpe_ratio()
    max_drawdown = test_env._calculate_drawdown()
    win_rate = info['win_rate']
    
    print(f"\n--- FINAL RESULTS ---")
    print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Win Rate: {win_rate:.4f}")
    print(f"Total Trades: {info['total_trades']}")
    
    # Calculate additional metrics
    portfolio_metrics = PortfolioMetrics()
    returns_series = np.diff(test_env.portfolio_values) / test_env.portfolio_values[:-1]
    
    if len(returns_series) > 0:
        sortino_ratio = portfolio_metrics.calculate_sortino_ratio(returns_series.tolist())
        print(f"Sortino Ratio: {sortino_ratio:.4f}")
    
    # Log results to wandb if enabled
    if args.use_wandb:
        wandb.log({
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': info['total_trades'],
            'sortino_ratio': sortino_ratio if 'sortino_ratio' in locals() else 0
        })
        
        # Log portfolio value over time
        wandb.log({"portfolio_value_over_time": wandb.plot.line_series(
            xs=list(range(len(test_env.portfolio_values))),
            ys=[test_env.portfolio_values],
            keys=["Portfolio Value"],
            title="Portfolio Value Over Time",
            xname="Time Step"
        )})
        
        wandb.finish()
    
    # Save model if requested
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        agent.save(args.model_path)
        print(f"Model saved to {args.model_path}")
    
    # Save results
    results = {
        'symbol': args.symbol,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': info['total_trades'],
        'sortino_ratio': sortino_ratio if 'sortino_ratio' in locals() else 0,
        'config': config
    }
    
    results_path = f"./results/{args.symbol}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('./results', exist_ok=True)
    
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {results_path}")
    
    # Print some example indicator selections
    print("\n--- Example Indicator Selections ---")
    market_state_example = {
        'volatility_level': train_data['volatility'].iloc[-1],
        'trend_strength': abs(train_data['trend_sma'].iloc[-1]),
        'momentum_strength': abs(train_data['momentum'].iloc[-1]),
        'volume_activity': train_data['volume_ratio'].iloc[-1],
        'market_regime': train_data['market_regime'].iloc[-1]
    }
    
    # Convert market state to tensor for the indicator selector
    market_tensor = torch.tensor([
        market_state_example['volatility_level'],
        market_state_example['trend_strength'], 
        market_state_example['momentum_strength'],
        market_state_example['volume_activity'],
        0 if market_state_example['market_regime'] == 'calm_bull' else 1 if market_state_example['market_regime'] == 'calm_neutral' else 2
    ], dtype=torch.float32)
    
    selected_indicators = indicator_selector.get_selected_indicators(market_tensor, top_k=5)
    print("Top 5 selected indicators for current market state:")
    for indicator, weight in selected_indicators:
        print(f"  {indicator}: {weight:.4f}")

if __name__ == "__main__":
    main()