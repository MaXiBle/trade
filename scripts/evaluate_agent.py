import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append('/workspace')

from utils.data_utils import prepare_data
from envs.trading_env import TradingEnv
from stable_baselines3 import SAC, PPO
from utils.reward_utils import get_risk_adjusted_reward


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def evaluate_agent(model_path: str, config_path: str = '/workspace/config/config.yaml', 
                   test_data: pd.DataFrame = None):
    """
    Evaluate the trained agent on out-of-sample data
    """
    # Load configuration
    config = load_config(config_path)
    
    # Prepare data if not provided
    if test_data is None:
        _, test_data = prepare_data(config['data'])
    
    # Create test environment
    test_env = TradingEnv(
        data=test_data,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        max_position_size=config['environment']['max_position_size'],
        stop_loss_threshold=config['environment']['stop_loss_threshold'],
        take_profit_threshold=config['environment']['take_profit_threshold']
    )
    
    # Load the trained model
    algorithm = config['model']['algorithm']
    if algorithm == 'SAC':
        model = SAC.load(model_path)
    elif algorithm == 'PPO':
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    print("Evaluating the agent on test data...")
    
    # Run evaluation
    obs, _ = test_env.reset()
    done = False
    step_count = 0
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if step_count % 1000 == 0:
            print(f"Evaluation step: {step_count}, Equity: {info['equity']:.2f}")
    
    # Get performance metrics
    metrics = test_env.get_performance_metrics()
    
    # Print evaluation results
    print("\n" + "="*50)
    print("AGENT EVALUATION RESULTS")
    print("="*50)
    print(f"Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)")
    print(f"Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.2f}%)")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Volatility: {metrics.get('volatility', 0):.4f}")
    print(f"Final Balance: ${metrics.get('final_balance', 0):.2f}")
    print(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
    print("="*50)
    
    # Generate evaluation report
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'test_period': f"{config['data']['test_start_date']} to {config['data']['test_end_date']}",
        'symbol': config['data']['symbol'],
        'metrics': metrics,
        'trades_log': test_env.trades_log,
        'cumulative_returns': test_env.cumulative_returns
    }
    
    # Save evaluation report
    report_path = f"/workspace/results/evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("/workspace/results", exist_ok=True)
    
    import json
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2, default=str)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    # Plot equity curve
    plot_equity_curve(test_env.cumulative_returns, test_data, metrics)
    
    # Plot trade log if available
    if test_env.trades_log:
        plot_trade_log(test_env.trades_log, test_data)
    
    return metrics, test_env.trades_log


def plot_equity_curve(cumulative_returns, test_data, metrics):
    """
    Plot the equity curve over time
    """
    if not cumulative_returns:
        print("No cumulative returns to plot.")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Calculate equity values from cumulative returns
    initial_balance = 10000  # From config
    equity_values = [(1 + ret) * initial_balance for ret in cumulative_returns]
    
    dates = test_data['date'].iloc[60:60+len(equity_values)].reset_index(drop=True)
    
    plt.plot(dates, equity_values, label='Equity Curve', linewidth=2)
    plt.title(f'Trading Agent Equity Curve\nTotal Return: {metrics["total_return"]:.2%}, Sharpe: {metrics["sharpe_ratio"]:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = f"/workspace/results/equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Equity curve plot saved to: {plot_path}")


def plot_trade_log(trades_log, test_data):
    """
    Plot trade entries and exits on price chart
    """
    if not trades_log:
        print("No trades to plot.")
        return
    
    plt.figure(figsize=(14, 10))
    
    # Get the date range for the test period
    trade_dates = [trade['date'] for trade in trades_log]
    trade_prices = [trade['entry_price'] for trade in trades_log]
    trade_actions = [trade['action'] for trade in trades_log]
    
    # Separate buy, sell, and hold actions
    buy_indices = [i for i, action in enumerate(trade_actions) if action > 0.1]
    sell_indices = [i for i, action in enumerate(trade_actions) if action < -0.1]
    hold_indices = [i for i, action in enumerate(trade_actions) if -0.1 <= action <= 0.1]
    
    # Plot price
    plt.subplot(2, 1, 1)
    test_dates = test_data['date'].iloc[60:].reset_index(drop=True)
    test_prices = test_data['close'].iloc[60:].reset_index(drop=True)
    plt.plot(test_dates, test_prices, label='Price', color='blue', alpha=0.7)
    
    # Plot trades
    if buy_indices:
        buy_dates = [trade_dates[i] for i in buy_indices]
        buy_prices = [trade_prices[i] for i in buy_indices]
        plt.scatter(buy_dates, buy_prices, c='green', marker='^', s=100, label='Buy', zorder=5)
    
    if sell_indices:
        sell_dates = [trade_dates[i] for i in sell_indices]
        sell_prices = [trade_prices[i] for i in sell_indices]
        plt.scatter(sell_dates, sell_prices, c='red', marker='v', s=100, label='Sell', zorder=5)
    
    plt.title('Trading Activity on Price Chart')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot position sizes
    plt.subplot(2, 1, 2)
    position_sizes = [abs(trade['position_change']) / 10000 for trade in trades_log]  # Normalize for plotting
    plt.bar(range(len(position_sizes)), position_sizes, alpha=0.7, color='orange')
    plt.title('Normalized Position Changes')
    plt.xlabel('Trade Index')
    plt.ylabel('Position Size (Normalized)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"/workspace/results/trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    plt.show()
    print(f"Trade analysis plot saved to: {plot_path}")


def walk_forward_analysis(config_path: str = '/workspace/config/config.yaml'):
    """
    Perform walk-forward analysis over multiple time periods
    """
    config = load_config(config_path)
    
    # Divide test period into quarters for walk-forward analysis
    test_start = pd.to_datetime(config['data']['test_start_date'])
    test_end = pd.to_datetime(config['data']['test_end_date'])
    
    # Calculate period length for walk-forward analysis
    total_days = (test_end - test_start).days
    period_days = total_days // config['evaluation']['walk_forward_periods']
    
    results = []
    
    for i in range(config['evaluation']['walk_forward_periods']):
        period_start = test_start + pd.Timedelta(days=i * period_days)
        if i == config['evaluation']['walk_forward_periods'] - 1:
            period_end = test_end  # Last period goes to the end
        else:
            period_end = test_start + pd.Timedelta(days=(i + 1) * period_days)
        
        print(f"\nWalk-forward period {i+1}/{config['evaluation']['walk_forward_periods']}: "
              f"{period_start.date()} to {period_end.date()}")
        
        # Fetch data for this specific period
        import yfinance as yf
        period_data = yf.download(
            config['data']['symbol'], 
            start=period_start, 
            end=period_end
        ).reset_index()
        period_data.columns = period_data.columns.str.lower()
        
        # Calculate technical indicators for this period
        from utils.data_utils import calculate_technical_indicators
        period_data = calculate_technical_indicators(period_data)
        
        # Select features
        feature_columns = config['data']['features']
        period_data = period_data.fillna(0)  # Fill any NaN values
        
        # Create environment for this period
        period_env = TradingEnv(
            data=period_data,
            initial_balance=config['environment']['initial_balance'],
            transaction_cost=config['environment']['transaction_cost'],
            max_position_size=config['environment']['max_position_size'],
            stop_loss_threshold=config['environment']['stop_loss_threshold'],
            take_profit_threshold=config['environment']['take_profit_threshold']
        )
        
        # Load the trained model
        algorithm = config['model']['algorithm']
        model_path = f'/workspace/models/{algorithm}_trading_agent_final.zip'
        
        if os.path.exists(model_path):
            if algorithm == 'SAC':
                model = SAC.load(model_path)
            elif algorithm == 'PPO':
                model = PPO.load(model_path)
            
            # Evaluate on this period
            obs, _ = period_env.reset()
            done = False
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, terminated, truncated, info = period_env.step(action)
                done = terminated or truncated
            
            # Get metrics for this period
            period_metrics = period_env.get_performance_metrics()
            
            period_result = {
                'period': f"{period_start.date()} to {period_end.date()}",
                'metrics': period_metrics
            }
            
            results.append(period_result)
            
            print(f"  Total Return: {period_metrics.get('total_return', 0):.4f} "
                  f"({period_metrics.get('total_return', 0)*100:.2f}%)")
            print(f"  Sharpe Ratio: {period_metrics.get('sharpe_ratio', 0):.4f}")
            print(f"  Max Drawdown: {period_metrics.get('max_drawdown', 0):.4f} "
                  f"({period_metrics.get('max_drawdown', 0)*100:.2f}%)")
            print(f"  Win Rate: {period_metrics.get('win_rate', 0):.4f} "
                  f"({period_metrics.get('win_rate', 0)*100:.2f}%)")
        else:
            print(f"Model file not found: {model_path}")
            break
    
    # Aggregate results
    if results:
        agg_results = {}
        for metric in config['evaluation']['metrics']:
            values = [period['metrics'].get(metric, 0) for period in results if 'metrics' in period]
            if values:
                agg_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        print("\n" + "="*60)
        print("WALK-FORWARD ANALYSIS AGGREGATE RESULTS")
        print("="*60)
        for metric, stats in agg_results.items():
            print(f"{metric.upper()}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
                  f"Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        print("="*60)
    
    return results


if __name__ == "__main__":
    # Create results directory
    os.makedirs('/workspace/results', exist_ok=True)
    
    # Evaluate the agent
    model_path = '/workspace/models/SAC_trading_agent_final'  # Default path
    if not os.path.exists(model_path + '.zip'):
        # If .zip doesn't exist, try without extension
        if os.path.exists(model_path):
            model_path = model_path
        else:
            print("Trained model not found. Please train the agent first.")
            sys.exit(1)
    
    # Run evaluation
    metrics, trades_log = evaluate_agent(model_path)
    
    # Run walk-forward analysis
    walk_forward_results = walk_forward_analysis()
    
    print("\nEvaluation completed successfully!")