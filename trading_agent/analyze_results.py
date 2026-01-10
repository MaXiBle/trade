import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import sys

# Add project root to path
sys.path.append('/workspace')

from utils.data_processor import DataProcessor
from environments.trading_env import TradingEnv
from models.ppo_agent import PortfolioMetrics

def load_results(file_path: str):
    """Load training results from JSON file"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results

def visualize_performance(data, actions_history, portfolio_values, symbol: str = "STOCK"):
    """Create comprehensive performance visualization"""
    
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=('Price Action', 'Portfolio Value', 'Actions Taken', 'Drawdown',
                       'Returns Distribution', 'Rolling Sharpe Ratio', 'Volume Profile', 'Market Regime'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Plot 1: Price Action
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['sma_20'], name='SMA 20', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=data.index, y=data['sma_50'], name='SMA 50', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # Plot 2: Portfolio Value
    fig.add_trace(
        go.Scatter(x=list(range(len(portfolio_values))), y=portfolio_values, 
                  name='Portfolio Value', line=dict(color='green')),
        row=1, col=2
    )
    
    # Plot 3: Actions taken
    fig.add_trace(
        go.Scatter(x=list(range(len(actions_history))), y=actions_history, 
                  name='Actions', line=dict(color='purple'), mode='lines'),
        row=2, col=1
    )
    
    # Plot 4: Drawdown
    portfolio_series = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_series)
    drawdown = (portfolio_series - running_max) / running_max
    fig.add_trace(
        go.Scatter(x=list(range(len(drawdown))), y=drawdown, 
                  name='Drawdown', line=dict(color='red')),
        row=2, col=2
    )
    
    # Plot 5: Returns distribution
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    fig.add_trace(
        go.Histogram(x=returns, name='Returns Distribution', nbinsx=50),
        row=3, col=1
    )
    
    # Plot 6: Rolling Sharpe Ratio
    window = 22  # 22 days ~ 1 month
    rolling_returns = pd.Series(returns).rolling(window=window).apply(
        lambda x: np.mean(x) / np.std(x) * np.sqrt(252) if np.std(x) != 0 else 0
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(rolling_returns))), y=rolling_returns, 
                  name='Rolling Sharpe', line=dict(color='orange')),
        row=3, col=2
    )
    
    # Plot 7: Volume profile
    fig.add_trace(
        go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightgray'),
        row=4, col=1
    )
    
    # Plot 8: Market regime
    regime_colors = {
        'calm_bull': 'lightgreen', 'calm_neutral': 'lightblue', 'calm_bear': 'lightcoral',
        'trending_bull': 'green', 'choppy': 'gray', 'trending_bear': 'red',
        'volatile_bull': 'darkgreen', 'volatile_neutral': 'darkgray', 'volatile_bear': 'darkred'
    }
    
    for regime, color in regime_colors.items():
        mask = data['market_regime'] == regime
        if mask.any():
            fig.add_trace(
                go.Scatter(x=data.index[mask], y=data['Close'][mask], 
                          mode='markers', name=regime, marker=dict(color=color, size=4)),
                row=4, col=2
            )
    
    fig.update_layout(height=1200, showlegend=True, 
                      title_text=f"Trading Performance Analysis for {symbol}")
    
    # Show the plot
    fig.show()
    
    return fig

def generate_report(results_file: str, data: pd.DataFrame):
    """Generate a comprehensive performance report"""
    
    results = load_results(results_file)
    
    print("="*60)
    print("TRADING AGENT PERFORMANCE REPORT")
    print("="*60)
    print(f"Symbol: {results['symbol']}")
    print(f"Training Period: {results['config']['train_period']}")
    print(f"Test Period: {results['config']['test_period']}")
    print(f"Initial Balance: ${results['config']['initial_balance']:,.2f}")
    print()
    
    print("PERFORMANCE METRICS:")
    print("-"*30)
    print(f"Final Portfolio Value: ${results['final_portfolio_value']:,.2f}")
    print(f"Total Return: {results['total_return']:+.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.4f}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']:,}")
    print(f"Sortino Ratio: {results['sortino_ratio']:.4f}")
    print()
    
    # Calculate additional metrics from data
    buy_and_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
    print(f"Buy & Hold Return: {buy_and_hold_return:+.2f}%")
    print(f"Alpha vs Buy & Hold: {results['total_return'] - buy_and_hold_return:+.2f}%")
    print()
    
    # Risk metrics
    volatility = data['returns'].std() * np.sqrt(252)  # Annualized
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Calmar Ratio (Return/DD): {results['total_return']/abs(results['max_drawdown']):.2f}" if results['max_drawdown'] != 0 else "N/A")
    print()
    
    print("TECHNICAL INSIGHTS:")
    print("-"*30)
    print(f"Average RSI: {data['rsi'].mean():.2f}")
    print(f"RSI Range: {data['rsi'].min():.1f} - {data['rsi'].max():.1f}")
    print(f"Average Volatility: {data['volatility'].mean():.4f}")
    print(f"Most Common Regime: {data['market_regime'].mode()[0] if len(data['market_regime']) > 0 else 'N/A'}")
    print()
    
    print("="*60)

def analyze_indicator_effectiveness(data: pd.DataFrame, actions_history: List[float], 
                                 portfolio_values: List[float]):
    """Analyze how well different indicators aligned with profitable trades"""
    
    # Calculate trade P&Ls
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    actions_array = np.array(actions_history)
    
    # Align indicators with actions
    indicators_aligned = data[['rsi', 'macd', 'bb_position', 'volatility']].iloc[:len(actions_array)]
    
    correlation_results = {}
    
    for indicator in indicators_aligned.columns:
        # Correlate indicator values with next period returns
        indicator_values = indicators_aligned[indicator].dropna()
        future_returns = pd.Series(returns).iloc[:len(indicator_values)]
        
        correlation = np.corrcoef(indicator_values, future_returns)[0, 1]
        correlation_results[indicator] = correlation
    
    print("INDICATOR EFFECTIVENESS ANALYSIS:")
    print("-"*40)
    for indicator, corr in correlation_results.items():
        print(f"{indicator:>12}: {corr:>+7.4f}")
    
    return correlation_results

def create_backtest_summary(env: TradingEnv, data: pd.DataFrame, symbol: str):
    """Create a detailed backtest summary"""
    
    # Generate metrics
    returns = np.diff(env.portfolio_values) / env.portfolio_values[:-1]
    
    metrics = {
        'total_return': (env.portfolio_values[-1] / env.portfolio_values[0] - 1) * 100,
        'annualized_return': ((env.portfolio_values[-1] / env.portfolio_values[0]) ** (252 / len(env.portfolio_values)) - 1) * 100,
        'volatility': np.std(returns) * np.sqrt(252) * 100,
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0,
        'max_drawdown': min(0, env._calculate_drawdown() * 100),
        'win_rate': env.wins / env.total_trades if env.total_trades > 0 else 0,
        'profit_factor': abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if sum(r for r in returns if r < 0) != 0 else float('inf'),
        'total_trades': env.total_trades,
        'avg_return_per_trade': np.mean(returns) * 100 if len(returns) > 0 else 0
    }
    
    print(f"\nBACKTEST SUMMARY FOR {symbol}")
    print("="*50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            if metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'avg_return_per_trade']:
                print(f"{metric.replace('_', ' ').title():>20}: {value:7.2f}%")
            elif metric in ['sharpe_ratio', 'profit_factor']:
                print(f"{metric.replace('_', ' ').title():>20}: {value:7.3f}")
            elif metric == 'win_rate':
                print(f"{metric.replace('_', ' ').title():>20}: {value:7.2%}")
            else:
                print(f"{metric.replace('_', ' ').title():>20}: {value:7.0f}")
        else:
            print(f"{metric.replace('_', ' ').title():>20}: {value}")
    
    return metrics

# Example usage function
def run_analysis_example():
    """Run an example analysis"""
    
    # This would normally load results from a training run
    # For now, we'll create example data
    print("Example Analysis Function - This would analyze real results from a training run")
    print("To run a real analysis, first train the agent using train_agent.py")

if __name__ == "__main__":
    run_analysis_example()