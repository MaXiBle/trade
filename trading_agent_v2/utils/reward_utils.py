import numpy as np
from typing import Dict, List
import pandas as pd


def calculate_reward(realized_pnl: float, transaction_cost: float, volatility: float = 1.0) -> float:
    """
    Calculate reward based on realized PnL with transaction costs and volatility normalization
    """
    # Net PnL after transaction costs
    net_pnl = realized_pnl - transaction_cost
    
    # Volatility-normalized reward to reduce risk
    if volatility > 0:
        normalized_reward = net_pnl / volatility
    else:
        normalized_reward = net_pnl
    
    # Additional risk adjustment: penalize large position changes relative to account size
    return normalized_reward


def calculate_volatility_adjusted_return(returns: List[float], window: int = 20) -> float:
    """
    Calculate volatility-adjusted return for reward shaping
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate rolling volatility
    if len(returns) >= window:
        recent_volatility = np.std(returns[-window:]) * np.sqrt(252)  # Annualized
    else:
        recent_volatility = np.std(returns) * np.sqrt(252)
    
    # Calculate return over the period
    period_return = sum(returns)
    
    # Adjust return by volatility (higher volatility reduces reward)
    if recent_volatility > 0:
        adjusted_return = period_return / (recent_volatility + 1e-8)
    else:
        adjusted_return = period_return
    
    return adjusted_return


def get_risk_adjusted_reward(equity_curve: List[float], transaction_costs: List[float]) -> float:
    """
    Calculate comprehensive risk-adjusted reward
    """
    if len(equity_curve) < 2:
        return 0.0
    
    # Calculate returns from equity curve
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    
    # Total net return
    total_return = equity_curve[-1] / equity_curve[0] - 1 if equity_curve[0] != 0 else 0
    
    # Calculate Sharpe-like ratio (return per unit of risk)
    if len(returns) > 1:
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        if volatility > 0:
            sharpe_like = (np.mean(returns) * 252) / volatility
        else:
            sharpe_like = total_return * 252  # If no volatility, just use annualized return
    else:
        sharpe_like = total_return
    
    # Total transaction costs
    total_costs = sum(transaction_costs)
    
    # Risk-adjusted reward combining return, risk, and costs
    risk_adjusted_reward = sharpe_like - total_costs
    
    return risk_adjusted_reward


def calculate_portfolio_change_reward(balance: float, prev_balance: float, transaction_cost: float) -> float:
    """
    Calculate reward based on portfolio value change with transaction cost adjustment
    """
    portfolio_change = balance - prev_balance
    net_reward = portfolio_change - transaction_cost
    return net_reward