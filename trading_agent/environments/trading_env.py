import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class TradingEnv(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    Supports realistic market simulation with slippage, fees, and partial fills.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 transaction_fee: float = 0.001,  # 0.1% per trade
                 slippage_factor: float = 0.0005,  # 0.05% slippage
                 max_position_size: float = 1.0,   # Max fraction of portfolio per position
                 lookback_window: int = 20):
        super(TradingEnv, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage_factor = slippage_factor
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        
        # Define action space: [hold, buy, sell] with position sizing
        # Actions: -1 (sell all), -0.5 (sell half), 0 (hold), 0.5 (buy half), 1 (buy all)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Define observation space
        # Includes: normalized prices, technical indicators, portfolio state
        # Only include numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [col for col in numeric_cols if col not in excluded_cols]
        num_features = len(feature_cols)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(lookback_window, num_features + 5),  # +5 for portfolio state
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.balance = initial_balance
        self.position_size = 0  # Position size as fraction of balance (-1 to 1)
        self.shares_held = 0
        self.cost_basis = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        
        # Track portfolio history
        self.portfolio_values = [initial_balance]
        self.actions_history = []
        self.positions_history = []
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position_size = 0
        self.shares_held = 0
        self.cost_basis = 0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        
        self.portfolio_values = [self.initial_balance]
        self.actions_history = []
        self.positions_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from environment"""
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        # Get feature window
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        features = self.data[feature_cols].iloc[start_idx:end_idx].values.astype(np.float32)
        
        # Pad if necessary
        if features.shape[0] < self.lookback_window:
            padding = np.zeros((self.lookback_window - features.shape[0], features.shape[1]))
            features = np.vstack([padding, features])
        
        # Add portfolio state features
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position_size,                   # Current position size
            self.shares_held * current_price / self.balance if self.balance > 0 else 0,  # Portfolio allocation
            self._calculate_sharpe_ratio(),       # Current Sharpe ratio estimate
            self._calculate_drawdown()            # Current drawdown
        ]).astype(np.float32)
        
        # Repeat portfolio features for each time step in lookback window
        portfolio_features_expanded = np.tile(portfolio_features, (self.lookback_window, 1))
        
        # Combine features
        observation = np.concatenate([features, portfolio_features_expanded], axis=1)
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        # Ensure action is within bounds
        action = np.clip(action, -1, 1)[0]  # Take first element and clip to [-1, 1]
        
        # Store action
        self.actions_history.append(action)
        self.positions_history.append(self.position_size)
        
        # Get current price
        current_price = self.data['Close'].iloc[self.current_step]
        prev_portfolio_value = self.balance + self.shares_held * current_price
        
        # Calculate trade size based on action
        target_position_size = np.clip(action, -self.max_position_size, self.max_position_size)
        position_change = target_position_size - self.position_size
        
        # Execute trade with realistic market conditions
        executed_shares, cost = self._execute_trade(position_change, current_price)
        
        # Update portfolio
        self.shares_held += executed_shares
        self.balance -= cost
        self.position_size = self.shares_held * current_price / (self.balance + self.shares_held * current_price) if (self.balance + self.shares_held * current_price) != 0 else 0
        
        # Update trade statistics
        if abs(position_change) > 0.01:  # Only count meaningful trades
            self.total_trades += 1
            if position_change > 0 and self.shares_held > 0:  # Bought
                self.cost_basis = ((self.cost_basis * (self.shares_held - executed_shares)) + (current_price * executed_shares)) / self.shares_held if self.shares_held != 0 else 0
            elif position_change < 0 and self.shares_held >= 0:  # Sold
                if self.shares_held >= 0:  # Closing long position
                    pnl = (current_price - self.cost_basis) * abs(executed_shares)
                    if pnl > 0:
                        self.wins += 1
                    elif pnl < 0:
                        self.losses += 1
        
        # Calculate reward (portfolio return)
        current_portfolio_value = self.balance + self.shares_held * current_price
        portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value != 0 else 0
        
        # Risk-adjusted reward
        risk_adjustment = self._calculate_risk_adjustment()
        reward = portfolio_return * 100  # Scale for training stability
        reward += risk_adjustment  # Add penalty for high risk positions
        
        # Update portfolio history
        self.portfolio_values.append(current_portfolio_value)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        truncated = False  # Not using truncated termination
        
        # Increment step
        self.current_step += 1
        
        # Additional info
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position_size': self.position_size,
            'portfolio_value': current_portfolio_value,
            'total_trades': self.total_trades,
            'win_rate': self.wins / self.total_trades if self.total_trades > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'drawdown': self._calculate_drawdown()
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_trade(self, position_change: float, current_price: float) -> Tuple[float, float]:
        """Execute trade with slippage and fees"""
        if position_change == 0:
            return 0, 0
        
        # Calculate target shares
        target_value = position_change * (self.balance + self.shares_held * current_price)
        target_shares = target_value / current_price if current_price != 0 else 0
        
        # Apply slippage (worse fill for larger orders)
        slippage = self.slippage_factor * abs(position_change) * 10  # Amplify effect
        if position_change > 0:  # Buying
            execution_price = current_price * (1 + slippage)
        else:  # Selling
            execution_price = current_price * (1 - slippage)
        
        # Calculate cost including fees
        cost = target_shares * execution_price
        fee = abs(cost) * self.transaction_fee
        total_cost = cost + fee
        
        # Handle insufficient funds
        if total_cost > self.balance and position_change > 0:
            # Reduce position size to available funds
            available_funds = self.balance / (execution_price * (1 + self.transaction_fee))
            target_shares = min(target_shares, available_funds)
            total_cost = target_shares * execution_price + abs(target_shares * execution_price) * self.transaction_fee
        
        return target_shares, total_cost
    
    def _calculate_risk_adjustment(self) -> float:
        """Calculate risk-based reward adjustment"""
        current_portfolio_value = self.balance + self.shares_held * self.data['Close'].iloc[self.current_step]
        
        # Drawdown penalty
        max_portfolio = max(self.portfolio_values) if self.portfolio_values else self.initial_balance
        drawdown = (max_portfolio - current_portfolio_value) / max_portfolio if max_portfolio != 0 else 0
        drawdown_penalty = min(drawdown * 10, 2)  # Cap penalty at 2
        
        # Position concentration penalty
        position_concentration = abs(self.position_size)
        concentration_penalty = position_concentration * 0.1
        
        return -(drawdown_penalty + concentration_penalty)
    
    def _calculate_sharpe_ratio(self, window: int = 252) -> float:
        """Calculate rolling Sharpe ratio"""
        if len(self.portfolio_values) < 2:
            return 0
        
        # Calculate returns
        values = np.array(self.portfolio_values[-window:]) if len(self.portfolio_values) >= window else np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]
        
        if len(returns) == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming daily returns)
        if np.std(returns) == 0:
            return 0
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.portfolio_values:
            return 0
        
        current_value = self.portfolio_values[-1]
        peak_value = max(self.portfolio_values)
        
        if peak_value == 0:
            return 0
        
        drawdown = (peak_value - current_value) / peak_value
        return drawdown
    
    def render(self, mode='human'):
        """Render the environment state"""
        current_price = self.data['Close'].iloc[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price
        
        print(f'Step: {self.current_step}')
        print(f'Balance: ${self.balance:.2f}')
        print(f'Shares: {self.shares_held:.4f}')
        print(f'Current Price: ${current_price:.2f}')
        print(f'Portfolio Value: ${portfolio_value:.2f}')
        print(f'Position Size: {self.position_size:.4f}')
        print(f'Total Trades: {self.total_trades}')
        print(f'Win Rate: {self.wins/self.total_trades if self.total_trades > 0 else 0:.2f}')
        print('-' * 40)