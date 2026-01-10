import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Optional


class TradingEnv(gym.Env):
    """
    Continuous action trading environment for RL agents
    """
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 max_position_size: float = 1.0,    # 100% max position
                 stop_loss_threshold: float = 0.05, # 5% stop loss
                 take_profit_threshold: float = 0.1 # 10% take profit
                 ):
        super(TradingEnv, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        
        # Extract feature columns from the data
        self.feature_columns = [col for col in data.columns if col not in [
            'date', 'open', 'high', 'low', 'close', 'adj close', 'volume'
        ]]
        
        # Environment dimensions
        self.n_features = len(self.feature_columns)
        self.lookback_period = 60  # Using lookback period from config
        
        # Define action and observation spaces
        # Action: position size [-1, 1] where -1 = full short, 1 = full long
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: features + account info
        # Account info: balance, equity, unrealized_pnl, position_size, avg_entry_price
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features * self.lookback_period + 5,), 
            dtype=np.float32
        )
        
        # Reset environment
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_period
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.position_size = 0.0  # Position size in dollars (0 = no position)
        self.avg_entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_log = []
        self.cumulative_returns = []
        self.episode_return = 0.0
        self.done = False
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation vector"""
        # Get the recent features for the lookback period
        start_idx = max(0, self.current_step - self.lookback_period)
        end_idx = self.current_step
        
        # Pad with zeros if we don't have enough historical data
        if end_idx - start_idx < self.lookback_period:
            pad_length = self.lookback_period - (end_idx - start_idx)
            features_slice = np.zeros((pad_length, self.n_features))
            actual_features = self.data[self.feature_columns].iloc[start_idx:end_idx].values
            combined_features = np.vstack([features_slice, actual_features])
        else:
            combined_features = self.data[self.feature_columns].iloc[start_idx:end_idx].values
        
        # Flatten the feature matrix
        flattened_features = combined_features.flatten()
        
        # Append account information
        current_price = self.data['close'].iloc[self.current_step] if self.current_step < len(self.data) else 1.0
        relative_avg_price = self.avg_entry_price / current_price if self.avg_entry_price != 0 and current_price != 0 else 0
        
        account_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.equity / self.initial_balance,   # Normalized equity
            self.unrealized_pnl / self.initial_balance,  # Normalized unrealized pnl
            self.position_size / self.initial_balance,   # Normalized position size
            relative_avg_price  # Relative to current price
        ])
        
        # Combine features and account info
        observation = np.concatenate([flattened_features, account_info]).astype(np.float32)
        
        # Ensure all values are finite to prevent NaN propagation to the neural network
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e5, neginf=-1e5)
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        if self.done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        
        # Get current price
        current_price = self.data['close'].iloc[self.current_step]
        
        # Current position value
        prev_unrealized_pnl = self.unrealized_pnl
        if self.position_size != 0 and self.avg_entry_price != 0:
            self.unrealized_pnl = (current_price - self.avg_entry_price) * (self.position_size / self.avg_entry_price)
            realized_pnl_change = self.unrealized_pnl - prev_unrealized_pnl
        else:
            realized_pnl_change = 0
        
        # Clamp action to valid range [-1, 1]
        target_position_pct = np.clip(action[0], -1.0, 1.0)
        
        # Check for stop loss/take profit conditions
        if abs(self.position_size) > 0 and self.avg_entry_price != 0:
            pct_change = abs(current_price - self.avg_entry_price) / self.avg_entry_price
            
            # Stop loss check
            if pct_change >= self.stop_loss_threshold and (
                (target_position_pct > 0 and current_price < self.avg_entry_price) or  # Long position going down
                (target_position_pct < 0 and current_price > self.avg_entry_price)    # Short position going up
            ):
                target_position_pct = 0.0  # Close position
                
            # Take profit check
            elif pct_change >= self.take_profit_threshold and (
                (target_position_pct > 0 and current_price > self.avg_entry_price) or  # Long position going up
                (target_position_pct < 0 and current_price < self.avg_entry_price)    # Short position going down
            ):
                target_position_pct = 0.0  # Close position
        
        # Calculate desired position value
        desired_position_value = target_position_pct * self.equity
        
        # Calculate position change needed
        position_change_value = desired_position_value - self.position_size
        
        # Apply transaction costs
        transaction_amount = abs(position_change_value)
        transaction_fee = transaction_amount * self.transaction_cost
        
        # Check if we have enough balance for the transaction
        if transaction_fee > self.balance:
            # Cannot execute transaction, maintain current position
            reward = 0.0
        else:
            # Update position
            self.position_size = desired_position_value
            if self.position_size != 0:
                # Only update average entry price when opening/closing positions
                if abs(desired_position_value) > abs(self.position_size - position_change_value):
                    # Position increased, update average entry price
                    if position_change_value > 0:
                        new_total_shares = abs(self.position_size) / current_price
                        old_shares = abs(self.position_size - position_change_value) / self.avg_entry_price if self.avg_entry_price != 0 else 0
                        if old_shares > 0:
                            self.avg_entry_price = ((old_shares * self.avg_entry_price) + (abs(position_change_value) / current_price * current_price)) / new_total_shares
                        else:
                            self.avg_entry_price = current_price
                    else:
                        # For short positions, the logic is inverted
                        if position_change_value < 0:
                            new_total_shares = abs(self.position_size) / current_price
                            old_shares = abs(self.position_size - position_change_value) / self.avg_entry_price if self.avg_entry_price != 0 else 0
                            if old_shares > 0:
                                self.avg_entry_price = ((old_shares * self.avg_entry_price) + (abs(position_change_value) / current_price * current_price)) / new_total_shares
                            else:
                                self.avg_entry_price = current_price
                # Calculate unrealized PnL based on new position
                if self.avg_entry_price != 0:
                    self.unrealized_pnl = (current_price - self.avg_entry_price) * (self.position_size / self.avg_entry_price)
                else:
                    self.unrealized_pnl = 0.0
            else:
                # Closed position
                self.avg_entry_price = 0.0
                self.unrealized_pnl = 0.0
            
            # Update balance (subtract transaction fees)
            self.balance -= transaction_fee
            
            # Calculate reward as change in equity (realized + unrealized PnL)
            old_equity = self.balance + prev_unrealized_pnl
            self.equity = self.balance + self.unrealized_pnl
            reward = self.equity - old_equity
            
            # Ensure reward is finite to avoid NaN propagation
            if not np.isfinite(reward):
                reward = 0.0
                print(f"Warning: Non-finite reward detected at step {self.current_step}, setting to 0.0")
            
            # Log the trade if position changed significantly
            if abs(position_change_value) > self.initial_balance * 0.001:  # Only log significant changes
                self.trades_log.append({
                    'step': self.current_step,
                    'date': self.data.iloc[self.current_step]['date'],
                    'action': target_position_pct,
                    'position_change': position_change_value,
                    'entry_price': current_price,
                    'balance': self.balance,
                    'equity': self.equity,
                    'unrealized_pnl': self.unrealized_pnl,
                    'transaction_fee': transaction_fee
                })
        
        # Calculate cumulative return
        self.cumulative_returns.append(self.equity / self.initial_balance - 1)
        
        # Check termination conditions
        self.done = (
            self.current_step >= len(self.data) - 1 or
            self.equity <= self.initial_balance * 0.1  # Stop if equity drops below 10% of initial balance
        )
        
        # Truncate is always False in this environment
        truncated = False
        
        # Update step
        self.current_step += 1
        
        # Return observation, reward, done, truncated, info
        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position_size': self.position_size,
            'avg_entry_price': self.avg_entry_price,
            'unrealized_pnl': self.unrealized_pnl,
            'step': self.current_step,
            'total_trades': len(self.trades_log),
            'current_price': current_price
        }
        
        return self._get_observation(), reward, self.done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment"""
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, '
              f'Equity: {self.equity:.2f}, Position: {self.position_size:.2f}')
    
    def get_performance_metrics(self) -> dict:
        """Calculate performance metrics"""
        if len(self.cumulative_returns) < 2:
            return {}
        
        returns = np.diff([0] + self.cumulative_returns)  # Daily returns
        
        total_return = self.cumulative_returns[-1] if self.cumulative_returns else 0
        total_trades = len(self.trades_log)
        
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility != 0 else 0
            max_drawdown = self._calculate_max_drawdown()
            
            # Win rate calculation
            positive_returns = sum(1 for r in returns if r > 0)
            win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'volatility': volatility,
            'final_balance': self.balance,
            'final_equity': self.equity
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.cumulative_returns:
            return 0
        
        peak = 0
        max_dd = 0
        
        for ret in self.cumulative_returns:
            if ret > peak:
                peak = ret
            dd = (peak - ret) / (1 + peak) if (1 + peak) != 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd