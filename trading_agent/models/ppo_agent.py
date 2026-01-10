import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    """Attention mechanism to focus on relevant features"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, features]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attended = x * attention_weights
        return attended.sum(dim=1)  # [batch_size, features]

class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom CNN-based feature extractor for time-series data"""
    
    def __init__(self, observation_space, features_dim: int = 256):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0] * observation_space.shape[1]  # Flatten temporal dimensions
        
        # Convolutional layers to capture temporal patterns
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=observation_space.shape[1], 
                     out_channels=32, 
                     kernel_size=3, 
                     stride=1, 
                     padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, 
                     out_channels=64, 
                     kernel_size=3, 
                     stride=1, 
                     padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        
        # Calculate output size after convolutions
        with torch.no_grad():
            dummy_input = torch.randn(1, observation_space.shape[1], observation_space.shape[0])
            conv_output_size = self.cnn(dummy_input).shape[1]
        
        # Dense layers after convolution
        self.linear = nn.Sequential(
            nn.Linear(conv_output_size, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape observations for CNN: [batch, channels, time_steps]
        x = observations.permute(0, 2, 1)  # [batch, features, time_steps]
        x = self.cnn(x)
        x = self.linear(x)
        return x

class AdaptivePPOAgent:
    """PPO agent with adaptive indicator selection"""
    
    def __init__(self, 
                 env,
                 indicator_selector,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 batch_size: int = 64,
                 n_epochs: int = 10):
        self.env = env
        self.indicator_selector = indicator_selector
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Create policy network with custom feature extractor
        policy_kwargs = {
            "features_extractor_class": CustomFeatureExtractor,
            "features_extractor_kwargs": {"features_dim": 256},
            "net_arch": [dict(pi=[256, 128], vf=[256, 128])]
        }
        
        self.model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
    
    def train(self, total_timesteps: int, callback: Optional[BaseCallback] = None):
        """Train the PPO agent"""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
    
    def predict(self, observation):
        """Make prediction using the trained model"""
        return self.model.predict(observation, deterministic=True)
    
    def save(self, path: str):
        """Save the trained model"""
        self.model.save(path)
    
    def load(self, path: str):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)

class TrainingCallback(BaseCallback):
    """Custom callback for logging during training"""
    
    def __init__(self, verbose: int = 0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        
    def _on_step(self) -> bool:
        # Log episode rewards
        if "episode" in self.locals:
            self.episode_rewards.append(self.locals["episode"]["r"])
            self.current_episode_reward = 0
        else:
            self.current_episode_reward += self.locals["rewards"][0]
        
        return True
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        if not self.episode_rewards:
            return {}
        
        return {
            "avg_reward": np.mean(self.episode_rewards),
            "std_reward": np.std(self.episode_rewards),
            "min_reward": np.min(self.episode_rewards),
            "max_reward": np.max(self.episode_rewards),
            "num_episodes": len(self.episode_rewards)
        }

class PortfolioMetrics:
    """Calculate portfolio performance metrics"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(252)  # Annualize
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return float(np.min(drawdown))
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], target_return: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - target_return
        
        negative_returns = returns[returns < target_return]
        if len(negative_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_deviation
        return sortino * np.sqrt(252)  # Annualize
    
    @staticmethod
    def calculate_win_rate(trade_pnls: List[float]) -> float:
        """Calculate win rate"""
        if not trade_pnls:
            return 0.0
        
        wins = sum(1 for pnl in trade_pnls if pnl > 0)
        return wins / len(trade_pnls)