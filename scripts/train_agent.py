import os
import sys
import yaml
import numpy as np
import torch
import pickle
from datetime import datetime

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
pathh = project_root + '/'

from utils.data_utils import prepare_data, get_feature_names
from envs.trading_env import TradingEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.reward_utils import get_risk_adjusted_reward


class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback for logging during training
    """
    def __init__(self, verbose=0):
        super(TrainingLoggerCallback, self).__init__(verbose)
        self.training_logs = []

    def _on_step(self) -> bool:
        # Log training metrics periodically
        if self.n_calls % 1000 == 0:
            print(f"Step {self.num_timesteps}: Training in progress...")
        return True


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train_agent(config_path: str = pathh + 'config/config.yaml'):
    """
    Main training function
    """
    # Load configuration
    config = load_config(config_path)
    
    # Prepare data
    print("Preparing training data...")
    train_data, test_data = prepare_data(config['data'])
    
    # Create environment
    env = TradingEnv(
        data=train_data,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        max_position_size=config['environment']['max_position_size'],
        stop_loss_threshold=config['environment']['stop_loss_threshold'],
        take_profit_threshold=config['environment']['take_profit_threshold']
    )
    
    # Create evaluation environment
    eval_env = TradingEnv(
        data=test_data,
        initial_balance=config['environment']['initial_balance'],
        transaction_cost=config['environment']['transaction_cost'],
        max_position_size=config['environment']['max_position_size'],
        stop_loss_threshold=config['environment']['stop_loss_threshold'],
        take_profit_threshold=config['environment']['take_profit_threshold']
    )
    
    # Choose algorithm based on config
    algorithm = config['model']['algorithm']
    
    if algorithm == 'SAC':
        print("Initializing SAC agent...")
        model = SAC(
            'MlpPolicy',
            env,
            buffer_size=config['model']['buffer_size'],
            batch_size=config['model']['batch_size'],
            learning_rate=config['model']['learning_rate'],
            gamma=config['model']['gamma'],
            tau=config['model']['tau'],
            ent_coef=config['model']['ent_coef'],
            verbose=1,
            tensorboard_log=os.path.join(project_root, f"logs/sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
    elif algorithm == 'PPO':
        print("Initializing PPO agent...")
        model = PPO(
            'MlpPolicy',
            env,
            n_steps=2048,
            batch_size=config['model']['batch_size'],
            learning_rate=config['model']['learning_rate'],
            gamma=config['model']['gamma'],
            verbose=1,
            tensorboard_log=os.path.join(project_root, f"logs/ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(project_root, 'models/best_model'),
        log_path=os.path.join(project_root, 'logs/'),
        eval_freq=config['training']['eval_freq'],
        deterministic=True,
        render=False
    )
    
    # Custom logger callback
    logger_callback = TrainingLoggerCallback(verbose=1)
    
    # Train the model
    print(f"Starting training with {algorithm} for {config['training']['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, logger_callback],
        progress_bar=True
    )
    
    # Save the trained model
    model_save_path = os.path.join(project_root, f"models/{algorithm}_trading_agent_final")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Save the environment for later use
    env_save_path = os.path.join(project_root, f"models/training_env.pkl")
    with open(env_save_path, 'wb') as f:
        pickle.dump(env, f)
    print(f"Training environment saved to {env_save_path}")
    
    return model, env, test_data


if __name__ == "__main__":
    # Create logs and models directories if they don't exist
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
    
    # Train the agent
    trained_model, training_env, test_data = train_agent()
    
    print("Training completed successfully!")