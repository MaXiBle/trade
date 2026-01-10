#!/usr/bin/env python3
"""
Trading Agent v2.0 Pipeline Runner

This script orchestrates the complete pipeline:
1. Data preparation
2. Model training
3. Evaluation
4. Walk-forward analysis
"""

import os
import sys
import argparse
from datetime import datetime

# Add the project root to the path
sys.path.append('/workspace/trading_agent_v2')

from scripts.train_agent import train_agent
from scripts.evaluate_agent import evaluate_agent, walk_forward_analysis
from utils.data_utils import prepare_data
from envs.trading_env import TradingEnv
from stable_baselines3 import SAC, PPO


def main():
    parser = argparse.ArgumentParser(description='Run Trading Agent v2.0 Pipeline')
    parser.add_argument('--config', type=str, default='/workspace/trading_agent_v2/config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and only run evaluation')
    parser.add_argument('--train-only', action='store_true',
                        help='Only train the agent without evaluation')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation without training')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRADING AGENT V2.0 PIPELINE")
    print("="*60)
    
    if not args.skip_training and not args.eval_only:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training phase...")
        trained_model, training_env, test_data = train_agent(args.config)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
    
    if not args.train_only:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting evaluation phase...")
        
        # Determine model path based on config
        import yaml
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        algorithm = config['model']['algorithm']
        model_path = f'/workspace/trading_agent_v2/models/{algorithm}_trading_agent_final'
        
        if os.path.exists(model_path + '.zip'):
            model_path = model_path + '.zip'
        elif os.path.exists(model_path):
            pass  # Path is already correct
        else:
            print(f"Error: Model file not found at {model_path} or {model_path}.zip")
            sys.exit(1)
        
        # Run evaluation
        metrics, trades_log = evaluate_agent(model_path, args.config)
        
        # Run walk-forward analysis
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting walk-forward analysis...")
        walk_forward_results = walk_forward_analysis(args.config)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluation completed!")
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()