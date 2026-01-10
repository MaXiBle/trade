#!/usr/bin/env python3
"""
Simple test script to verify the trading agent components work correctly
"""

import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import warnings
warnings.filterwarnings('ignore')

def test_components():
    print("Testing Quantitative Trading Agent Components...")
    
    try:
        # Test 1: Data Processor
        print("\n1. Testing Data Processor...")
        from trading_agent.utils.data_processor import DataProcessor
        dp = DataProcessor()
        
        # Fetch sample data
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        sample_data = ticker.history(period="2y", interval="1d")
        
        # Calculate indicators
        processed_data = dp.calculate_technical_indicators(sample_data)
        print(f"   ✓ Processed {len(processed_data)} data points with {len(processed_data.columns)} features")
        print(f"   ✓ Features include: {list(processed_data.columns[:10])}...")
        
        # Test 2: Indicator Selector
        print("\n2. Testing Indicator Selector...")
        from trading_agent.indicators.indicator_selector import IndicatorSelector
        indicator_selector = IndicatorSelector()
        
        # Test forward pass
        import torch
        sample_market_state = torch.tensor([
            0.2,  # volatility_level
            0.1,  # trend_strength 
            0.05, # momentum_strength
            1.2,  # volume_activity
            0     # market_regime (encoded)
        ], dtype=torch.float32)
        
        weights = indicator_selector(sample_market_state.unsqueeze(0))
        print(f"   ✓ Indicator weights shape: {weights.shape}")
        
        # Get selected indicators
        selected = indicator_selector.get_selected_indicators(sample_market_state, top_k=3)
        print(f"   ✓ Top 3 selected indicators: {selected}")
        
        # Test 3: Trading Environment
        print("\n3. Testing Trading Environment...")
        from trading_agent.environments.trading_env import TradingEnv
        env = TradingEnv(processed_data, initial_balance=10000)
        
        obs, info = env.reset()
        print(f"   ✓ Observation shape: {obs.shape}")
        print(f"   ✓ Action space: {env.action_space}")
        print(f"   ✓ Observation space: {env.observation_space}")
        
        # Test one step
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step executed successfully, reward: {reward:.4f}")
        
        # Test 4: PPO Agent initialization
        print("\n4. Testing PPO Agent Initialization...")
        from trading_agent.models.ppo_agent import AdaptivePPOAgent
        agent = AdaptivePPOAgent(
            env=env,
            indicator_selector=indicator_selector,
            learning_rate=3e-4,
            gamma=0.99
        )
        print(f"   ✓ PPO agent initialized successfully")
        
        # Test 5: Prediction
        action, _ = agent.predict(obs)
        print(f"   ✓ Prediction made successfully: {action}")
        
        print("\n✅ All components tested successfully!")
        print("\nThe quantitative trading agent system is properly configured.")
        print("\nTo train the agent, run:")
        print("python trading_agent/train_agent.py --symbol AAPL --train_steps 10000")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_components()
    if not success:
        sys.exit(1)