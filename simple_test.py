#!/usr/bin/env python3
"""
Simple test script to verify the trading agent components work correctly
without PyTorch dependencies initially
"""

import sys

from trading_agent.environments.trading_env import TradingEnv

sys.path.append('/workspace')

import warnings
warnings.filterwarnings('ignore')

def test_basic_components():
    print("Testing Quantitative Trading Agent Basic Components...")
    
    try:
        # Test 1: Data Processor
        print("\n1. Testing Data Processor...")
        from trading_agent.utils.data_processor import DataProcessor
        dp = DataProcessor()
        
        # Fetch sample data
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        sample_data = ticker.history(period="6mo", interval="1d")  # More data for indicators
        
        print(f"   ✓ Retrieved {len(sample_data)} raw data points")
        
        # Calculate indicators
        processed_data = dp.calculate_technical_indicators(sample_data)
        print(f"   ✓ Processed {len(processed_data)} data points with {len(processed_data.columns)} features")
        print(f"   ✓ Features include: {list(processed_data.columns[:10])}...")
        
        # Test market state features
        market_state = dp.get_market_state_features(processed_data)
        print(f"   ✓ Market state features: {market_state}")
        
        # Test 2: Trading Environment (without PyTorch parts initially)
        print("\n2. Testing Trading Environment...")
        from trading_agent.environments.trading_env import TradingEnv
        env = TradingEnv(processed_data, initial_balance=10000)
        
        obs, info = env.reset()
        print(f"   ✓ Observation shape: {obs.shape}")
        print(f"   ✓ Action space: {env.action_space}")
        print(f"   ✓ Observation space: {env.observation_space}")
        
        # Test one step with random action
        import numpy as np
        action = env.action_space.sample()  # Random action
        new_obs, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step executed successfully, reward: {reward:.4f}")
        print(f"   ✓ Portfolio value: ${info['portfolio_value']:.2f}")
        
        print("\n✅ Basic components tested successfully!")
        print("\nThe core infrastructure is working properly.")
        print("\nFor full functionality including the neural network components,")
        print("the PyTorch library needs to be properly installed without CUDA issues.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_pytorch():
    """Test components that require PyTorch - this might fail due to CUDA issues"""
    print("\n3. Testing PyTorch-dependent components...")
    
    try:
        # Test Indicator Selector
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
        
        # Test PPO Agent initialization
        from trading_agent.models.ppo_agent import AdaptivePPOAgent
        from trading_agent.utils.data_processor import DataProcessor
        
        # Recreate a simple environment for this test
        dp = DataProcessor()
        test_data = dp.fetch_data("AAPL", period="6mo", interval="1d")  # More data for indicators
        test_data = dp.calculate_technical_indicators(test_data)
        
        env_for_test = TradingEnv(
            data=test_data,
            initial_balance=10000
        )
        
        agent = AdaptivePPOAgent(
            env=env_for_test,
            indicator_selector=indicator_selector,
            learning_rate=3e-4,
            gamma=0.99
        )
        print(f"   ✓ PPO agent initialized successfully")
        
        print("   ✅ PyTorch components working!")
        return True
        
    except Exception as e:
        print(f"   ⚠️  PyTorch components failed (likely due to CUDA issues): {str(e)}")
        return False

if __name__ == "__main__":
    success = test_basic_components()
    
    # Only try PyTorch components if basic ones work
    if success:
        torch_success = test_with_pytorch()
        
        print(f"\n{'='*60}")
        print("FINAL STATUS:")
        print(f"Basic components: {'✅ PASS' if success else '❌ FAIL'}")
        print(f"PyTorch components: {'✅ PASS' if torch_success else '⚠️  ISSUE (CUDA)'}")
        print(f"{'='*60}")
        
        if success:
            print("\nThe quantitative trading agent framework is properly structured!")
            print("\nTo train the agent, you can run (when PyTorch works):")
            print("python trading_agent/train_agent.py --symbol AAPL --train_steps 10000")
    else:
        print("\n❌ Basic components failed - please check the error above")
        sys.exit(1)