"""
Quantitative Trading Agent Package
An advanced reinforcement learning system for autonomous stock trading.
"""

__version__ = "1.0.0"
__author__ = "Quant Trading AI"

from .utils.data_processor import DataProcessor
from .environments.trading_env import TradingEnv

# Import classes that don't require PyTorch at startup
__all__ = [
    'DataProcessor',
    'TradingEnv'
]

# Delay imports that require PyTorch until they're needed
def get_indicator_selector():
    from .indicators.indicator_selector import IndicatorSelector
    return IndicatorSelector

def get_adaptive_feature_extractor():
    from .indicators.indicator_selector import AdaptiveFeatureExtractor
    return AdaptiveFeatureExtractor

def get_adaptive_ppo_agent():
    from .models.ppo_agent import AdaptivePPOAgent
    return AdaptivePPOAgent