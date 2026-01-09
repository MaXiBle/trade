"""
Rebalancing engine module for portfolio management system
Calculates rebalancing signals based on current vs target weights
"""

from typing import Dict, List
import numpy as np


class RebalancingEngine:
    """
    Class responsible for calculating rebalancing signals
    """
    
    def __init__(self, config: dict):
        """
        Initialize rebalancing engine with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def calculate_rebalance_signals(self, target_weights: Dict[str, float], 
                                  current_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate rebalancing signals based on deviation from target weights
        
        Args:
            target_weights: Target weights for each asset
            current_weights: Current weights for each asset
            
        Returns:
            Dictionary with rebalancing signals (positive = buy, negative = sell)
        """
        signals = {}
        
        # Calculate deviation from target for each asset
        for asset in target_weights:
            target_weight = target_weights[asset]
            current_weight = current_weights.get(asset, 0.0)
            
            # Calculate the difference between current and target
            deviation = current_weight - target_weight
            
            # The signal is proportional to the deviation
            # Positive deviation means overweighted (need to sell)
            # Negative deviation means underweighted (need to buy)
            signals[asset] = -deviation  # Negative because positive deviation needs selling
        
        # Normalize signals so they sum to zero (conservation of capital)
        total_signal = sum(signals.values())
        
        # Adjust signals to ensure they sum to approximately zero
        # This is important to maintain the total portfolio value
        for asset in signals:
            signals[asset] -= total_signal / len(signals)
        
        return signals
    
    def apply_transaction_costs(self, signals: Dict[str, float], prices: Dict[str, float]) -> Dict[str, float]:
        """
        Apply transaction costs to rebalancing signals
        
        Args:
            signals: Original rebalancing signals
            prices: Current prices for each asset
            
        Returns:
            Adjusted signals accounting for transaction costs
        """
        adjusted_signals = signals.copy()
        
        # Get transaction cost from config (default 0.1% per transaction)
        transaction_cost_rate = self.config.get('transaction_cost_rate', 0.001)
        
        for asset, signal in signals.items():
            if asset in prices:
                # Reduce the signal magnitude based on transaction costs
                # For simplicity, we reduce both buy and sell signals by the same rate
                adjusted_signals[asset] *= (1 - transaction_cost_rate)
        
        return adjusted_signals