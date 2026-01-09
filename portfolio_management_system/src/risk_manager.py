"""
Risk management module for portfolio management system
Implements risk controls and safety measures
"""

from typing import Dict, List
import numpy as np


class RiskManager:
    """
    Class responsible for implementing risk management controls
    """
    
    def __init__(self, config: dict):
        """
        Initialize risk manager with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.critical_drop_threshold = config.get('critical_drop_threshold', -30.0)  # -30%
        self.max_asset_weight = config.get('max_asset_weight', 0.20)  # 20%
        self.min_asset_weight = config.get('min_asset_weight', 0.01)  # 1%
    
    def apply_filters(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        Apply risk management filters to rebalancing signals
        
        Args:
            signals: Original rebalancing signals
            
        Returns:
            Filtered signals with risk controls applied
        """
        filtered_signals = signals.copy()
        
        # Check for critical drops in asset prices
        recent_changes = self._get_recent_price_changes()
        
        for asset, signal in signals.items():
            # Don't buy assets that have dropped critically
            if asset in recent_changes:
                if recent_changes[asset] <= self.critical_drop_threshold and signal > 0:
                    print(f"Risk filter: Preventing purchase of {asset} due to critical drop ({recent_changes[asset]:.2f}%)")
                    filtered_signals[asset] = 0.0  # Cancel buy signal
        
        # Apply weight limits after rebalancing
        current_weights = self._get_current_weights()  # This would come from the portfolio manager
        proposed_weights = self._calculate_proposed_weights(current_weights, filtered_signals)
        
        # Adjust signals to respect weight limits
        for asset in filtered_signals:
            if asset in proposed_weights:
                if proposed_weights[asset] > self.max_asset_weight:
                    # Reduce buy signals or increase sell signals for overweighted assets
                    excess = proposed_weights[asset] - self.max_asset_weight
                    filtered_signals[asset] -= excess
                elif proposed_weights[asset] < self.min_asset_weight and filtered_signals[asset] < 0:
                    # Reduce sell signals for underweighted assets
                    deficit = self.min_asset_weight - proposed_weights[asset]
                    filtered_signals[asset] += min(deficit, abs(filtered_signals[asset]))
        
        return filtered_signals
    
    def validate_position_size(self, asset: str, target_weight: float) -> bool:
        """
        Validate if a position size is acceptable based on risk limits
        
        Args:
            asset: Asset symbol
            target_weight: Proposed weight for the asset
            
        Returns:
            True if position size is valid, False otherwise
        """
        if target_weight > self.max_asset_weight:
            print(f"Risk validation failed: {asset} weight {target_weight:.2%} exceeds maximum {self.max_asset_weight:.2%}")
            return False
        if target_weight < self.min_asset_weight and target_weight != 0:
            print(f"Risk validation failed: {asset} weight {target_weight:.2%} below minimum {self.min_asset_weight:.2%}")
            return False
        return True
    
    def _get_recent_price_changes(self) -> Dict[str, float]:
        """
        Get recent price changes for assets (would integrate with data loader in practice)
        
        Returns:
            Dictionary with recent price changes for each asset
        """
        # This is a placeholder - in practice, this would call the data loader
        # For now, returning empty dict to avoid errors
        return {}
    
    def _get_current_weights(self) -> Dict[str, float]:
        """
        Get current portfolio weights (placeholder)
        
        Returns:
            Dictionary with current weights
        """
        # Placeholder implementation
        return {}
    
    def _calculate_proposed_weights(self, current_weights: Dict[str, float], 
                                   signals: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate what weights would be after applying signals
        
        Args:
            current_weights: Current asset weights
            signals: Rebalancing signals to apply
            
        Returns:
            Dictionary with proposed weights after rebalancing
        """
        proposed_weights = {}
        
        for asset in current_weights:
            new_weight = current_weights[asset] + signals.get(asset, 0.0)
            # Ensure weights stay between 0 and 1
            proposed_weights[asset] = max(0.0, min(1.0, new_weight))
        
        # Renormalize weights to sum to 1
        total_weight = sum(proposed_weights.values())
        if total_weight > 0:
            for asset in proposed_weights:
                proposed_weights[asset] /= total_weight
        
        return proposed_weights