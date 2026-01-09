"""
Main portfolio management system module
Implements the core logic for portfolio monitoring, rebalancing, and risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
import json
import yaml
from .data_loader import DataLoader
from .rebalancing_engine import RebalancingEngine
from .risk_manager import RiskManager


class PortfolioManager:
    """
    Main class for managing the investment portfolio
    """
    
    def __init__(self, config_file: str = 'config/default_config.json'):
        """
        Initialize the portfolio manager with configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.data_loader = DataLoader()
        self.rebalancing_engine = RebalancingEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Portfolio state
        self.portfolio_weights = {}
        self.portfolio_value = 0.0
        self.last_rebalance_date = None
        self.transaction_costs = 0.0
        
        # Initialize portfolio
        self._initialize_portfolio()
    
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from file"""
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                return json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                return yaml.safe_load(f)
        raise ValueError(f"Unsupported config file format: {config_file}")
    
    def _initialize_portfolio(self):
        """Initialize portfolio with target weights"""
        assets = self.config['assets']
        target_weights = self.config.get('target_weights', {})
        
        # If no target weights provided, distribute equally
        if not target_weights:
            equal_weight = 1.0 / len(assets)
            self.portfolio_weights = {asset: equal_weight for asset in assets}
        else:
            self.portfolio_weights = target_weights.copy()
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Calculate current portfolio weights based on market prices
        
        Returns:
            Dictionary with current asset weights
        """
        # Get current prices for all assets
        current_prices = self.data_loader.get_current_prices(list(self.portfolio_weights.keys()))
        
        # Calculate market values
        market_values = {}
        total_value = 0.0
        
        for asset, weight in self.portfolio_weights.items():
            if asset in current_prices:
                market_values[asset] = current_prices[asset]
                total_value += current_prices[asset]
        
        # Calculate current weights
        current_weights = {}
        for asset, value in market_values.items():
            current_weights[asset] = value / total_value if total_value > 0 else 0.0
        
        return current_weights
    
    def should_rebalance(self) -> bool:
        """
        Check if portfolio needs rebalancing based on deviation threshold
        
        Returns:
            True if rebalancing is needed, False otherwise
        """
        current_weights = self.get_current_weights()
        target_weights = self.portfolio_weights
        
        # Check if minimum rebalance interval has passed
        min_interval = self.config.get('min_rebalance_interval_days', 7)
        if self.last_rebalance_date:
            days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
            if days_since_rebalance < min_interval:
                return False
        
        # Check if any asset deviates beyond threshold
        deviation_threshold = self.config.get('rebalance_threshold', 0.05)  # 5%
        
        for asset in target_weights:
            current_weight = current_weights.get(asset, 0.0)
            target_weight = target_weights[asset]
            
            if abs(current_weight - target_weight) > deviation_threshold:
                return True
        
        return False
    
    def execute_rebalance(self):
        """
        Execute portfolio rebalancing based on current market conditions
        """
        if not self.should_rebalance():
            return
        
        # Calculate rebalancing signals
        signals = self.rebalancing_engine.calculate_rebalance_signals(
            self.portfolio_weights,
            self.get_current_weights()
        )
        
        # Apply risk management filters
        filtered_signals = self.risk_manager.apply_filters(signals)
        
        # Execute the rebalancing
        self._apply_rebalance_signals(filtered_signals)
        
        # Update last rebalance date
        self.last_rebalance_date = datetime.now()
    
    def _apply_rebalance_signals(self, signals: Dict[str, float]):
        """
        Apply rebalancing signals to update portfolio weights
        
        Args:
            signals: Dictionary with rebalancing signals for each asset
        """
        for asset, signal in signals.items():
            if asset in self.portfolio_weights:
                # Update portfolio weight based on signal
                new_weight = self.portfolio_weights[asset] + signal
                # Ensure weight stays within bounds
                max_weight = self.config.get('max_asset_weight', 0.20)  # 20%
                min_weight = self.config.get('min_asset_weight', 0.01)  # 1%
                
                self.portfolio_weights[asset] = max(min_weight, min(max_weight, new_weight))
    
    def get_portfolio_performance(self) -> Dict:
        """
        Calculate portfolio performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        current_weights = self.get_current_weights()
        
        # Calculate total portfolio value
        current_prices = self.data_loader.get_current_prices(list(current_weights.keys()))
        total_value = sum(current_prices.values())
        
        return {
            'total_value': total_value,
            'current_weights': current_weights,
            'last_rebalance_date': self.last_rebalance_date,
            'transaction_costs': self.transaction_costs
        }
    
    def run(self):
        """
        Main execution loop for the portfolio manager
        """
        print("Starting portfolio management system...")
        
        while True:
            try:
                # Monitor portfolio
                current_weights = self.get_current_weights()
                print(f"Current portfolio weights: {current_weights}")
                
                # Check if rebalancing is needed
                if self.should_rebalance():
                    print("Rebalancing signal detected. Executing rebalance...")
                    self.execute_rebalance()
                    print(f"Rebalanced portfolio weights: {self.portfolio_weights}")
                else:
                    print("No rebalancing needed at this time.")
                
                # Wait before next check (in a real system, this would be scheduled)
                break  # For demo purposes, we'll run once
                
            except Exception as e:
                print(f"Error in portfolio management: {e}")
                break
    
    def get_historical_performance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical performance data for backtesting
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical performance data
        """
        return self.data_loader.get_historical_data(
            list(self.portfolio_weights.keys()),
            start_date,
            end_date
        )