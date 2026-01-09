"""
Unit tests for the portfolio management system
"""

import unittest
import os
import sys
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from src.portfolio_manager import PortfolioManager


class TestPortfolioManager(unittest.TestCase):
    """
    Unit tests for PortfolioManager class
    """
    
    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        # Use a simple config for testing
        self.test_config = {
            'assets': ['AAPL', 'MSFT'],
            'target_weights': {
                'AAPL': 0.5,
                'MSFT': 0.5
            },
            'rebalance_threshold': 0.05,
            'min_rebalance_interval_days': 7,
            'max_asset_weight': 0.20,
            'min_asset_weight': 0.01,
            'critical_drop_threshold': -30.0,
            'transaction_cost_rate': 0.001
        }
        
        # Save test config to file
        import json
        with open('/tmp/test_config.json', 'w') as f:
            json.dump(self.test_config, f)
        
        self.manager = PortfolioManager(config_file='/tmp/test_config.json')
    
    def tearDown(self):
        """
        Clean up after each test method
        """
        import os
        if os.path.exists('/tmp/test_config.json'):
            os.remove('/tmp/test_config.json')
    
    def test_initialization(self):
        """
        Test that PortfolioManager initializes correctly
        """
        self.assertEqual(set(self.manager.config['assets']), {'AAPL', 'MSFT'})
        self.assertEqual(self.manager.portfolio_weights, {'AAPL': 0.5, 'MSFT': 0.5})
    
    def test_get_current_weights(self):
        """
        Test getting current portfolio weights
        """
        weights = self.manager.get_current_weights()
        self.assertIsInstance(weights, dict)
        self.assertEqual(set(weights.keys()), {'AAPL', 'MSFT'})
    
    def test_should_rebalance(self):
        """
        Test rebalancing decision logic
        """
        # Initially, should not rebalance since we just started
        should_rebalance = self.manager.should_rebalance()
        # This might be true or false depending on current market conditions
        # So we just test that it returns a boolean
        self.assertIsInstance(should_rebalance, bool)


if __name__ == '__main__':
    unittest.main()