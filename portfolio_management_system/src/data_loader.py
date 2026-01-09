"""
Data loading module for portfolio management system
Handles fetching market data from various sources
"""

import pandas as pd
import yfinance as yf
from typing import Dict, List
from datetime import datetime, timedelta


class DataLoader:
    """
    Class responsible for loading market data
    """
    
    def __init__(self):
        pass
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current market prices for a list of symbols
        
        Args:
            symbols: List of stock symbols/tickers
            
        Returns:
            Dictionary mapping symbols to their current prices
        """
        prices = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    prices[symbol] = hist['Close'].iloc[-1]
                else:
                    # Fallback to previous close if today's data not available
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        prices[symbol] = hist['Close'].iloc[-1]
                    else:
                        prices[symbol] = 0.0
            except Exception as e:
                print(f"Error fetching price for {symbol}: {e}")
                prices[symbol] = 0.0
        
        return prices
    
    def get_historical_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical price data for a list of symbols
        
        Args:
            symbols: List of stock symbols/tickers
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Download data for all symbols at once
            data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
            
            # If only one symbol, yfinance returns differently structured data
            if len(symbols) == 1:
                symbol = symbols[0]
                df = data[['Close']].copy()
                df.columns = [symbol]
                return df
            else:
                # Multiple symbols case
                df = pd.DataFrame()
                for symbol in symbols:
                    if isinstance(data.columns, pd.MultiIndex):
                        # Multi-index case (multiple stocks)
                        df[symbol] = data[symbol]['Close']
                    else:
                        # Single stock case
                        df[symbol] = data[symbol]
                return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def get_price_changes(self, symbols: List[str], period: str = "1mo") -> Dict[str, float]:
        """
        Get price changes for symbols over a specific period
        
        Args:
            symbols: List of stock symbols/tickers
            period: Time period (e.g., '1mo', '3mo', '1y')
            
        Returns:
            Dictionary mapping symbols to their percentage change
        """
        changes = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                
                if len(hist) >= 2:
                    start_price = hist['Close'].iloc[0]
                    end_price = hist['Close'].iloc[-1]
                    change_pct = ((end_price - start_price) / start_price) * 100
                    changes[symbol] = change_pct
                else:
                    changes[symbol] = 0.0
            except Exception as e:
                print(f"Error calculating price change for {symbol}: {e}")
                changes[symbol] = 0.0
        
        return changes