import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """
    Advanced data processor for financial time series data.
    Handles OHLCV data, technical indicators, and market state features.
    """
    
    def __init__(self):
        self.indicator_params = {}
        
    def fetch_data(self, symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """
        Fetch historical market data using yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'TSLA')
            period: Time period ('1mo', '3mo', '6mo', '1y', '2y', etc.)
            interval: Data interval ('1d', '1h', '15m', etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        return data
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators for the dataset
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Copy original dataframe
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / df['bb_width']
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)  # Annualized
        
        # High-Low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Price position in range
        df['high_low_pct'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Momentum
        df['momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Trend indicators
        df['trend_sma'] = (df['Close'] - df['sma_20']) / df['sma_20']
        
        # Market regime indicators
        df['market_regime'] = self._calculate_market_regime(df)
        
        return df.dropna()
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate market regime based on volatility and trend
        """
        volatility_regime = pd.cut(df['volatility'], bins=3, labels=['low', 'medium', 'high'])
        trend_regime = pd.cut(df['trend_sma'], bins=3, labels=['bearish', 'neutral', 'bullish'])
        
        # Combine volatility and trend regimes
        regime_map = {
            ('low', 'bullish'): 'calm_bull',
            ('low', 'neutral'): 'calm_neutral', 
            ('low', 'bearish'): 'calm_bear',
            ('medium', 'bullish'): 'trending_bull',
            ('medium', 'neutral'): 'choppy',
            ('medium', 'bearish'): 'trending_bear',
            ('high', 'bullish'): 'volatile_bull',
            ('high', 'neutral'): 'volatile_neutral',
            ('high', 'bearish'): 'volatile_bear'
        }
        
        regime = []
        for vol, trend in zip(volatility_regime, trend_regime):
            regime.append(regime_map.get((vol, trend), 'unknown'))
        
        return pd.Series(regime, index=df.index)
    
    def prepare_features(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Prepare normalized features for the RL agent
        
        Args:
            df: DataFrame with technical indicators
            feature_columns: Specific columns to include as features
            
        Returns:
            Normalized DataFrame ready for ML models
        """
        if feature_columns is None:
            # Default feature set
            feature_columns = [
                'returns', 'rsi', 'macd', 'macd_signal', 'bb_position', 
                'volume_ratio', 'volatility', 'hl_spread', 'momentum', 
                'trend_sma'
            ]
        
        # Select features
        features_df = df[feature_columns].copy()
        
        # Normalize features
        for col in features_df.columns:
            mean_val = features_df[col].mean()
            std_val = features_df[col].std()
            if std_val != 0:
                features_df[col] = (features_df[col] - mean_val) / std_val
            else:
                features_df[col] = 0  # Handle constant values
        
        return features_df
    
    def get_market_state_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract current market state features for adaptive indicator selection
        
        Args:
            df: Current market data
            
        Returns:
            Dictionary of market state features
        """
        latest = df.iloc[-1]
        
        return {
            'volatility_level': latest['volatility'],
            'trend_strength': abs(latest['trend_sma']),
            'momentum_strength': abs(latest['momentum']),
            'volume_activity': latest['volume_ratio'],
            'market_regime': latest['market_regime']
        }