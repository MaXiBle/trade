import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance
    """
    df = yf.download(symbol, start=start_date, end=end_date)
    
    # Handle multi-index columns if present (common with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the multi-index by taking the first level (price type) and second level (ticker)
        df.columns = [col[0].lower() for col in df.columns.values]  # Just take the price type (Open, High, etc.)
    
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    
    return df

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators with proper lagging to avoid lookahead bias
    """
    df = df.copy()
    
    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(window=5).mean().shift(1)  # Lagged by 1
    df['sma_20'] = df['close'].rolling(window=20).mean().shift(1)
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff().shift(1)  # Lagged by 1
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12).mean().shift(1)  # Lagged by 1
    exp2 = df['close'].ewm(span=26).mean().shift(1)
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean().shift(1)  # Lagged by 1
    rolling_std = df['close'].rolling(window=20).std().shift(1)
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_middle'] = rolling_mean
    
    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(2))  # Lagged close
    low_close = np.abs(df['low'] - df['close'].shift(2))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    
    # Volatility (rolling standard deviation)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std().shift(1)  # Lagged by 1
    
    # Price-based features (lagged)
    df['price_change_pct'] = df['close'].pct_change().shift(1)
    df['volume_change_pct'] = df['volume'].pct_change().shift(1)
    
    # Normalize indicators to [0, 1] range to prevent lookahead bias
    def normalize_column(col):
        min_val = col.rolling(window=60, min_periods=1).min()
        max_val = col.rolling(window=60, min_periods=1).max()
        return (col - min_val) / (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Apply normalization to relevant indicators
    indicator_cols = ['sma_5', 'sma_20', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr', 'volatility']
    for col in indicator_cols:
        if col in df.columns:
            df[f'{col}_norm'] = normalize_column(df[col])
    
    return df

def prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare training and testing datasets with proper time separation
    """
    # Fetch raw data
    train_df = fetch_data(
        config['symbol'], 
        config['train_start_date'], 
        config['train_end_date']
    )
    test_df = fetch_data(
        config['symbol'], 
        config['test_start_date'], 
        config['test_end_date']
    )
    
    # Calculate technical indicators (with lagging to prevent lookahead bias)
    train_df = calculate_technical_indicators(train_df)
    test_df = calculate_technical_indicators(test_df)
    
    # Select features for the model
    feature_columns = config['features']
    missing_features = [col for col in feature_columns if col not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    # Fill NaN values with 0 (for early periods where indicators can't be calculated)
    train_df[feature_columns] = train_df[feature_columns].fillna(0)
    test_df[feature_columns] = test_df[feature_columns].fillna(0)
    
    return train_df, test_df

def get_feature_names(config: Dict) -> list:
    """
    Get the list of feature names for the environment
    """
    return config['features']