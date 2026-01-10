import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class IndicatorSelector(nn.Module):
    """
    Neural network that adaptively selects technical indicators based on market conditions.
    Uses market state features to decide which indicators to focus on.
    """
    
    def __init__(self, num_market_features: int = 5, num_indicators: int = 10, hidden_size: int = 64):
        super(IndicatorSelector, self).__init__()
        
        self.num_market_features = num_market_features
        self.num_indicators = num_indicators
        
        # Market state encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(num_market_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Indicator importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_indicators),
            nn.Softmax(dim=-1)  # Softmax to get importance weights
        )
        
        # Indicator names mapping
        self.indicator_names = [
            'RSI', 'MACD', 'Bollinger_Bands', 'Volume_Ratio', 
            'Volatility', 'Momentum', 'Trend', 'Returns', 
            'HL_Spread', 'Price_Position'
        ]
        
    def forward(self, market_state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict indicator importance weights
        
        Args:
            market_state: Tensor of market state features [batch_size, num_market_features]
            
        Returns:
            indicator_weights: Tensor of indicator importance weights [batch_size, num_indicators]
        """
        encoded_state = self.market_encoder(market_state)
        indicator_weights = self.importance_predictor(encoded_state)
        return indicator_weights
    
    def get_selected_indicators(self, market_state: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k most important indicators for current market state
        
        Args:
            market_state: Current market state features
            top_k: Number of top indicators to return
            
        Returns:
            List of tuples (indicator_name, importance_weight)
        """
        with torch.no_grad():
            weights = self.forward(market_state.unsqueeze(0))  # Add batch dimension
            weights = weights.squeeze(0)  # Remove batch dimension
            
            # Get top-k indicators
            top_indices = torch.topk(weights, top_k).indices
            top_weights = weights[top_indices]
            
            selected = [(self.indicator_names[i], weight.item()) 
                       for i, weight in zip(top_indices, top_weights)]
            
            return sorted(selected, key=lambda x: x[1], reverse=True)

class AdaptiveFeatureExtractor:
    """
    Wrapper class that combines the indicator selector with feature extraction
    """
    
    def __init__(self, indicator_selector: IndicatorSelector, base_features: List[str]):
        self.selector = indicator_selector
        self.base_features = base_features  # List of all possible features
        self.feature_mapping = {name: idx for idx, name in enumerate(base_features)}
        
    def extract_adaptive_features(self, 
                                full_features: pd.DataFrame, 
                                market_state: Dict[str, float]) -> pd.DataFrame:
        """
        Extract features based on current market conditions
        
        Args:
            full_features: DataFrame with all possible features
            market_state: Current market state features
            
        Returns:
            DataFrame with selected features based on market conditions
        """
        # Convert market state to tensor
        market_tensor = torch.tensor([
            market_state['volatility_level'],
            market_state['trend_strength'], 
            market_state['momentum_strength'],
            market_state['volume_activity'],
            0 if market_state['market_regime'] == 'calm_bull' else 1 if market_state['market_regime'] == 'calm_neutral' else 2
        ], dtype=torch.float32)
        
        # Get selected indicators
        selected_indicators = self.selector.get_selected_indicators(market_tensor, top_k=7)
        
        # Map indicator names to feature column names
        feature_cols = []
        for indicator_name, weight in selected_indicators:
            # Map indicator name to actual column name
            if indicator_name == 'RSI':
                feature_cols.extend(['rsi'])
            elif indicator_name == 'MACD':
                feature_cols.extend(['macd', 'macd_signal', 'macd_histogram'])
            elif indicator_name == 'Bollinger_Bands':
                feature_cols.extend(['bb_position', 'bb_width'])
            elif indicator_name == 'Volume_Ratio':
                feature_cols.extend(['volume_ratio'])
            elif indicator_name == 'Volatility':
                feature_cols.extend(['volatility'])
            elif indicator_name == 'Momentum':
                feature_cols.extend(['momentum'])
            elif indicator_name == 'Trend':
                feature_cols.extend(['trend_sma'])
            elif indicator_name == 'Returns':
                feature_cols.extend(['returns'])
            elif indicator_name == 'HL_Spread':
                feature_cols.extend(['hl_spread'])
            elif indicator_name == 'Price_Position':
                feature_cols.extend(['high_low_pct'])
        
        # Remove duplicates and ensure they exist in the dataframe
        feature_cols = list(set(feature_cols))
        available_cols = [col for col in feature_cols if col in full_features.columns]
        
        return full_features[available_cols].copy()

class MarketRegimeClassifier:
    """
    Classifier to categorize market regimes based on technical features
    """
    
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the classifier on market regime data
        
        Args:
            X: Feature matrix
            y: Regime labels
        """
        # Encode regime labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.rf_classifier.fit(X, y_encoded)
        self.is_trained = True
        
    def predict_regime(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict market regime for given features
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted regime labels
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before prediction")
        
        y_pred_encoded = self.rf_classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def get_regime_importance(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature importance for regime classification
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained:
            raise ValueError("Classifier must be trained before getting importance")
        
        feature_names = X.columns.tolist()
        importances = self.rf_classifier.feature_importances_
        
        return dict(zip(feature_names, importances))