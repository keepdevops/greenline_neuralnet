import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

class MarketSentimentAnalyzer:
    def __init__(self):
        self.sentiment_indicators = {}
        self.sentiment_history = []
        
    def calculate_sentiment_score(self, data):
        """Calculate overall market sentiment score from -1 (extremely bearish) to 1 (extremely bullish)"""
        # Initialize component scores
        volatility_score = self._calculate_volatility_score(data)
        momentum_score = self._calculate_momentum_score(data)
        volume_score = self._calculate_volume_score(data)
        distribution_score = self._calculate_distribution_score(data)
        
        # Combine scores with weights
        sentiment_score = (
            0.3 * volatility_score +    # Volatility indicates fear/greed
            0.3 * momentum_score +      # Price momentum shows trend strength
            0.2 * volume_score +        # Volume shows conviction
            0.2 * distribution_score    # Return distribution shows market psychology
        )
        
        # Store indicators for analysis
        self.sentiment_indicators = {
            'volatility_score': volatility_score,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'distribution_score': distribution_score,
            'overall_sentiment': sentiment_score
        }
        
        # Add to history
        self.sentiment_history.append(self.sentiment_indicators)
        
        return sentiment_score
    
    def _calculate_volatility_score(self, data, window=20):
        """Calculate sentiment score based on volatility"""
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Compare current volatility to recent history
        current_vol = volatility.iloc[-1]
        historical_vol = volatility.mean()
        
        # Convert to score: high volatility -> negative sentiment
        vol_ratio = current_vol / historical_vol
        score = 1 - min(vol_ratio, 2) / 2  # Cap at -1
        
        return score
    
    def _calculate_momentum_score(self, data, windows=[5, 10, 20]):
        """Calculate sentiment score based on price momentum"""
        scores = []
        
        for window in windows:
            # Calculate moving averages
            ma = data['close'].rolling(window=window).mean()
            # Compare current price to MA
            price_ratio = data['close'].iloc[-1] / ma.iloc[-1]
            # Convert to score
            score = np.clip(price_ratio - 1, -1, 1)
            scores.append(score)
        
        # Weight shorter-term momentum more heavily
        weights = [0.5, 0.3, 0.2]
        return np.average(scores, weights=weights)
    
    def _calculate_volume_score(self, data, window=20):
        """Calculate sentiment score based on volume analysis"""
        # Calculate relative volume
        avg_volume = data['vol'].rolling(window=window).mean()
        rel_volume = data['vol'] / avg_volume
        
        # Calculate price-volume correlation
        returns = data['close'].pct_change()
        volume_score = np.corrcoef(returns.iloc[-window:], 
                                 rel_volume.iloc[-window:])[0, 1]
        
        return np.clip(volume_score, -1, 1)
    
    def _calculate_distribution_score(self, data, window=50):
        """Calculate sentiment score based on return distribution"""
        returns = data['close'].pct_change().iloc[-window:]
        
        # Calculate statistical moments
        skewness = skew(returns.dropna())
        kurt = kurtosis(returns.dropna())
        
        # Convert to score
        # Negative skewness indicates fear (more large drops)
        # High kurtosis indicates uncertainty (fat tails)
        score = -0.7 * np.clip(skewness, -1, 1) - 0.3 * min(kurt / 10, 1)
        
        return np.clip(score, -1, 1)
    
    def get_sentiment_signals(self):
        """Get trading signals based on sentiment analysis"""
        if not self.sentiment_indicators:
            return None
        
        sentiment = self.sentiment_indicators['overall_sentiment']
        
        signals = {
            'sentiment_level': 'neutral',
            'trading_bias': 'hold',
            'confidence': 'low'
        }
        
        # Determine sentiment level
        if sentiment > 0.5:
            signals['sentiment_level'] = 'very bullish'
            signals['trading_bias'] = 'strong buy'
            signals['confidence'] = 'high'
        elif sentiment > 0.2:
            signals['sentiment_level'] = 'bullish'
            signals['trading_bias'] = 'buy'
            signals['confidence'] = 'medium'
        elif sentiment < -0.5:
            signals['sentiment_level'] = 'very bearish'
            signals['trading_bias'] = 'strong sell'
            signals['confidence'] = 'high'
        elif sentiment < -0.2:
            signals['sentiment_level'] = 'bearish'
            signals['trading_bias'] = 'sell'
            signals['confidence'] = 'medium'
        
        return signals
    
    def get_sentiment_report(self):
        """Generate a detailed sentiment analysis report"""
        if not self.sentiment_indicators:
            return "No sentiment analysis performed yet."
        
        signals = self.get_sentiment_signals()
        
        report = f"""Market Sentiment Analysis Report
=============================
Overall Sentiment Score: {self.sentiment_indicators['overall_sentiment']:.2f}
Sentiment Level: {signals['sentiment_level']}
Trading Bias: {signals['trading_bias']}
Signal Confidence: {signals['confidence']}

Component Scores:
---------------
Volatility: {self.sentiment_indicators['volatility_score']:.2f}
Momentum: {self.sentiment_indicators['momentum_score']:.2f}
Volume: {self.sentiment_indicators['volume_score']:.2f}
Distribution: {self.sentiment_indicators['distribution_score']:.2f}
"""
        return report 