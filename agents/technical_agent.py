from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from .base_agent import BaseAgent

class TechnicalAgent(BaseAgent):
    """
    Rule-based agent for technical analysis of stocks.
    """
    
    def __init__(self):
        super().__init__(
            name="Technical Analysis Agent",
            description="Analyzes stock price patterns and technical indicators"
        )
        self.required_data = ['history']
    
    async def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform technical analysis on the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters
            
        Returns:
            Dict containing technical analysis results
        """
        data = self.get_stock_data(ticker, period='6mo')
        
        if 'error' in data:
            return {'error': data['error']}
        
        if not data['history']:
            return {'error': 'No historical data available'}
        
        df = pd.DataFrame(data['history'])
        if len(df) < 20:  # Need at least 20 days for meaningful technical analysis
            return {'error': 'Insufficient historical data for technical analysis'}
            
        # Calculate technical indicators
        df = self._calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # Generate signals
        signals = self._generate_signals(df)
        
        # Calculate score based on signals
        score = 50  # Neutral score
        score += sum(signals.values()) * 5  # Each signal contributes +/- 5 points
        score = max(0, min(100, score))  # Clamp between 0-100
        
        return {
            'ticker': ticker,
            'current_price': latest['Close'],
            'price_change_pct': ((latest['Close'] - prev['Close']) / prev['Close']) * 100,
            'volume': latest['Volume'],
            'indicators': {
                'sma_20': latest.get('SMA_20', 0),
                'sma_50': latest.get('SMA_50', 0),
                'rsi': latest.get('RSI', 50),
                'macd': latest.get('MACD', 0),
                'macd_signal': latest.get('MACD_Signal', 0),
                'bollinger_upper': latest.get('Bollinger_Upper', 0),
                'bollinger_lower': latest.get('Bollinger_Lower', 0),
            },
            'signals': signals,
            'score': score,
            'summary': self._generate_summary(score, latest, signals)
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['SMA'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Bollinger_Upper'] = df['SMA'] + (df['STD'] * 2)
        df['Bollinger_Lower'] = df['SMA'] - (df['STD'] * 2)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, int]:
        """Generate trading signals based on technical indicators."""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        signals = {}
        
        # Moving Average Crossover
        if 'SMA_20' in latest and 'SMA_50' in latest and 'SMA_20' in prev and 'SMA_50' in prev:
            if prev['SMA_20'] <= prev['SMA_50'] and latest['SMA_20'] > latest['SMA_50']:
                signals['ma_crossover'] = 1  # Bullish
            elif prev['SMA_20'] >= prev['SMA_50'] and latest['SMA_20'] < latest['SMA_50']:
                signals['ma_crossover'] = -1  # Bearish
        
        # RSI
        if 'RSI' in latest:
            if latest['RSI'] > 70:
                signals['rsi_overbought'] = -1
            elif latest['RSI'] < 30:
                signals['rsi_oversold'] = 1
        
        # MACD
        if 'MACD' in latest and 'MACD_Signal' in latest and 'MACD' in prev and 'MACD_Signal' in prev:
            if prev['MACD'] <= prev['MACD_Signal'] and latest['MACD'] > latest['MACD_Signal']:
                signals['macd_crossover'] = 1  # Bullish
            elif prev['MACD'] >= prev['MACD_Signal'] and latest['MACD'] < latest['MACD_Signal']:
                signals['macd_crossover'] = -1  # Bearish
        
        # Bollinger Bands
        if 'Close' in latest and 'Bollinger_Lower' in latest and 'Bollinger_Upper' in latest:
            if latest['Close'] < latest['Bollinger_Lower']:
                signals['bollinger_oversold'] = 1
            elif latest['Close'] > latest['Bollinger_Upper']:
                signals['bollinger_overbought'] = -1
        
        return signals
    
    def _generate_summary(self, score: float, latest: pd.Series, signals: Dict[str, int]) -> str:
        """Generate a summary based on the technical analysis."""
        summary = []
        
        # Overall score
        if score >= 70:
            summary.append("Strong bullish technical indicators suggest potential upward momentum.")
        elif score >= 50:
            summary.append("Mixed technical signals with a neutral to slightly bullish bias.")
        else:
            summary.append("Bearish technical indicators suggest caution.")
        
        # Add specific signals
        if 'ma_crossover' in signals:
            if signals['ma_crossover'] > 0:
                summary.append("Bullish moving average crossover detected.")
            else:
                summary.append("Bearish moving average crossover detected.")
                
        if 'rsi_oversold' in signals:
            summary.append("RSI indicates oversold conditions.")
        elif 'rsi_overbought' in signals:
            summary.append("RSI indicates overbought conditions.")
            
        if 'bollinger_oversold' in signals:
            summary.append("Price is near the lower Bollinger Band, indicating potential oversold conditions.")
        elif 'bollinger_overbought' in signals:
            summary.append("Price is near the upper Bollinger Band, indicating potential overbought conditions.")
        
        return " ".join(summary)
