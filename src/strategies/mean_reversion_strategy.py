"""
Mean Reversion Strategy

Implements mean reversion trading strategy:
- Price deviation from moving averages
- Bollinger Bands
- RSI (Relative Strength Index)
- Volatility analysis
- Position sizing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass

from src.core.signals import StrategySignal
from ..utils.technical_indicators import TechnicalAnalyzer

@dataclass
class MeanReversionAnalysis:
    """Detailed analysis results for Mean Reversion strategy."""
    symbol: str
    deviation_score: float
    rsi: float
    volatility: float
    recommendation: str

class MeanReversionStrategy:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.lookback_period = 20
        self.rsi_period = 14
        self.deviation_threshold = 2.0  # Standard deviations
        self.rsi_overbought = 70
        self.rsi_oversold = 30

    def analyze_company(self, price_data: Dict[str, List[float]], symbol: str) -> MeanReversionAnalysis:
        """
        Analyze mean reversion potential for a company
        
        Args:
            price_data: Dictionary containing price history
            symbol: Company symbol
            
        Returns:
            MeanReversionAnalysis object with analysis results
        """
        try:
            # Convert price data to DataFrame
            df = pd.DataFrame(price_data)
            
            # Calculate moving average
            ma = df['close'].rolling(window=self.lookback_period).mean()
            
            # Calculate standard deviation
            std = df['close'].rolling(window=self.lookback_period).std()
            
            # Calculate deviation from mean
            deviation = (df['close'].iloc[-1] - ma.iloc[-1]) / std.iloc[-1]
            
            # Calculate RSI
            rsi = self.technical_analyzer.calculate_rsi(df, self.rsi_period)
            
            # Calculate volatility
            volatility = df['close'].pct_change().rolling(window=self.lookback_period).std().iloc[-1]
            
            # Make recommendation
            if abs(deviation) > self.deviation_threshold and rsi < self.rsi_oversold:
                recommendation = "BUY"
            elif abs(deviation) < self.deviation_threshold and rsi > self.rsi_overbought:
                recommendation = "SELL"
            else:
                recommendation = "HOLD"
            
            return MeanReversionAnalysis(
                symbol=symbol,
                deviation_score=deviation,
                rsi=rsi,
                volatility=volatility,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return MeanReversionAnalysis(
                symbol=symbol,
                deviation_score=0,
                rsi=0,
                volatility=0,
                recommendation="ERROR"
            )
