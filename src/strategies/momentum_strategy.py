"""
Momentum Strategy

Implements momentum-based trading strategy:
- Price momentum analysis
- Trend following
- Relative strength comparison
- Moving averages
- Volatility adjustment
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
class MomentumAnalysis:
    """Detailed analysis results for Momentum strategy."""
    symbol: str
    momentum_score: float
    trend_strength: float
    volatility: float
    recommendation: str

class MomentumStrategy:
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.momentum_periods = [20, 50, 100]
        self.volatility_window = 20

    def analyze_company(self, price_data: Dict[str, List[float]], symbol: str) -> MomentumAnalysis:
        """
        Analyze momentum and trend for a company
        
        Args:
            price_data: Dictionary containing price history
            symbol: Company symbol
            
        Returns:
            MomentumAnalysis object with analysis results
        """
        try:
            # Convert price data to DataFrame
            df = pd.DataFrame(price_data)
            
            # Calculate momentum scores
            momentum_scores = []
            for period in self.momentum_periods:
                if len(df) >= period:
                    momentum = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                    momentum_scores.append(momentum)
            
            # Calculate average momentum
            momentum_score = np.mean(momentum_scores) if momentum_scores else 0
            
            # Calculate trend strength
            trend_strength = self.technical_analyzer.calculate_trend_strength(df)
            
            # Calculate volatility
            volatility = df['close'].pct_change().rolling(window=self.volatility_window).std().iloc[-1]
            
            # Make recommendation
            if momentum_score > 0.1 and trend_strength > 0.7:
                recommendation = "BUY"
            elif momentum_score > 0.05 and trend_strength > 0.5:
                recommendation = "HOLD"
            else:
                recommendation = "SELL"
            
            return MomentumAnalysis(
                symbol=symbol,
                momentum_score=momentum_score,
                trend_strength=trend_strength,
                volatility=volatility,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return MomentumAnalysis(
                symbol=symbol,
                momentum_score=0,
                trend_strength=0,
                volatility=0,
                recommendation="ERROR"
            )
