"""
Benjamin Graham AI Strategy

Implements Benjamin Graham's value investing philosophy:
- Margin of safety
- Defensive investing
- Business fundamentals analysis
- Financial strength assessment
- Market price vs intrinsic value
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass

from src.core.signals import StrategySignal
from ..utils.valuation_models import DCFModel, GrahamNumber
from ..utils.quality_metrics import QualityScorer

@dataclass
class GrahamAnalysis:
    """Detailed analysis results for Graham strategy."""
    symbol: str
    intrinsic_value: float
    current_price: float
    margin_of_safety: float
    quality_score: float
    recommendation: str

class BenjaminGrahamAI:
    def __init__(self):
        self.dcf_model = DCFModel()
        self.graham_number = GrahamNumber()
        self.quality_scorer = QualityScorer()
        self.min_margin_of_safety = 0.5  # 50% margin of safety

    def analyze_company(self, financial_data: Dict[str, Any]) -> GrahamAnalysis:
        """
        Analyze a company using Benjamin Graham's criteria
        
        Args:
            financial_data: Dictionary containing company financials
            
        Returns:
            GrahamAnalysis object with analysis results
        """
        try:
            # Calculate intrinsic value using Graham Number
            intrinsic_value = self.graham_number.calculate(
                eps=financial_data.get('eps', 0),
                book_value=financial_data.get('book_value', 0)
            )
            
            # Calculate quality score
            quality_score = self.quality_scorer.score_company(financial_data)
            
            # Calculate margin of safety
            current_price = financial_data.get('current_price', 0)
            if intrinsic_value > 0:
                margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
            else:
                margin_of_safety = 0
            
            # Make recommendation based on Graham's criteria
            if margin_of_safety >= self.min_margin_of_safety and quality_score > 70:
                recommendation = "BUY"
            elif margin_of_safety < self.min_margin_of_safety and quality_score > 70:
                recommendation = "HOLD"
            else:
                recommendation = "SELL"
            
            return GrahamAnalysis(
                symbol=financial_data['symbol'],
                intrinsic_value=intrinsic_value,
                current_price=current_price,
                margin_of_safety=margin_of_safety,
                quality_score=quality_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {financial_data.get('symbol', 'unknown')}: {e}")
            return GrahamAnalysis(
                symbol=financial_data.get('symbol', ''),
                intrinsic_value=0,
                current_price=0,
                margin_of_safety=0,
                quality_score=0,
                recommendation="ERROR"
            )
