"""
Peter Lynch AI Strategy

Implements Peter Lynch's investment philosophy:
- Growth at Reasonable Price (GARP)
- Focus on earnings growth
- P/E ratio relative to growth rate
- Company fundamentals analysis
- Industry growth trends
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
from ..utils.technical_indicators import TechnicalAnalyzer

@dataclass
class LynchAnalysis:
    """Detailed analysis results for Lynch strategy."""
    symbol: str
    pe_ratio: float
    growth_rate: float
    peg_ratio: float
    quality_score: float
    technical_score: float
    recommendation: str

class PeterLynchAI:
    def __init__(self):
        self.dcf_model = DCFModel()
        self.graham_number = GrahamNumber()
        self.quality_scorer = QualityScorer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.max_peg_ratio = 1.0  # Lynch's rule of thumb

    def analyze_company(self, financial_data: Dict[str, Any]) -> LynchAnalysis:
        """
        Analyze a company using Peter Lynch's criteria
        
        Args:
            financial_data: Dictionary containing company financials
            
        Returns:
            LynchAnalysis object with analysis results
        """
        try:
            # Calculate key metrics
            pe_ratio = financial_data.get('pe_ratio', 0)
            growth_rate = financial_data.get('growth_rate', 0)
            
            # Calculate PEG ratio (PE to Growth)
            peg_ratio = pe_ratio / growth_rate if growth_rate != 0 else float('inf')
            
            # Calculate quality score
            quality_score = self.quality_scorer.score_company(financial_data)
            
            # Calculate technical score
            technical_score = self.technical_analyzer.score_company(financial_data)
            
            # Make recommendation based on Lynch's criteria
            if peg_ratio <= self.max_peg_ratio and quality_score > 70:
                recommendation = "BUY"
            elif peg_ratio > self.max_peg_ratio and quality_score > 70:
                recommendation = "HOLD"
            else:
                recommendation = "SELL"
            
            return LynchAnalysis(
                symbol=financial_data['symbol'],
                pe_ratio=pe_ratio,
                growth_rate=growth_rate,
                peg_ratio=peg_ratio,
                quality_score=quality_score,
                technical_score=technical_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {financial_data.get('symbol', 'unknown')}: {e}")
            return LynchAnalysis(
                symbol=financial_data.get('symbol', ''),
                pe_ratio=0,
                growth_rate=0,
                peg_ratio=float('inf'),
                quality_score=0,
                technical_score=0,
                recommendation="ERROR"
            )
