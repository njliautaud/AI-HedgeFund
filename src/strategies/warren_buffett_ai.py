#!/usr/bin/env python3
"""
Warren Buffett AI Strategy

Implements Warren Buffett's value investing philosophy:
- Focus on intrinsic value vs market price
- Strong fundamentals (low debt, high ROE, consistent earnings)
- Long-term competitive advantages (moats)
- Management quality assessment
- Margin of safety principle
- "Buy and hold forever" mentality
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
class BuffettAnalysis:
    """Detailed analysis results for Buffett strategy."""
    symbol: str
    intrinsic_value: float
    current_price: float
    margin_of_safety: float
    quality_score: float
    moat_score: float
    management_score: float
    financial_strength: float
    growth_sustainability: float
    recommendation: str
    confidence: float
    reasoning: List[str]


class WarrenBuffettAI:
    """
    AI implementation of Warren Buffett's investment strategy.
    
    Key principles:
    1. Intrinsic value calculation using DCF and other models
    2. Quality assessment (ROE, debt levels, consistency)
    3. Competitive moat analysis
    4. Management quality evaluation
    5. Margin of safety requirement
    """
    
    def __init__(self, config: Dict, data_manager):
        self.config = config
        self.data_manager = data_manager
        self.enabled = True
        
        # Buffett's criteria thresholds
        self.thresholds = config.get('thresholds', {
            'pe_ratio_max': 15,
            'price_to_book_max': 1.5,
            'debt_to_equity_max': 0.5,
            'roe_min': 0.15,
            'margin_of_safety_min': 0.25,  # 25% discount to intrinsic value
            'quality_score_min': 0.7,
            'moat_score_min': 0.6
        })
        
        # Initialize valuation models
        self.dcf_model = DCFModel()
        self.graham_number = GrahamNumber()
        self.quality_scorer = QualityScorer()
        
        logger.info("Warren Buffett AI strategy initialized")
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Generate investment signals based on Buffett's methodology."""
        signals = []
        
        market_symbols = market_data.get('market_data', {})
        
        for symbol in market_symbols.keys():
            try:
                analysis = await self.analyze_symbol(symbol, market_data)
                
                if analysis:
                    signal = self._create_signal_from_analysis(analysis)
                    if signal:
                        signals.append(signal)
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol} with Buffett strategy: {e}")
        
        # Sort by confidence and margin of safety
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals[:20]  # Return top 20 opportunities
    
    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[BuffettAnalysis]:
        """Perform comprehensive Buffett-style analysis on a symbol."""
        try:
            # Get fundamental data
            fundamentals = self.data_manager.get_fundamental_metrics(symbol)
            if not fundamentals:
                return None
            
            # Get price data
            symbol_data = market_data['market_data'].get(symbol, {})
            price_data = symbol_data.get('price_data')
            if price_data is None or price_data.empty:
                return None
            
            current_price = price_data['Close'].iloc[-1]
            
            # 1. Calculate intrinsic value
            intrinsic_value = await self._calculate_intrinsic_value(symbol, fundamentals, price_data)
            
            # 2. Assess quality metrics
            quality_score = self._assess_quality(fundamentals, price_data)
            
            # 3. Evaluate competitive moat
            moat_score = self._evaluate_moat(symbol, fundamentals)
            
            # 4. Assess management quality
            management_score = self._assess_management(fundamentals)
            
            # 5. Calculate financial strength
            financial_strength = self._calculate_financial_strength(fundamentals)
            
            # 6. Evaluate growth sustainability
            growth_sustainability = self._evaluate_growth_sustainability(fundamentals, price_data)
            
            # 7. Calculate margin of safety
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else -1
            
            # 8. Generate recommendation
            recommendation, confidence, reasoning = self._generate_recommendation(
                fundamentals, intrinsic_value, current_price, margin_of_safety,
                quality_score, moat_score, management_score, financial_strength
            )
            
            return BuffettAnalysis(
                symbol=symbol,
                intrinsic_value=intrinsic_value,
                current_price=current_price,
                margin_of_safety=margin_of_safety,
                quality_score=quality_score,
                moat_score=moat_score,
                management_score=management_score,
                financial_strength=financial_strength,
                growth_sustainability=growth_sustainability,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in Buffett analysis for {symbol}: {e}")
            return None
    
    async def _calculate_intrinsic_value(self, symbol: str, fundamentals: Dict, price_data: pd.DataFrame) -> float:
        """Calculate intrinsic value using multiple methods."""
        try:
            # Method 1: DCF Model
            dcf_value = self.dcf_model.calculate(
                free_cash_flow=fundamentals.get('free_cash_flow', 0),
                growth_rate=fundamentals.get('revenue_growth', 0.03),
                discount_rate=0.10,  # Buffett's required return
                terminal_growth=0.03
            )
            
            # Method 2: Graham Number (for value stocks)
            graham_value = self.graham_number.calculate(
                eps=fundamentals.get('eps', 0),
                book_value_per_share=fundamentals.get('book_value', 0)
            )
            
            # Method 3: Earnings-based valuation
            earnings_value = self._calculate_earnings_value(fundamentals)
            
            # Method 4: Asset-based valuation
            asset_value = self._calculate_asset_value(fundamentals)
            
            # Weighted average of methods (Buffett uses multiple approaches)
            values = [v for v in [dcf_value, graham_value, earnings_value, asset_value] if v > 0]
            
            if not values:
                return 0
            
            # Weight DCF more heavily for growth companies, Graham number for value
            pe_ratio = fundamentals.get('pe_ratio', 0)
            if pe_ratio > 0 and pe_ratio < 15:  # Value stock
                weights = [0.3, 0.4, 0.2, 0.1][:len(values)]
            else:  # Growth stock
                weights = [0.5, 0.2, 0.2, 0.1][:len(values)]
            
            weights = weights[:len(values)]
            weights = [w / sum(weights) for w in weights]  # Normalize
            
            intrinsic_value = sum(v * w for v, w in zip(values, weights))
            
            return intrinsic_value
            
        except Exception as e:
            logger.error(f"Error calculating intrinsic value for {symbol}: {e}")
            return 0
    
    def _calculate_earnings_value(self, fundamentals: Dict) -> float:
        """Calculate value based on earnings power."""
        try:
            # Use normalized earnings and conservative multiple
            market_cap = fundamentals.get('market_cap', 0)
            pe_ratio = fundamentals.get('pe_ratio', 0)
            
            if pe_ratio > 0:
                earnings = market_cap / pe_ratio
                # Apply conservative multiple (Buffett typically uses 10-15x)
                conservative_multiple = min(12, pe_ratio * 0.8)
                return earnings * conservative_multiple
            
            return 0
            
        except Exception:
            return 0
    
    def _calculate_asset_value(self, fundamentals: Dict) -> float:
        """Calculate asset-based value (book value approach)."""
        try:
            book_value = fundamentals.get('book_value', 0)
            roe = fundamentals.get('roe', 0)
            
            # Adjust book value based on ROE quality
            if roe > 0.15:  # High quality ROE
                return book_value * 1.5
            elif roe > 0.10:
                return book_value * 1.2
            else:
                return book_value * 0.8
                
        except Exception:
            return 0
    
    def _assess_quality(self, fundamentals: Dict, price_data: pd.DataFrame) -> float:
        """Assess the quality of the business using Buffett's criteria."""
        score = 0
        max_score = 0
        
        # ROE consistency and level
        roe = fundamentals.get('roe', 0)
        if roe >= 0.20:
            score += 25
        elif roe >= 0.15:
            score += 20
        elif roe >= 0.10:
            score += 10
        max_score += 25
        
        # Debt levels (Buffett prefers low debt)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity <= 0.3:
            score += 20
        elif debt_to_equity <= 0.5:
            score += 15
        elif debt_to_equity <= 1.0:
            score += 5
        max_score += 20
        
        # Profit margins
        # (Would need income statement data for precise calculation)
        max_score += 15
        score += 10  # Placeholder
        
        # Revenue growth consistency
        revenue_growth = fundamentals.get('revenue_growth', 0)
        if revenue_growth > 0.10:
            score += 15
        elif revenue_growth > 0.05:
            score += 10
        elif revenue_growth > 0:
            score += 5
        max_score += 15
        
        # Free cash flow generation
        free_cash_flow = fundamentals.get('free_cash_flow', 0)
        market_cap = fundamentals.get('market_cap', 1)
        fcf_yield = free_cash_flow / market_cap if market_cap > 0 else 0
        
        if fcf_yield > 0.08:
            score += 15
        elif fcf_yield > 0.05:
            score += 10
        elif fcf_yield > 0.02:
            score += 5
        max_score += 15
        
        # Current ratio (liquidity)
        current_ratio = fundamentals.get('current_ratio', 0)
        if current_ratio >= 2.0:
            score += 10
        elif current_ratio >= 1.5:
            score += 7
        elif current_ratio >= 1.0:
            score += 3
        max_score += 10
        
        return score / max_score if max_score > 0 else 0
    
    def _evaluate_moat(self, symbol: str, fundamentals: Dict) -> float:
        """Evaluate competitive moat strength."""
        # This is a simplified moat assessment
        # In practice, this would involve more sophisticated analysis
        
        score = 0
        max_score = 0
        
        # High ROE suggests competitive advantages
        roe = fundamentals.get('roe', 0)
        if roe > 0.20:
            score += 30
        elif roe > 0.15:
            score += 20
        max_score += 30
        
        # High profit margins (proxy)
        # Would need more detailed financial data
        max_score += 25
        score += 15  # Placeholder
        
        # Market cap (larger companies often have stronger moats)
        market_cap = fundamentals.get('market_cap', 0)
        if market_cap > 100e9:  # $100B+
            score += 20
        elif market_cap > 10e9:  # $10B+
            score += 15
        elif market_cap > 1e9:  # $1B+
            score += 10
        max_score += 20
        
        # Price stability (lower volatility suggests moat)
        # Would calculate from price data
        max_score += 25
        score += 15  # Placeholder
        
        return score / max_score if max_score > 0 else 0
    
    def _assess_management(self, fundamentals: Dict) -> float:
        """Assess management quality (simplified)."""
        # This would ideally include insider ownership, capital allocation, etc.
        # For now, use financial metrics as proxies
        
        score = 0
        max_score = 0
        
        # ROE trend (good management improves ROE over time)
        roe = fundamentals.get('roe', 0)
        if roe > 0.15:
            score += 40
        elif roe > 0.10:
            score += 25
        max_score += 40
        
        # Debt management
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:
            score += 30
        elif debt_to_equity < 0.6:
            score += 20
        max_score += 30
        
        # Growth in book value
        # Would need historical data
        max_score += 30
        score += 20  # Placeholder
        
        return score / max_score if max_score > 0 else 0
    
    def _calculate_financial_strength(self, fundamentals: Dict) -> float:
        """Calculate overall financial strength score."""
        score = 0
        max_score = 0
        
        # Current ratio
        current_ratio = fundamentals.get('current_ratio', 0)
        if current_ratio >= 2.0:
            score += 25
        elif current_ratio >= 1.5:
            score += 20
        elif current_ratio >= 1.0:
            score += 10
        max_score += 25
        
        # Debt to equity
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity <= 0.3:
            score += 25
        elif debt_to_equity <= 0.6:
            score += 15
        elif debt_to_equity <= 1.0:
            score += 5
        max_score += 25
        
        # Interest coverage (would need income statement)
        max_score += 25
        score += 15  # Placeholder
        
        # Cash position relative to debt
        max_score += 25
        score += 15  # Placeholder
        
        return score / max_score if max_score > 0 else 0
    
    def _evaluate_growth_sustainability(self, fundamentals: Dict, price_data: pd.DataFrame) -> float:
        """Evaluate sustainability of growth."""
        # Simplified growth sustainability assessment
        revenue_growth = fundamentals.get('revenue_growth', 0)
        roe = fundamentals.get('roe', 0)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        
        # Sustainable growth rate approximation
        retention_ratio = 0.7  # Assume 70% earnings retention
        sustainable_growth = roe * retention_ratio
        
        if revenue_growth <= sustainable_growth and debt_to_equity < 0.5:
            return 0.8
        elif revenue_growth <= sustainable_growth * 1.2:
            return 0.6
        else:
            return 0.3
    
    def _generate_recommendation(self, fundamentals: Dict, intrinsic_value: float, 
                               current_price: float, margin_of_safety: float,
                               quality_score: float, moat_score: float, 
                               management_score: float, financial_strength: float) -> tuple:
        """Generate buy/sell/hold recommendation with reasoning."""
        reasoning = []
        
        # Check Buffett's key criteria
        pe_ratio = fundamentals.get('pe_ratio', 0)
        roe = fundamentals.get('roe', 0)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        
        # Calculate overall score
        scores = [quality_score, moat_score, management_score, financial_strength]
        overall_score = np.mean(scores)
        
        # Decision logic based on Buffett's criteria
        if (margin_of_safety >= self.thresholds['margin_of_safety_min'] and
            quality_score >= self.thresholds['quality_score_min'] and
            moat_score >= self.thresholds['moat_score_min'] and
            roe >= self.thresholds['roe_min'] and
            debt_to_equity <= self.thresholds['debt_to_equity_max']):
            
            recommendation = 'BUY'
            confidence = min(0.95, 0.5 + margin_of_safety + overall_score * 0.3)
            reasoning.append(f"Strong margin of safety: {margin_of_safety:.1%}")
            reasoning.append(f"High quality business (score: {quality_score:.2f})")
            reasoning.append(f"Strong competitive moat (score: {moat_score:.2f})")
            
        elif (margin_of_safety >= 0.15 and
              quality_score >= 0.6 and
              roe >= 0.10):
            
            recommendation = 'BUY'
            confidence = min(0.8, 0.4 + margin_of_safety + overall_score * 0.25)
            reasoning.append(f"Adequate margin of safety: {margin_of_safety:.1%}")
            reasoning.append(f"Decent quality metrics")
            
        elif margin_of_safety < -0.2:  # Overvalued by 20%+
            recommendation = 'SELL'
            confidence = min(0.8, 0.5 + abs(margin_of_safety) * 0.5)
            reasoning.append(f"Significantly overvalued (margin: {margin_of_safety:.1%})")
            
        else:
            recommendation = 'HOLD'
            confidence = 0.3 + overall_score * 0.2
            reasoning.append("Insufficient margin of safety or quality concerns")
        
        # Add specific reasoning based on metrics
        if roe < 0.10:
            reasoning.append(f"Low ROE: {roe:.1%}")
        if debt_to_equity > 0.6:
            reasoning.append(f"High debt levels: {debt_to_equity:.2f}")
        if pe_ratio > 20:
            reasoning.append(f"High valuation: P/E {pe_ratio:.1f}")
        
        return recommendation, confidence, reasoning
    
    def _create_signal_from_analysis(self, analysis: BuffettAnalysis) -> Optional[StrategySignal]:
        """Create a trading signal from the analysis."""
        if analysis.recommendation == 'HOLD':
            return None
        
        # Calculate target weight based on confidence and margin of safety
        base_weight = self.config.get('weight', 0.2)
        
        if analysis.recommendation == 'BUY':
            target_weight = base_weight * analysis.confidence * (1 + analysis.margin_of_safety)
        else:  # SELL
            target_weight = 0
        
        # Cap maximum position size
        target_weight = min(target_weight, 0.05)  # Max 5% per position
        
        return StrategySignal(
            strategy_name='warren_buffett',
            symbol=analysis.symbol,
            action=analysis.recommendation,
            confidence=analysis.confidence,
            target_weight=target_weight,
            reasoning='; '.join(analysis.reasoning),
            timestamp=datetime.now(),
            metadata={
                'intrinsic_value': analysis.intrinsic_value,
                'current_price': analysis.current_price,
                'margin_of_safety': analysis.margin_of_safety,
                'quality_score': analysis.quality_score,
                'moat_score': analysis.moat_score
            }
        )
    
    async def backtest(self, historical_data: Dict, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Backtest the Buffett strategy over a historical period."""
        # Implementation would run the strategy over historical data
        # and calculate performance metrics
        pass