#!/usr/bin/env python3
"""
Warren Buffett Agent - Implementation of Warren Buffett's investment strategy

This agent implements Warren Buffett's value investing philosophy:
- Focus on intrinsic value vs market price
- Strong fundamentals (low debt, high ROE, consistent earnings)
- Long-term competitive advantages (moats)
- Management quality assessment
- Margin of safety principle
- "Buy and hold forever" mentality
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass

# Import the base agent class
from .agent_base import Agent, AgentAction, AgentState

# Import existing Warren Buffett strategy components
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


class WarrenBuffettAgent(Agent):
    """
    Agent implementation of Warren Buffett's investment strategy.
    
    Key principles:
    1. Intrinsic value calculation using DCF and other models
    2. Quality assessment (ROE, debt levels, consistency)
    3. Competitive moat analysis
    4. Management quality evaluation
    5. Margin of safety requirement
    """
    
    def __init__(self, config: Dict, data_manager):
        """
        Initialize the Warren Buffett Agent.
        
        Args:
            config: Configuration dictionary
            data_manager: Data manager instance for accessing market data
        """
        super().__init__("warren_buffett", config)
        
        self.data_manager = data_manager
        
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
        
        logger.info("Warren Buffett Agent initialized")
    
    async def process(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Process market data and generate investment signals.
        
        Args:
            inputs: Dictionary containing market data
            
        Returns:
            An AgentAction with investment signals
        """
        await self.update_state("ANALYZING")
        
        market_data = inputs.get("market_data", {})
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
        signals.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        action = AgentAction(
            action_type="SIGNAL",
            content={
                "signals": signals[:20],  # Return top 20 opportunities
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        await self.update_state("IDLE", action)
        return action
    
    async def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[BuffettAnalysis]:
        """
        Perform Warren Buffett style analysis on a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Dictionary of market data
            
        Returns:
            BuffettAnalysis object or None if analysis fails
        """
        try:
            # Get fundamental data
            fundamentals = await self.data_manager.get_fundamentals(symbol)
            if not fundamentals:
                return None
            
            # Get price data
            price_data = await self.data_manager.get_price_history(symbol, days=365*5)  # 5 years
            if price_data.empty:
                return None
            
            current_price = price_data.iloc[-1]['close']
            
            # Calculate intrinsic value
            intrinsic_value = await self._calculate_intrinsic_value(symbol, fundamentals, price_data)
            
            # Calculate margin of safety
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else 0
            
            # Assess business quality
            quality_score = self._assess_quality(fundamentals, price_data)
            
            # Assess competitive moat
            moat_score = self._assess_moat(fundamentals, price_data)
            
            # Assess management
            management_score = self._assess_management(fundamentals)
            
            # Assess financial strength
            financial_strength = self._assess_financial_strength(fundamentals)
            
            # Evaluate growth sustainability
            growth_sustainability = self._evaluate_growth_sustainability(fundamentals, price_data)
            
            # Generate recommendation
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
        """Calculate intrinsic value using multiple models."""
        # Use DCF model
        dcf_value = await self.dcf_model.calculate(symbol, fundamentals)
        
        # Use Graham Number as a secondary check
        graham_value = self.graham_number.calculate(fundamentals)
        
        # Weight the values (prefer DCF but consider Graham)
        if dcf_value > 0 and graham_value > 0:
            # If both are available, use weighted average
            intrinsic_value = (dcf_value * 0.7) + (graham_value * 0.3)
        elif dcf_value > 0:
            intrinsic_value = dcf_value
        elif graham_value > 0:
            intrinsic_value = graham_value
        else:
            # Fallback to a simple earnings multiple
            eps = fundamentals.get('eps', 0)
            pe_ratio = self.thresholds.get('pe_ratio_max', 15)
            intrinsic_value = eps * pe_ratio
        
        return intrinsic_value
    
    def _assess_quality(self, fundamentals: Dict, price_data: pd.DataFrame) -> float:
        """Assess the quality of the business using Buffett's criteria."""
        score = 0
        max_score = 0
        
        # ROE consistency and level
        roe = fundamentals.get('roe', 0)
        if roe >= 0.20:
            score += 30
        elif roe >= 0.15:
            score += 20
        elif roe >= 0.10:
            score += 10
        max_score += 30
        
        # Earnings stability
        earnings_stability = fundamentals.get('earnings_stability', 0)
        if earnings_stability >= 0.9:
            score += 20
        elif earnings_stability >= 0.7:
            score += 10
        max_score += 20
        
        # Debt levels
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity <= 0.3:
            score += 20
        elif debt_to_equity <= 0.5:
            score += 10
        max_score += 20
        
        # Gross margins
        gross_margin = fundamentals.get('gross_margin', 0)
        if gross_margin >= 0.40:
            score += 15
        elif gross_margin >= 0.30:
            score += 10
        max_score += 15
        
        # Free cash flow
        fcf_to_sales = fundamentals.get('fcf_to_sales', 0)
        if fcf_to_sales >= 0.15:
            score += 15
        elif fcf_to_sales >= 0.10:
            score += 10
        max_score += 15
        
        # Calculate final score (0-1 range)
        return score / max_score if max_score > 0 else 0
    
    def _assess_moat(self, fundamentals: Dict, price_data: pd.DataFrame) -> float:
        """Assess the competitive moat of the business."""
        score = 0
        max_score = 0
        
        # Brand value (approximated by gross margin)
        gross_margin = fundamentals.get('gross_margin', 0)
        if gross_margin >= 0.50:  # Luxury/strong brand
            score += 25
        elif gross_margin >= 0.35:  # Good brand
            score += 15
        max_score += 25
        
        # Market share and leadership
        market_share = fundamentals.get('market_share', 0)
        if market_share >= 0.30:  # Market leader
            score += 25
        elif market_share >= 0.15:  # Strong player
            score += 15
        max_score += 25
        
        # Pricing power (approximated by margin trends)
        margin_trend = fundamentals.get('margin_trend', 0)
        if margin_trend > 0.05:  # Increasing margins
            score += 20
        elif margin_trend > 0:  # Stable margins
            score += 10
        max_score += 20
        
        # Switching costs and lock-in
        recurring_revenue = fundamentals.get('recurring_revenue_pct', 0)
        if recurring_revenue >= 0.70:  # High lock-in
            score += 15
        elif recurring_revenue >= 0.40:  # Moderate lock-in
            score += 10
        max_score += 15
        
        # Network effects
        network_effect = fundamentals.get('network_effect', 0)
        if network_effect >= 0.7:  # Strong network effects
            score += 15
        elif network_effect >= 0.3:  # Some network effects
            score += 10
        max_score += 15
        
        # Calculate final score (0-1 range)
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
        
        # Capital allocation (approximated by ROIC vs WACC)
        roic = fundamentals.get('roic', 0)
        wacc = fundamentals.get('wacc', 0.08)  # Default to 8% if not available
        
        if roic > wacc + 0.05:  # Significantly above cost of capital
            score += 30
        elif roic > wacc:  # Above cost of capital
            score += 20
        max_score += 30
        
        # Shareholder-friendly actions
        dividend_payout = fundamentals.get('dividend_payout', 0)
        buyback_yield = fundamentals.get('buyback_yield', 0)
        
        if dividend_payout > 0.3 or buyback_yield > 0.02:
            score += 30
        elif dividend_payout > 0.15 or buyback_yield > 0.01:
            score += 15
        max_score += 30
        
        # Calculate final score (0-1 range)
        return score / max_score if max_score > 0 else 0
    
    def _assess_financial_strength(self, fundamentals: Dict) -> float:
        """Assess the financial strength of the company."""
        score = 0
        max_score = 0
        
        # Current ratio
        current_ratio = fundamentals.get('current_ratio', 0)
        if current_ratio >= 2.0:
            score += 25
        elif current_ratio >= 1.5:
            score += 15
        elif current_ratio >= 1.0:
            score += 5
        max_score += 25
        
        # Debt to equity
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity <= 0.3:
            score += 25
        elif debt_to_equity <= 0.5:
            score += 15
        elif debt_to_equity <= 1.0:
            score += 5
        max_score += 25
        
        # Interest coverage ratio
        interest_coverage = fundamentals.get('interest_coverage', 0)
        if interest_coverage >= 10:
            score += 25
        elif interest_coverage >= 5:
            score += 15
        elif interest_coverage >= 2:
            score += 5
        max_score += 25
        
        # Cash to debt
        cash_to_debt = fundamentals.get('cash_to_debt', 0)
        if cash_to_debt >= 1.0:  # More cash than debt
            score += 25
        elif cash_to_debt >= 0.5:
            score += 15
        elif cash_to_debt >= 0.2:
            score += 5
        max_score += 25
        
        # Calculate final score (0-1 range)
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
            return 0.9  # Very sustainable
        elif revenue_growth <= sustainable_growth * 1.2 and debt_to_equity < 1.0:
            return 0.7  # Moderately sustainable
        elif revenue_growth <= sustainable_growth * 1.5 and debt_to_equity < 1.5:
            return 0.5  # Somewhat sustainable
        else:
            return 0.3  # Less sustainable
    
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
        
        # Valuation check
        if margin_of_safety >= self.thresholds['margin_of_safety_min']:
            reasoning.append(f"Margin of safety of {margin_of_safety:.1%} exceeds minimum threshold of {self.thresholds['margin_of_safety_min']:.1%}")
        else:
            reasoning.append(f"Insufficient margin of safety: {margin_of_safety:.1%} vs required {self.thresholds['margin_of_safety_min']:.1%}")
        
        # Quality check
        if quality_score >= self.thresholds['quality_score_min']:
            reasoning.append(f"Business quality score of {quality_score:.2f} exceeds minimum threshold of {self.thresholds['quality_score_min']:.2f}")
        else:
            reasoning.append(f"Insufficient business quality: {quality_score:.2f} vs required {self.thresholds['quality_score_min']:.2f}")
        
        # Moat check
        if moat_score >= self.thresholds['moat_score_min']:
            reasoning.append(f"Competitive moat score of {moat_score:.2f} exceeds minimum threshold of {self.thresholds['moat_score_min']:.2f}")
        else:
            reasoning.append(f"Insufficient competitive moat: {moat_score:.2f} vs required {self.thresholds['moat_score_min']:.2f}")
        
        # Financial metrics check
        if pe_ratio <= self.thresholds['pe_ratio_max']:
            reasoning.append(f"P/E ratio of {pe_ratio:.1f} is below maximum threshold of {self.thresholds['pe_ratio_max']:.1f}")
        else:
            reasoning.append(f"P/E ratio too high: {pe_ratio:.1f} vs maximum {self.thresholds['pe_ratio_max']:.1f}")
            
        if roe >= self.thresholds['roe_min']:
            reasoning.append(f"ROE of {roe:.1%} exceeds minimum threshold of {self.thresholds['roe_min']:.1%}")
        else:
            reasoning.append(f"ROE too low: {roe:.1%} vs minimum {self.thresholds['roe_min']:.1%}")
            
        if debt_to_equity <= self.thresholds['debt_to_equity_max']:
            reasoning.append(f"Debt-to-equity ratio of {debt_to_equity:.2f} is below maximum threshold of {self.thresholds['debt_to_equity_max']:.2f}")
        else:
            reasoning.append(f"Debt-to-equity ratio too high: {debt_to_equity:.2f} vs maximum {self.thresholds['debt_to_equity_max']:.2f}")
        
        # Determine recommendation
        buy_criteria = [
            margin_of_safety >= self.thresholds['margin_of_safety_min'],
            quality_score >= self.thresholds['quality_score_min'],
            moat_score >= self.thresholds['moat_score_min'],
            pe_ratio <= self.thresholds['pe_ratio_max'],
            roe >= self.thresholds['roe_min'],
            debt_to_equity <= self.thresholds['debt_to_equity_max']
        ]
        
        buy_score = sum(buy_criteria)
        
        if buy_score >= 5:  # At least 5 of 6 criteria met
            recommendation = "BUY"
            confidence = 0.8 + (buy_score - 5) * 0.1  # 0.8 to 0.9
        elif buy_score >= 4:  # 4 of 6 criteria met
            recommendation = "BUY"
            confidence = 0.7
        elif buy_score <= 2:  # 2 or fewer criteria met
            recommendation = "SELL"
            confidence = 0.7 + (2 - buy_score) * 0.1  # 0.7 to 0.9
        else:  # 3 criteria met
            recommendation = "HOLD"
            confidence = 0.6
        
        # Adjust confidence based on financial strength
        if financial_strength < 0.5 and recommendation != "SELL":
            confidence -= 0.1
            reasoning.append(f"Confidence reduced due to financial strength concerns: {financial_strength:.2f}")
        
        return recommendation, confidence, reasoning
    
    def _create_signal_from_analysis(self, analysis: BuffettAnalysis) -> Optional[Dict]:
        """Create a trading signal from the analysis."""
        if analysis.recommendation == "HOLD":
            return None
        
        # Calculate target weight based on confidence and margin of safety
        base_weight = self.config.get('weight', 0.2)
        weight_adjustment = (analysis.confidence - 0.5) * 0.5  # -0.25 to +0.25
        
        if analysis.margin_of_safety > 0.4:  # Exceptional value
            weight_adjustment += 0.1
        
        target_weight = base_weight + weight_adjustment
        target_weight = max(0.05, min(0.3, target_weight))  # Cap between 5% and 30%
        
        return {
            "strategy": "warren_buffett",
            "symbol": analysis.symbol,
            "action": analysis.recommendation,
            "confidence": analysis.confidence,
            "target_weight": target_weight,
            "reasoning": "\n".join(analysis.reasoning),
            "intrinsic_value": analysis.intrinsic_value,
            "current_price": analysis.current_price,
            "margin_of_safety": analysis.margin_of_safety,
            "quality_score": analysis.quality_score,
            "moat_score": analysis.moat_score,
            "timestamp": datetime.now().isoformat(),
            "weight": self.config.get('weight', 0.2)  # Strategy weight in portfolio
        }