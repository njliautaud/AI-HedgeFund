#!/usr/bin/env python3
"""
Benjamin Graham Agent - Implementation of Benjamin Graham's value investing strategy

This agent implements Benjamin Graham's deep value investing philosophy:
- Focus on stocks trading below intrinsic value
- Emphasis on margin of safety
- Quantitative analysis of financial statements
- Conservative valuation methods
- Preference for stocks with low P/E, P/B ratios
- Strong balance sheet requirements
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from dataclasses import dataclass

# Import the base agent class
from .agent_base import Agent, AgentAction, AgentState

# Import utility functions
from ..utils.valuation_models import GrahamNumber, NetNetWorkingCapital


@dataclass
class GrahamAnalysis:
    """Detailed analysis results for Benjamin Graham strategy."""
    symbol: str
    graham_number: float
    ncav: float  # Net Current Asset Value
    current_price: float
    price_to_book: float
    pe_ratio: float
    current_ratio: float
    debt_to_equity: float
    dividend_yield: float
    margin_of_safety: float
    recommendation: str
    confidence: float
    reasoning: List[str]


class BenjaminGrahamAgent(Agent):
    """
    Agent implementation of Benjamin Graham's value investing strategy.
    
    Key principles:
    1. Stocks trading below intrinsic value (Graham Number, NCAV)
    2. Strong margin of safety
    3. Low price-to-book and P/E ratios
    4. Strong balance sheet (low debt, high current ratio)
    5. Dividend history
    """
    
    def __init__(self, config: Dict, data_manager):
        """
        Initialize the Benjamin Graham Agent.
        
        Args:
            config: Configuration dictionary
            data_manager: Data manager instance for accessing market data
        """
        super().__init__("benjamin_graham", config)
        
        self.data_manager = data_manager
        
        # Graham's criteria thresholds
        self.thresholds = config.get('thresholds', {
            'pe_ratio_max': 15,
            'price_to_book_max': 1.5,
            'current_ratio_min': 2.0,
            'debt_to_equity_max': 0.5,
            'dividend_years_min': 20,  # Years of continuous dividends
            'margin_of_safety_min': 0.3,  # 30% below intrinsic value
            'earnings_growth_min': 0.03  # 3% minimum earnings growth
        })
        
        # Initialize valuation models
        self.graham_number_calculator = GrahamNumber()
        self.ncav_calculator = NetNetWorkingCapital()
        
        logger.info("Benjamin Graham Agent initialized")
    
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
                logger.error(f"Error analyzing {symbol} with Graham strategy: {e}")
        
        # Sort by margin of safety (highest first)
        signals.sort(key=lambda x: x.get("margin_of_safety", 0), reverse=True)
        
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
    
    async def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[GrahamAnalysis]:
        """
        Perform Benjamin Graham style analysis on a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Dictionary of market data
            
        Returns:
            GrahamAnalysis object or None if analysis fails
        """
        try:
            # Get fundamental data
            fundamentals = await self.data_manager.get_fundamentals(symbol)
            if not fundamentals:
                return None
            
            # Get price data
            price_data = await self.data_manager.get_price_history(symbol, days=365)  # 1 year
            if price_data.empty:
                return None
            
            current_price = price_data.iloc[-1]['close']
            
            # Calculate Graham Number
            graham_number = self._calculate_graham_number(fundamentals)
            
            # Calculate Net Current Asset Value (NCAV)
            ncav = self._calculate_ncav(fundamentals)
            
            # Get key financial ratios
            price_to_book = fundamentals.get('price_to_book', float('inf'))
            pe_ratio = fundamentals.get('pe_ratio', float('inf'))
            current_ratio = fundamentals.get('current_ratio', 0)
            debt_to_equity = fundamentals.get('debt_to_equity', float('inf'))
            dividend_yield = fundamentals.get('dividend_yield', 0)
            
            # Calculate margin of safety (using the higher of Graham Number and NCAV)
            intrinsic_value = max(graham_number, ncav)
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else 0
            
            # Generate recommendation
            recommendation, confidence, reasoning = self._generate_recommendation(
                fundamentals, graham_number, ncav, current_price, price_to_book,
                pe_ratio, current_ratio, debt_to_equity, dividend_yield, margin_of_safety
            )
            
            return GrahamAnalysis(
                symbol=symbol,
                graham_number=graham_number,
                ncav=ncav,
                current_price=current_price,
                price_to_book=price_to_book,
                pe_ratio=pe_ratio,
                current_ratio=current_ratio,
                debt_to_equity=debt_to_equity,
                dividend_yield=dividend_yield,
                margin_of_safety=margin_of_safety,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in Graham analysis for {symbol}: {e}")
            return None
    
    def _calculate_graham_number(self, fundamentals: Dict) -> float:
        """
        Calculate the Graham Number.
        
        Graham Number = sqrt(15 * EPS * 1.5 * BVPS)
        
        Args:
            fundamentals: Dictionary of fundamental data
            
        Returns:
            Graham Number
        """
        try:
            eps = fundamentals.get('eps', 0)
            book_value_per_share = fundamentals.get('book_value_per_share', 0)
            
            if eps <= 0 or book_value_per_share <= 0:
                return 0.0
            
            graham_number = self.graham_number_calculator.calculate(
                eps=eps,
                book_value_per_share=book_value_per_share
            )
            
            return graham_number
            
        except Exception as e:
            logger.error(f"Error calculating Graham Number: {e}")
            return 0.0
    
    def _calculate_ncav(self, fundamentals: Dict) -> float:
        """
        Calculate the Net Current Asset Value (NCAV) per share.
        
        NCAV = (Current Assets - Total Liabilities) / Shares Outstanding
        
        Args:
            fundamentals: Dictionary of fundamental data
            
        Returns:
            NCAV per share
        """
        try:
            current_assets = fundamentals.get('current_assets', 0)
            total_liabilities = fundamentals.get('total_liabilities', 0)
            shares_outstanding = fundamentals.get('shares_outstanding', 0)
            
            if shares_outstanding <= 0:
                return 0.0
            
            ncav = self.ncav_calculator.calculate(
                current_assets=current_assets,
                total_liabilities=total_liabilities,
                shares_outstanding=shares_outstanding
            )
            
            return ncav
            
        except Exception as e:
            logger.error(f"Error calculating NCAV: {e}")
            return 0.0
    
    def _generate_recommendation(self, fundamentals: Dict, graham_number: float, 
                               ncav: float, current_price: float, price_to_book: float,
                               pe_ratio: float, current_ratio: float, debt_to_equity: float,
                               dividend_yield: float, margin_of_safety: float) -> Tuple[str, float, List[str]]:
        """
        Generate a recommendation based on Graham's criteria.
        
        Args:
            fundamentals: Dictionary of fundamental data
            graham_number: Calculated Graham Number
            ncav: Net Current Asset Value per share
            current_price: Current stock price
            price_to_book: Price-to-book ratio
            pe_ratio: Price-to-earnings ratio
            current_ratio: Current ratio
            debt_to_equity: Debt-to-equity ratio
            dividend_yield: Dividend yield
            margin_of_safety: Margin of safety
            
        Returns:
            Tuple of (recommendation, confidence, reasoning)
        """
        reasoning = []
        points = 0
        max_points = 0
        
        # Check if stock is trading below Graham Number
        max_points += 2
        if graham_number > 0 and current_price < graham_number:
            points += 2
            reasoning.append(f"Trading below Graham Number: ${graham_number:.2f} vs. ${current_price:.2f}")
        elif graham_number > 0:
            reasoning.append(f"Trading above Graham Number: ${graham_number:.2f} vs. ${current_price:.2f}")
        else:
            reasoning.append("Graham Number could not be calculated (negative EPS or book value)")
        
        # Check if stock is trading below NCAV
        max_points += 3
        if ncav > 0 and current_price < ncav:
            points += 3
            reasoning.append(f"Trading below NCAV: ${ncav:.2f} vs. ${current_price:.2f} (rare bargain)")
        elif ncav > 0:
            reasoning.append(f"Trading above NCAV: ${ncav:.2f} vs. ${current_price:.2f}")
        else:
            reasoning.append("NCAV is negative or could not be calculated")
        
        # Check price-to-book ratio
        max_points += 2
        if price_to_book <= self.thresholds['price_to_book_max']:
            if price_to_book <= 1.0:
                points += 2
                reasoning.append(f"Excellent price-to-book ratio: {price_to_book:.2f} (below 1.0)")
            else:
                points += 1
                reasoning.append(f"Good price-to-book ratio: {price_to_book:.2f} (below {self.thresholds['price_to_book_max']})")
        else:
            reasoning.append(f"Price-to-book ratio of {price_to_book:.2f} is above threshold of {self.thresholds['price_to_book_max']}")
        
        # Check P/E ratio
        max_points += 2
        if pe_ratio > 0 and pe_ratio <= self.thresholds['pe_ratio_max']:
            if pe_ratio <= 10:
                points += 2
                reasoning.append(f"Excellent P/E ratio: {pe_ratio:.2f} (below 10)")
            else:
                points += 1
                reasoning.append(f"Good P/E ratio: {pe_ratio:.2f} (below {self.thresholds['pe_ratio_max']})")
        elif pe_ratio <= 0:
            reasoning.append("Negative P/E ratio (negative earnings)")
        else:
            reasoning.append(f"P/E ratio of {pe_ratio:.2f} is above threshold of {self.thresholds['pe_ratio_max']}")
        
        # Check current ratio
        max_points += 2
        if current_ratio >= self.thresholds['current_ratio_min']:
            if current_ratio >= 3.0:
                points += 2
                reasoning.append(f"Excellent current ratio: {current_ratio:.2f} (above 3.0)")
            else:
                points += 1
                reasoning.append(f"Good current ratio: {current_ratio:.2f} (above {self.thresholds['current_ratio_min']})")
        else:
            reasoning.append(f"Current ratio of {current_ratio:.2f} is below threshold of {self.thresholds['current_ratio_min']}")
        
        # Check debt-to-equity ratio
        max_points += 2
        if debt_to_equity <= self.thresholds['debt_to_equity_max']:
            if debt_to_equity <= 0.3:
                points += 2
                reasoning.append(f"Excellent debt-to-equity ratio: {debt_to_equity:.2f} (below 0.3)")
            else:
                points += 1
                reasoning.append(f"Good debt-to-equity ratio: {debt_to_equity:.2f} (below {self.thresholds['debt_to_equity_max']})")
        else:
            reasoning.append(f"Debt-to-equity ratio of {debt_to_equity:.2f} is above threshold of {self.thresholds['debt_to_equity_max']}")
        
        # Check dividend yield
        max_points += 1
        if dividend_yield > 0:
            points += 1
            reasoning.append(f"Pays dividends with yield of {dividend_yield:.2%}")
        else:
            reasoning.append("Does not pay dividends")
        
        # Check margin of safety
        max_points += 3
        if margin_of_safety >= self.thresholds['margin_of_safety_min']:
            if margin_of_safety >= 0.5:
                points += 3
                reasoning.append(f"Excellent margin of safety: {margin_of_safety:.2%} (above 50%)")
            elif margin_of_safety >= 0.4:
                points += 2
                reasoning.append(f"Strong margin of safety: {margin_of_safety:.2%} (above 40%)")
            else:
                points += 1
                reasoning.append(f"Good margin of safety: {margin_of_safety:.2%} (above 30%)")
        else:
            reasoning.append(f"Margin of safety of {margin_of_safety:.2%} is below threshold of {self.thresholds['margin_of_safety_min']:.2%}")
        
        # Calculate confidence
        confidence = points / max_points if max_points > 0 else 0.0
        
        # Determine recommendation
        if confidence >= 0.7:
            recommendation = "BUY"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Strong buy recommendation")
        elif confidence >= 0.5:
            recommendation = "BUY"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Buy recommendation")
        elif confidence >= 0.3:
            recommendation = "HOLD"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Hold recommendation")
        else:
            recommendation = "PASS"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Pass recommendation")
        
        return recommendation, confidence, reasoning
    
    def _create_signal_from_analysis(self, analysis: GrahamAnalysis) -> Dict:
        """
        Create a signal dictionary from analysis results.
        
        Args:
            analysis: GrahamAnalysis object
            
        Returns:
            Signal dictionary
        """
        if analysis.recommendation not in ["BUY", "SELL"]:
            return None
        
        return {
            "agent_id": self.agent_id,
            "symbol": analysis.symbol,
            "action": analysis.recommendation,
            "confidence": analysis.confidence,
            "reasoning": "\n".join(analysis.reasoning),
            "graham_number": analysis.graham_number,
            "ncav": analysis.ncav,
            "price_to_book": analysis.price_to_book,
            "pe_ratio": analysis.pe_ratio,
            "current_ratio": analysis.current_ratio,
            "debt_to_equity": analysis.debt_to_equity,
            "margin_of_safety": analysis.margin_of_safety,
            "timestamp": datetime.now().isoformat()
        }