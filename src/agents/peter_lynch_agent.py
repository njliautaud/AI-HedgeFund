#!/usr/bin/env python3
"""
Peter Lynch Agent - Implementation of Peter Lynch's investment strategy

This agent implements Peter Lynch's growth investing philosophy:
- Focus on companies with strong growth potential
- PEG ratio (Price/Earnings to Growth) analysis
- Preference for companies with high growth and reasonable P/E
- Categorization of stocks (fast growers, stalwarts, turnarounds, etc.)
- Emphasis on understanding the business model
- "Invest in what you know" principle
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
from ..utils.growth_metrics import calculate_peg_ratio, categorize_growth


@dataclass
class LynchAnalysis:
    """Detailed analysis results for Peter Lynch strategy."""
    symbol: str
    category: str  # Fast grower, stalwart, slow grower, cyclical, turnaround, asset play
    peg_ratio: float
    earnings_growth: float
    revenue_growth: float
    current_price: float
    target_price: float
    upside_potential: float
    story_quality: float  # How compelling is the business story
    recommendation: str
    confidence: float
    reasoning: List[str]


class PeterLynchAgent(Agent):
    """
    Agent implementation of Peter Lynch's investment strategy.
    
    Key principles:
    1. PEG ratio analysis (P/E relative to growth rate)
    2. Stock categorization (fast growers, stalwarts, etc.)
    3. Business story evaluation
    4. Earnings and revenue growth analysis
    5. "Invest in what you know" principle
    """
    
    def __init__(self, config: Dict, data_manager):
        """
        Initialize the Peter Lynch Agent.
        
        Args:
            config: Configuration dictionary
            data_manager: Data manager instance for accessing market data
        """
        super().__init__("peter_lynch", config)
        
        self.data_manager = data_manager
        
        # Lynch's criteria thresholds
        self.thresholds = config.get('thresholds', {
            'peg_ratio_max': 1.0,  # PEG ratio below 1.0 is considered good
            'earnings_growth_min': 0.15,  # 15% minimum earnings growth
            'revenue_growth_min': 0.10,  # 10% minimum revenue growth
            'upside_potential_min': 0.25,  # 25% minimum upside potential
            'story_quality_min': 0.7  # Minimum score for business story
        })
        
        # Category weights - how much emphasis to place on each stock category
        self.category_weights = config.get('category_weights', {
            'fast_grower': 1.0,
            'stalwart': 0.8,
            'slow_grower': 0.4,
            'cyclical': 0.6,
            'turnaround': 0.7,
            'asset_play': 0.5
        })
        
        logger.info("Peter Lynch Agent initialized")
    
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
                logger.error(f"Error analyzing {symbol} with Lynch strategy: {e}")
        
        # Sort by confidence and upside potential
        signals.sort(key=lambda x: (x.get("confidence", 0) * x.get("upside_potential", 0)), reverse=True)
        
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
    
    async def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[LynchAnalysis]:
        """
        Perform Peter Lynch style analysis on a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            market_data: Dictionary of market data
            
        Returns:
            LynchAnalysis object or None if analysis fails
        """
        try:
            # Get fundamental data
            fundamentals = await self.data_manager.get_fundamentals(symbol)
            if not fundamentals:
                return None
            
            # Get price data
            price_data = await self.data_manager.get_price_history(symbol, days=365*3)  # 3 years
            if price_data.empty:
                return None
            
            current_price = price_data.iloc[-1]['close']
            
            # Calculate growth metrics
            earnings_growth = self._calculate_earnings_growth(fundamentals)
            revenue_growth = self._calculate_revenue_growth(fundamentals)
            
            # Calculate PEG ratio
            pe_ratio = fundamentals.get('pe_ratio', 0)
            peg_ratio = calculate_peg_ratio(pe_ratio, earnings_growth * 100)  # Convert to percentage
            
            # Categorize the stock
            category = self._categorize_stock(fundamentals, earnings_growth, revenue_growth)
            
            # Evaluate business story
            story_quality = self._evaluate_business_story(fundamentals, category)
            
            # Calculate target price
            target_price = self._calculate_target_price(fundamentals, earnings_growth, category)
            
            # Calculate upside potential
            upside_potential = (target_price - current_price) / current_price if current_price > 0 else 0
            
            # Generate recommendation
            recommendation, confidence, reasoning = self._generate_recommendation(
                fundamentals, peg_ratio, earnings_growth, revenue_growth,
                category, story_quality, upside_potential
            )
            
            return LynchAnalysis(
                symbol=symbol,
                category=category,
                peg_ratio=peg_ratio,
                earnings_growth=earnings_growth,
                revenue_growth=revenue_growth,
                current_price=current_price,
                target_price=target_price,
                upside_potential=upside_potential,
                story_quality=story_quality,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in Lynch analysis for {symbol}: {e}")
            return None
    
    def _calculate_earnings_growth(self, fundamentals: Dict) -> float:
        """
        Calculate the earnings growth rate.
        
        Args:
            fundamentals: Dictionary of fundamental data
            
        Returns:
            Earnings growth rate as a decimal
        """
        try:
            # Get annual EPS data for the last 5 years
            eps_data = fundamentals.get('annual_eps', [])
            
            if len(eps_data) < 2:
                return 0.0
            
            # Calculate compound annual growth rate
            start_eps = eps_data[0]
            end_eps = eps_data[-1]
            years = len(eps_data) - 1
            
            if start_eps <= 0:
                return 0.0
            
            cagr = (end_eps / start_eps) ** (1 / years) - 1
            return cagr
            
        except Exception as e:
            logger.error(f"Error calculating earnings growth: {e}")
            return 0.0
    
    def _calculate_revenue_growth(self, fundamentals: Dict) -> float:
        """
        Calculate the revenue growth rate.
        
        Args:
            fundamentals: Dictionary of fundamental data
            
        Returns:
            Revenue growth rate as a decimal
        """
        try:
            # Get annual revenue data for the last 5 years
            revenue_data = fundamentals.get('annual_revenue', [])
            
            if len(revenue_data) < 2:
                return 0.0
            
            # Calculate compound annual growth rate
            start_revenue = revenue_data[0]
            end_revenue = revenue_data[-1]
            years = len(revenue_data) - 1
            
            if start_revenue <= 0:
                return 0.0
            
            cagr = (end_revenue / start_revenue) ** (1 / years) - 1
            return cagr
            
        except Exception as e:
            logger.error(f"Error calculating revenue growth: {e}")
            return 0.0
    
    def _categorize_stock(self, fundamentals: Dict, earnings_growth: float, revenue_growth: float) -> str:
        """
        Categorize the stock according to Peter Lynch's categories.
        
        Args:
            fundamentals: Dictionary of fundamental data
            earnings_growth: Earnings growth rate
            revenue_growth: Revenue growth rate
            
        Returns:
            Category string
        """
        # Get key metrics
        market_cap = fundamentals.get('market_cap', 0)
        pe_ratio = fundamentals.get('pe_ratio', 0)
        dividend_yield = fundamentals.get('dividend_yield', 0)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        assets = fundamentals.get('total_assets', 0)
        liabilities = fundamentals.get('total_liabilities', 0)
        
        # Fast Grower: High growth companies (>20% annual growth)
        if earnings_growth > 0.20 and revenue_growth > 0.20:
            return "fast_grower"
        
        # Stalwart: Large, established companies with moderate growth (10-20%)
        if market_cap > 10e9 and 0.10 <= earnings_growth <= 0.20 and dividend_yield > 0:
            return "stalwart"
        
        # Slow Grower: Large companies with slow growth, often with high dividends
        if market_cap > 10e9 and earnings_growth < 0.10 and dividend_yield > 0.03:
            return "slow_grower"
        
        # Cyclical: Companies whose sales and profits rise and fall in regular cycles
        if self._is_cyclical_industry(fundamentals.get('sector', ''), fundamentals.get('industry', '')):
            return "cyclical"
        
        # Turnaround: Companies recovering from hard times
        if pe_ratio < 10 and debt_to_equity > 1.0 and earnings_growth > 0:
            return "turnaround"
        
        # Asset Play: Companies with valuable assets not reflected in stock price
        if (assets - liabilities) > market_cap * 0.8:
            return "asset_play"
        
        # Default to fast grower if high growth but doesn't fit other categories
        if earnings_growth > 0.15 or revenue_growth > 0.15:
            return "fast_grower"
        
        # Default
        return "unknown"
    
    def _is_cyclical_industry(self, sector: str, industry: str) -> bool:
        """
        Determine if a sector/industry is cyclical.
        
        Args:
            sector: Company sector
            industry: Company industry
            
        Returns:
            True if cyclical, False otherwise
        """
        cyclical_sectors = [
            "Energy", "Materials", "Industrials", "Consumer Discretionary",
            "Financials", "Real Estate"
        ]
        
        cyclical_industries = [
            "Automotive", "Airlines", "Hotels", "Construction", "Steel",
            "Semiconductors", "Banking", "Insurance"
        ]
        
        return sector in cyclical_sectors or any(ind in industry for ind in cyclical_industries)
    
    def _evaluate_business_story(self, fundamentals: Dict, category: str) -> float:
        """
        Evaluate the quality of the business story.
        
        Args:
            fundamentals: Dictionary of fundamental data
            category: Stock category
            
        Returns:
            Story quality score (0.0 to 1.0)
        """
        # Start with base score
        score = 0.5
        
        # Adjust based on category
        category_bonus = {
            "fast_grower": 0.2,
            "stalwart": 0.15,
            "slow_grower": 0.05,
            "cyclical": 0.1,
            "turnaround": 0.15,
            "asset_play": 0.1,
            "unknown": 0.0
        }
        
        score += category_bonus.get(category, 0.0)
        
        # Adjust based on fundamentals
        if fundamentals.get('roe', 0) > 0.15:
            score += 0.1
        
        if fundamentals.get('net_margin', 0) > 0.15:
            score += 0.1
        
        if fundamentals.get('debt_to_equity', 0) < 0.5:
            score += 0.1
        
        # Cap at 1.0
        return min(1.0, score)
    
    def _calculate_target_price(self, fundamentals: Dict, earnings_growth: float, category: str) -> float:
        """
        Calculate target price based on Lynch's methodology.
        
        Args:
            fundamentals: Dictionary of fundamental data
            earnings_growth: Earnings growth rate
            category: Stock category
            
        Returns:
            Target price
        """
        try:
            current_price = fundamentals.get('price', 0)
            eps = fundamentals.get('eps', 0)
            pe_ratio = fundamentals.get('pe_ratio', 0)
            
            if eps <= 0 or pe_ratio <= 0:
                return current_price
            
            # Lynch's fair PE ratio is roughly equal to growth rate
            growth_rate_pct = earnings_growth * 100
            
            # Adjust target PE based on category
            category_adjustments = {
                "fast_grower": 1.0,  # Full growth rate
                "stalwart": 0.75,  # 75% of growth rate
                "slow_grower": 0.5,  # 50% of growth rate
                "cyclical": 0.6,  # 60% of growth rate
                "turnaround": 0.8,  # 80% of growth rate
                "asset_play": 0.7,  # 70% of growth rate
                "unknown": 0.5  # 50% of growth rate
            }
            
            adjustment = category_adjustments.get(category, 0.5)
            
            # Calculate fair PE ratio (capped between 5 and 30)
            fair_pe = max(5, min(30, growth_rate_pct * adjustment))
            
            # Calculate target price
            target_price = eps * fair_pe
            
            return target_price
            
        except Exception as e:
            logger.error(f"Error calculating target price: {e}")
            return fundamentals.get('price', 0)
    
    def _generate_recommendation(self, fundamentals: Dict, peg_ratio: float, 
                               earnings_growth: float, revenue_growth: float,
                               category: str, story_quality: float, 
                               upside_potential: float) -> Tuple[str, float, List[str]]:
        """
        Generate a recommendation based on Lynch's criteria.
        
        Args:
            fundamentals: Dictionary of fundamental data
            peg_ratio: PEG ratio
            earnings_growth: Earnings growth rate
            revenue_growth: Revenue growth rate
            category: Stock category
            story_quality: Business story quality score
            upside_potential: Upside potential percentage
            
        Returns:
            Tuple of (recommendation, confidence, reasoning)
        """
        reasoning = []
        points = 0
        max_points = 0
        
        # PEG ratio check
        max_points += 3
        if peg_ratio <= self.thresholds['peg_ratio_max']:
            if peg_ratio <= 0.5:
                points += 3
                reasoning.append(f"Excellent PEG ratio of {peg_ratio:.2f} (below 0.5)")
            elif peg_ratio <= 0.75:
                points += 2
                reasoning.append(f"Very good PEG ratio of {peg_ratio:.2f} (below 0.75)")
            else:
                points += 1
                reasoning.append(f"Good PEG ratio of {peg_ratio:.2f} (below 1.0)")
        else:
            reasoning.append(f"PEG ratio of {peg_ratio:.2f} is above threshold of {self.thresholds['peg_ratio_max']}")
        
        # Earnings growth check
        max_points += 3
        if earnings_growth >= self.thresholds['earnings_growth_min']:
            if earnings_growth >= 0.30:
                points += 3
                reasoning.append(f"Excellent earnings growth of {earnings_growth:.1%} (above 30%)")
            elif earnings_growth >= 0.20:
                points += 2
                reasoning.append(f"Strong earnings growth of {earnings_growth:.1%} (above 20%)")
            else:
                points += 1
                reasoning.append(f"Good earnings growth of {earnings_growth:.1%} (above 15%)")
        else:
            reasoning.append(f"Earnings growth of {earnings_growth:.1%} is below threshold of {self.thresholds['earnings_growth_min']:.1%}")
        
        # Revenue growth check
        max_points += 2
        if revenue_growth >= self.thresholds['revenue_growth_min']:
            if revenue_growth >= 0.20:
                points += 2
                reasoning.append(f"Strong revenue growth of {revenue_growth:.1%} (above 20%)")
            else:
                points += 1
                reasoning.append(f"Good revenue growth of {revenue_growth:.1%} (above 10%)")
        else:
            reasoning.append(f"Revenue growth of {revenue_growth:.1%} is below threshold of {self.thresholds['revenue_growth_min']:.1%}")
        
        # Category check
        max_points += 2
        category_weight = self.category_weights.get(category, 0.5)
        if category_weight >= 0.8:
            points += 2
            reasoning.append(f"Preferred category: {category} (high weight of {category_weight})")
        elif category_weight >= 0.6:
            points += 1
            reasoning.append(f"Acceptable category: {category} (moderate weight of {category_weight})")
        else:
            reasoning.append(f"Less preferred category: {category} (low weight of {category_weight})")
        
        # Story quality check
        max_points += 2
        if story_quality >= self.thresholds['story_quality_min']:
            if story_quality >= 0.9:
                points += 2
                reasoning.append(f"Excellent business story (quality score: {story_quality:.2f})")
            else:
                points += 1
                reasoning.append(f"Good business story (quality score: {story_quality:.2f})")
        else:
            reasoning.append(f"Business story quality of {story_quality:.2f} is below threshold of {self.thresholds['story_quality_min']}")
        
        # Upside potential check
        max_points += 3
        if upside_potential >= self.thresholds['upside_potential_min']:
            if upside_potential >= 0.50:
                points += 3
                reasoning.append(f"Excellent upside potential of {upside_potential:.1%} (above 50%)")
            elif upside_potential >= 0.35:
                points += 2
                reasoning.append(f"Strong upside potential of {upside_potential:.1%} (above 35%)")
            else:
                points += 1
                reasoning.append(f"Good upside potential of {upside_potential:.1%} (above 25%)")
        else:
            reasoning.append(f"Upside potential of {upside_potential:.1%} is below threshold of {self.thresholds['upside_potential_min']:.1%}")
        
        # Calculate confidence
        confidence = points / max_points if max_points > 0 else 0.0
        
        # Determine recommendation
        if confidence >= 0.8:
            recommendation = "BUY"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Strong buy recommendation")
        elif confidence >= 0.6:
            recommendation = "BUY"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Buy recommendation")
        elif confidence >= 0.4:
            recommendation = "HOLD"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Hold recommendation")
        else:
            recommendation = "PASS"
            reasoning.append(f"Overall confidence: {confidence:.2f} - Pass recommendation")
        
        return recommendation, confidence, reasoning
    
    def _create_signal_from_analysis(self, analysis: LynchAnalysis) -> Dict:
        """
        Create a signal dictionary from analysis results.
        
        Args:
            analysis: LynchAnalysis object
            
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
            "category": analysis.category,
            "peg_ratio": analysis.peg_ratio,
            "earnings_growth": analysis.earnings_growth,
            "revenue_growth": analysis.revenue_growth,
            "upside_potential": analysis.upside_potential,
            "timestamp": datetime.now().isoformat()
        }