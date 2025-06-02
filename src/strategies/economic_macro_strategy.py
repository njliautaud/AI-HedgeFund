#!/usr/bin/env python3
"""
Economic Macro Strategy

Implements macroeconomic analysis and Fed policy-driven investment strategy:
- Federal Reserve policy analysis (interest rates, balance sheet)
- Yield curve analysis and inversion signals
- Inflation trends and expectations
- Economic growth indicators (GDP, employment)
- Currency and commodity impacts
- Sector rotation based on economic cycles
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from dataclasses import dataclass
from scipy import stats

from ..core.signals import StrategySignal
from ..utils.economic_indicators import EconomicAnalyzer
from ..utils.yield_curve import YieldCurveAnalyzer
from ..utils.sector_rotation import SectorRotationModel


@dataclass
class MacroEnvironment:
    """Current macroeconomic environment assessment."""
    fed_policy_stance: str  # 'dovish', 'neutral', 'hawkish'
    economic_cycle: str     # 'expansion', 'peak', 'contraction', 'trough'
    inflation_trend: str    # 'rising', 'stable', 'falling'
    yield_curve_shape: str  # 'normal', 'flat', 'inverted'
    risk_sentiment: str     # 'risk_on', 'risk_off', 'neutral'
    dollar_trend: str       # 'strengthening', 'weakening', 'stable'
    commodity_trend: str    # 'rising', 'falling', 'stable'
    confidence_score: float # 0.0 to 1.0


@dataclass
class MacroSignal:
    """Macro-driven investment signal."""
    asset_class: str       # 'equities', 'bonds', 'commodities', 'cash'
    sector: Optional[str]  # Specific sector if equity
    direction: str         # 'overweight', 'underweight', 'neutral'
    strength: float        # Signal strength 0.0 to 1.0
    reasoning: str
    time_horizon: str      # 'short', 'medium', 'long'


class EconomicMacroStrategy:
    """
    Macroeconomic strategy that analyzes Fed policy, economic indicators,
    and market cycles to generate investment signals.
    
    Key components:
    1. Fed policy analysis (rates, balance sheet, communications)
    2. Economic cycle identification
    3. Yield curve analysis
    4. Inflation monitoring
    5. Sector rotation based on macro environment
    """
    
    def __init__(self, config: Dict, data_manager):
        self.config = config
        self.data_manager = data_manager
        self.enabled = True
        
        # Initialize analyzers
        self.economic_analyzer = EconomicAnalyzer()
        self.yield_curve_analyzer = YieldCurveAnalyzer()
        self.sector_rotation_model = SectorRotationModel()
        
        # Fed policy thresholds
        self.fed_thresholds = {
            'rate_change_significant': 0.25,  # 25 bps
            'balance_sheet_change_significant': 0.05,  # 5%
            'inflation_target': 0.02,  # 2%
            'unemployment_natural_rate': 0.04,  # 4%
            'yield_curve_inversion_threshold': -0.10  # -10 bps
        }
        
        # Sector mappings for different macro environments
        self.sector_preferences = {
            'early_cycle': ['XLF', 'XLI', 'XLY', 'XLK'],  # Financials, Industrials, Consumer Disc, Tech
            'mid_cycle': ['XLK', 'XLI', 'XLB', 'XLE'],    # Tech, Industrials, Materials, Energy
            'late_cycle': ['XLE', 'XLB', 'XLF', 'XLV'],   # Energy, Materials, Financials, Healthcare
            'recession': ['XLU', 'XLP', 'XLV', 'TLT'],    # Utilities, Staples, Healthcare, Bonds
            'recovery': ['XLF', 'XLI', 'XLY', 'XLRE']     # Financials, Industrials, Consumer Disc, REITs
        }
        
        logger.info("Economic Macro Strategy initialized")
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Generate macro-driven investment signals."""
        signals = []
        
        try:
            # 1. Analyze current macro environment
            macro_env = await self._analyze_macro_environment(market_data)
            
            # 2. Generate asset allocation signals
            asset_signals = await self._generate_asset_allocation_signals(macro_env, market_data)
            
            # 3. Generate sector rotation signals
            sector_signals = await self._generate_sector_signals(macro_env, market_data)
            
            # 4. Generate specific equity signals based on macro themes
            equity_signals = await self._generate_macro_equity_signals(macro_env, market_data)
            
            # Combine all signals
            all_macro_signals = asset_signals + sector_signals + equity_signals
            
            # Convert to StrategySignal format
            for macro_signal in all_macro_signals:
                strategy_signal = self._convert_to_strategy_signal(macro_signal, macro_env)
                if strategy_signal:
                    signals.append(strategy_signal)
            
            logger.info(f"Generated {len(signals)} macro signals for {macro_env.economic_cycle} cycle")
            
        except Exception as e:
            logger.error(f"Error generating macro signals: {e}")
        
        return signals
    
    async def _analyze_macro_environment(self, market_data: Dict[str, Any]) -> MacroEnvironment:
        """Analyze current macroeconomic environment."""
        economic_data = market_data.get('economic_data', {})
        
        # 1. Fed Policy Analysis
        fed_stance = await self._analyze_fed_policy(economic_data)
        
        # 2. Economic Cycle Analysis
        economic_cycle = await self._identify_economic_cycle(economic_data)
        
        # 3. Inflation Analysis
        inflation_trend = await self._analyze_inflation_trend(economic_data)
        
        # 4. Yield Curve Analysis
        yield_curve_shape = await self._analyze_yield_curve(economic_data)
        
        # 5. Risk Sentiment Analysis
        risk_sentiment = await self._analyze_risk_sentiment(market_data)
        
        # 6. Dollar Trend Analysis
        dollar_trend = await self._analyze_dollar_trend(economic_data)
        
        # 7. Commodity Trend Analysis
        commodity_trend = await self._analyze_commodity_trend(market_data)
        
        # Calculate overall confidence
        confidence = self._calculate_macro_confidence(
            fed_stance, economic_cycle, inflation_trend, yield_curve_shape
        )
        
        return MacroEnvironment(
            fed_policy_stance=fed_stance,
            economic_cycle=economic_cycle,
            inflation_trend=inflation_trend,
            yield_curve_shape=yield_curve_shape,
            risk_sentiment=risk_sentiment,
            dollar_trend=dollar_trend,
            commodity_trend=commodity_trend,
            confidence_score=confidence
        )
    
    async def _analyze_fed_policy(self, economic_data: Dict) -> str:
        """Analyze Federal Reserve policy stance."""
        try:
            # Get Fed Funds Rate
            fed_funds = self.data_manager.get_economic_indicator('FEDFUNDS')
            if fed_funds is None or fed_funds.empty:
                return 'neutral'
            
            # Calculate recent rate changes
            recent_change = fed_funds.iloc[-1] - fed_funds.iloc[-12]  # 12-month change
            
            # Get Fed Balance Sheet
            fed_balance_sheet = self.data_manager.get_economic_indicator('WALCL')
            balance_sheet_change = 0
            if fed_balance_sheet is not None and not fed_balance_sheet.empty:
                balance_sheet_change = (fed_balance_sheet.iloc[-1] - fed_balance_sheet.iloc[-12]) / fed_balance_sheet.iloc[-12]
            
            # Determine stance
            if recent_change > self.fed_thresholds['rate_change_significant']:
                return 'hawkish'
            elif recent_change < -self.fed_thresholds['rate_change_significant']:
                return 'dovish'
            elif balance_sheet_change > self.fed_thresholds['balance_sheet_change_significant']:
                return 'dovish'  # QE is dovish
            elif balance_sheet_change < -self.fed_thresholds['balance_sheet_change_significant']:
                return 'hawkish'  # QT is hawkish
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing Fed policy: {e}")
            return 'neutral'
    
    async def _identify_economic_cycle(self, economic_data: Dict) -> str:
        """Identify current phase of economic cycle."""
        try:
            # Get key indicators
            gdp = self.data_manager.get_economic_indicator('GDP')
            unemployment = self.data_manager.get_economic_indicator('UNRATE')
            
            if gdp is None or unemployment is None:
                return 'expansion'  # Default assumption
            
            # Calculate trends
            gdp_growth = gdp.pct_change(periods=4).iloc[-1]  # YoY growth
            unemployment_trend = unemployment.iloc[-1] - unemployment.iloc[-12]  # 12-month change
            
            # Simple cycle identification logic
            if gdp_growth > 0.02 and unemployment_trend < 0:  # Growing GDP, falling unemployment
                return 'expansion'
            elif gdp_growth > 0.01 and unemployment_trend > 0.5:  # Slowing growth, rising unemployment
                return 'peak'
            elif gdp_growth < 0 and unemployment_trend > 0:  # Negative growth, rising unemployment
                return 'contraction'
            elif gdp_growth < 0.01 and unemployment_trend < -0.5:  # Low growth, falling unemployment
                return 'trough'
            else:
                return 'expansion'
                
        except Exception as e:
            logger.error(f"Error identifying economic cycle: {e}")
            return 'expansion'
    
    async def _analyze_inflation_trend(self, economic_data: Dict) -> str:
        """Analyze inflation trend."""
        try:
            cpi = self.data_manager.get_economic_indicator('CPIAUCSL')
            if cpi is None or cpi.empty:
                return 'stable'
            
            # Calculate YoY inflation rate
            inflation_rate = cpi.pct_change(periods=12).iloc[-1]
            recent_trend = cpi.pct_change(periods=3).iloc[-1] * 4  # Annualized 3-month trend
            
            if recent_trend > inflation_rate + 0.005:  # Rising faster than trend
                return 'rising'
            elif recent_trend < inflation_rate - 0.005:  # Falling faster than trend
                return 'falling'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error analyzing inflation: {e}")
            return 'stable'
    
    async def _analyze_yield_curve(self, economic_data: Dict) -> str:
        """Analyze yield curve shape."""
        try:
            # Get 2-year and 10-year Treasury rates
            two_year = self.data_manager.get_economic_indicator('DGS2')
            ten_year = self.data_manager.get_economic_indicator('DGS10')
            
            if two_year is None or ten_year is None:
                return 'normal'
            
            # Calculate spread
            spread = ten_year.iloc[-1] - two_year.iloc[-1]
            
            if spread < self.fed_thresholds['yield_curve_inversion_threshold']:
                return 'inverted'
            elif spread < 0.5:  # 50 bps
                return 'flat'
            else:
                return 'normal'
                
        except Exception as e:
            logger.error(f"Error analyzing yield curve: {e}")
            return 'normal'
    
    async def _analyze_risk_sentiment(self, market_data: Dict) -> str:
        """Analyze market risk sentiment."""
        try:
            sentiment_data = market_data.get('sentiment_data', {})
            vix_data = sentiment_data.get('vix', {})
            
            if not vix_data:
                return 'neutral'
            
            current_vix = vix_data.get('current_level', 20)
            
            if current_vix > 30:
                return 'risk_off'
            elif current_vix < 15:
                return 'risk_on'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error analyzing risk sentiment: {e}")
            return 'neutral'
    
    async def _analyze_dollar_trend(self, economic_data: Dict) -> str:
        """Analyze US Dollar trend."""
        try:
            # Use EUR/USD as proxy for dollar strength
            eur_usd = self.data_manager.get_economic_indicator('DEXUSEU')
            if eur_usd is None or eur_usd.empty:
                return 'stable'
            
            # Calculate 3-month trend
            recent_change = (eur_usd.iloc[-1] - eur_usd.iloc[-63]) / eur_usd.iloc[-63]  # ~3 months
            
            if recent_change > 0.02:  # EUR strengthening = USD weakening
                return 'weakening'
            elif recent_change < -0.02:  # EUR weakening = USD strengthening
                return 'strengthening'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error analyzing dollar trend: {e}")
            return 'stable'
    
    async def _analyze_commodity_trend(self, market_data: Dict) -> str:
        """Analyze commodity price trends."""
        try:
            # Use oil price as commodity proxy
            oil_price = self.data_manager.get_economic_indicator('DCOILWTICO')
            if oil_price is None or oil_price.empty:
                return 'stable'
            
            # Calculate 3-month trend
            recent_change = (oil_price.iloc[-1] - oil_price.iloc[-63]) / oil_price.iloc[-63]
            
            if recent_change > 0.1:  # 10% increase
                return 'rising'
            elif recent_change < -0.1:  # 10% decrease
                return 'falling'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error analyzing commodities: {e}")
            return 'stable'
    
    def _calculate_macro_confidence(self, fed_stance: str, economic_cycle: str, 
                                  inflation_trend: str, yield_curve: str) -> float:
        """Calculate confidence in macro analysis."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for clear trends
        if fed_stance in ['hawkish', 'dovish']:
            confidence += 0.1
        
        if economic_cycle in ['expansion', 'contraction']:
            confidence += 0.1
        
        if inflation_trend in ['rising', 'falling']:
            confidence += 0.1
        
        if yield_curve in ['normal', 'inverted']:
            confidence += 0.1
        
        return min(confidence, 0.9)
    
    async def _generate_asset_allocation_signals(self, macro_env: MacroEnvironment, 
                                               market_data: Dict) -> List[MacroSignal]:
        """Generate high-level asset allocation signals."""
        signals = []
        
        # Equity allocation based on cycle and Fed policy
        if macro_env.economic_cycle in ['expansion', 'recovery'] and macro_env.fed_policy_stance != 'hawkish':
            signals.append(MacroSignal(
                asset_class='equities',
                sector=None,
                direction='overweight',
                strength=0.7,
                reasoning=f"Favorable for equities: {macro_env.economic_cycle} cycle with {macro_env.fed_policy_stance} Fed",
                time_horizon='medium'
            ))
        elif macro_env.economic_cycle == 'contraction' or macro_env.yield_curve_shape == 'inverted':
            signals.append(MacroSignal(
                asset_class='equities',
                sector=None,
                direction='underweight',
                strength=0.6,
                reasoning=f"Risk-off environment: {macro_env.economic_cycle} cycle, {macro_env.yield_curve_shape} yield curve",
                time_horizon='medium'
            ))
        
        # Bond allocation
        if macro_env.fed_policy_stance == 'dovish' or macro_env.economic_cycle == 'contraction':
            signals.append(MacroSignal(
                asset_class='bonds',
                sector=None,
                direction='overweight',
                strength=0.6,
                reasoning=f"Favorable for bonds: {macro_env.fed_policy_stance} Fed policy",
                time_horizon='medium'
            ))
        elif macro_env.inflation_trend == 'rising' and macro_env.fed_policy_stance == 'hawkish':
            signals.append(MacroSignal(
                asset_class='bonds',
                sector=None,
                direction='underweight',
                strength=0.5,
                reasoning="Rising rates environment unfavorable for bonds",
                time_horizon='short'
            ))
        
        # Commodity allocation
        if macro_env.inflation_trend == 'rising' or macro_env.dollar_trend == 'weakening':
            signals.append(MacroSignal(
                asset_class='commodities',
                sector=None,
                direction='overweight',
                strength=0.5,
                reasoning=f"Inflation hedge: {macro_env.inflation_trend} inflation, {macro_env.dollar_trend} dollar",
                time_horizon='medium'
            ))
        
        return signals
    
    async def _generate_sector_signals(self, macro_env: MacroEnvironment, 
                                     market_data: Dict) -> List[MacroSignal]:
        """Generate sector rotation signals based on economic cycle."""
        signals = []
        
        # Map economic cycle to sector preferences
        cycle_mapping = {
            'expansion': 'mid_cycle',
            'peak': 'late_cycle',
            'contraction': 'recession',
            'trough': 'recovery'
        }
        
        cycle_phase = cycle_mapping.get(macro_env.economic_cycle, 'mid_cycle')
        preferred_sectors = self.sector_preferences.get(cycle_phase, [])
        
        # Generate overweight signals for preferred sectors
        for sector in preferred_sectors:
            strength = 0.6 if macro_env.confidence_score > 0.7 else 0.4
            
            signals.append(MacroSignal(
                asset_class='equities',
                sector=sector,
                direction='overweight',
                strength=strength,
                reasoning=f"Sector rotation for {macro_env.economic_cycle} cycle",
                time_horizon='medium'
            ))
        
        # Special considerations
        if macro_env.fed_policy_stance == 'hawkish':
            # Favor financials in rising rate environment
            signals.append(MacroSignal(
                asset_class='equities',
                sector='XLF',
                direction='overweight',
                strength=0.5,
                reasoning="Rising rates benefit financial sector",
                time_horizon='short'
            ))
        
        if macro_env.inflation_trend == 'rising':
            # Favor energy and materials
            for sector in ['XLE', 'XLB']:
                signals.append(MacroSignal(
                    asset_class='equities',
                    sector=sector,
                    direction='overweight',
                    strength=0.4,
                    reasoning="Inflation benefits commodity-linked sectors",
                    time_horizon='short'
                ))
        
        return signals
    
    async def _generate_macro_equity_signals(self, macro_env: MacroEnvironment, 
                                           market_data: Dict) -> List[MacroSignal]:
        """Generate specific equity signals based on macro themes."""
        signals = []
        
        # Dollar-sensitive stocks
        if macro_env.dollar_trend == 'weakening':
            # Favor exporters and international companies
            export_heavy_stocks = ['AAPL', 'MSFT', 'KO', 'PEP', 'JNJ']
            for stock in export_heavy_stocks:
                signals.append(MacroSignal(
                    asset_class='equities',
                    sector=stock,
                    direction='overweight',
                    strength=0.3,
                    reasoning="Weak dollar benefits exporters",
                    time_horizon='short'
                ))
        
        # Interest rate sensitive stocks
        if macro_env.fed_policy_stance == 'dovish':
            # Favor REITs and utilities
            rate_sensitive = ['XLRE', 'XLU']
            for sector in rate_sensitive:
                signals.append(MacroSignal(
                    asset_class='equities',
                    sector=sector,
                    direction='overweight',
                    strength=0.4,
                    reasoning="Low rates benefit rate-sensitive sectors",
                    time_horizon='medium'
                ))
        
        return signals
    
    def _convert_to_strategy_signal(self, macro_signal: MacroSignal, 
                                  macro_env: MacroEnvironment) -> Optional[StrategySignal]:
        """Convert MacroSignal to StrategySignal format."""
        if macro_signal.direction == 'neutral':
            return None
        
        # Determine action
        action = 'BUY' if macro_signal.direction == 'overweight' else 'SELL'
        
        # Calculate target weight
        base_weight = self.config.get('weight', 0.2)
        target_weight = base_weight * macro_signal.strength
        
        if macro_signal.direction == 'underweight':
            target_weight = base_weight * (1 - macro_signal.strength)
        
        # Use sector ETF or asset class proxy
        symbol = macro_signal.sector if macro_signal.sector else self._get_asset_class_proxy(macro_signal.asset_class)
        
        if not symbol:
            return None
        
        return StrategySignal(
            strategy_name='economic_macro',
            symbol=symbol,
            action=action,
            confidence=macro_signal.strength * macro_env.confidence_score,
            target_weight=target_weight,
            reasoning=macro_signal.reasoning,
            timestamp=datetime.now(),
            metadata={
                'macro_environment': macro_env.__dict__,
                'time_horizon': macro_signal.time_horizon,
                'asset_class': macro_signal.asset_class
            }
        )
    
    def _get_asset_class_proxy(self, asset_class: str) -> Optional[str]:
        """Get ETF symbol for asset class."""
        proxies = {
            'equities': 'SPY',
            'bonds': 'TLT',
            'commodities': 'DBA',
            'cash': 'SHY'
        }
        return proxies.get(asset_class)
    
    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze a symbol from macro perspective."""
        macro_env = await self._analyze_macro_environment(market_data)
        
        # Determine how macro environment affects this symbol
        analysis = {
            'symbol': symbol,
            'macro_environment': macro_env.__dict__,
            'macro_impact': self._assess_macro_impact(symbol, macro_env),
            'recommendation': 'HOLD',
            'confidence': 0.5
        }
        
        return analysis
    
    def _assess_macro_impact(self, symbol: str, macro_env: MacroEnvironment) -> Dict[str, Any]:
        """Assess how macro environment impacts a specific symbol."""
        impact = {
            'fed_policy_impact': 'neutral',
            'cycle_impact': 'neutral',
            'inflation_impact': 'neutral',
            'dollar_impact': 'neutral'
        }
        
        # This would contain detailed logic for how different symbols
        # are affected by various macro factors
        
        return impact