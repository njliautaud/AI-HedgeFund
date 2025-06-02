import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class EconomicData:
    """
    Container for economic data.
    
    Attributes:
        indicator: Name of the economic indicator
        value: Current value
        trend: Recent trend (positive, negative, neutral)
        historical_data: Historical data series
        metadata: Additional information
    """
    indicator: str
    value: float
    trend: str
    historical_data: pd.Series
    metadata: Dict[str, Any] = None


class EconomicAnalyzer:
    """
    Analyzes economic indicators and their impact on markets.
    
    Attributes:
        indicators: Dictionary of economic indicators
        lookback_period: Number of periods to analyze trends
    """
    
    def __init__(self, lookback_period: int = 12):
        self.indicators = {}
        self.lookback_period = lookback_period
        
    def add_indicator(self, name: str, data: pd.Series, metadata: Dict[str, Any] = None):
        """
        Add an economic indicator.
        
        Args:
            name: Name of the indicator
            data: Historical data series
            metadata: Additional information
        """
        try:
            current_value = data.iloc[-1]
            trend = self._calculate_trend(data)
            
            self.indicators[name] = EconomicData(
                indicator=name,
                value=current_value,
                trend=trend,
                historical_data=data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error adding indicator {name}: {e}")
            
    def _calculate_trend(self, data: pd.Series) -> str:
        """
        Calculate trend of an economic indicator.
        
        Args:
            data: Historical data series
            
        Returns:
            Trend direction ('positive', 'negative', 'neutral')
        """
        try:
            if len(data) < 3:
                return 'neutral'
                
            # Calculate recent changes
            recent_change = data.iloc[-1] - data.iloc[-2]
            longer_term_change = data.iloc[-1] - data.iloc[-3]
            
            # Determine trend based on recent and longer-term changes
            if recent_change > 0 and longer_term_change > 0:
                return 'positive'
            elif recent_change < 0 and longer_term_change < 0:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 'neutral'
            
    def analyze_fed_policy(self) -> Dict[str, Any]:
        """
        Analyze Federal Reserve policy stance.
        
        Returns:
            Dictionary of Fed policy analysis
        """
        try:
            # Get relevant indicators
            fed_funds = self.indicators.get('Fed Funds Rate', None)
            balance_sheet = self.indicators.get('Fed Balance Sheet', None)
            
            if not fed_funds or not balance_sheet:
                return {
                    'policy_stance': 'unknown',
                    'reasoning': 'Insufficient Fed data available'
                }
                
            # Analyze policy stance
            if fed_funds.trend == 'positive' and balance_sheet.trend == 'negative':
                policy_stance = 'hawkish'
            elif fed_funds.trend == 'negative' and balance_sheet.trend == 'positive':
                policy_stance = 'dovish'
            else:
                policy_stance = 'neutral'
                
            return {
                'policy_stance': policy_stance,
                'reasoning': f"Fed Funds trend: {fed_funds.trend}, Balance Sheet trend: {balance_sheet.trend}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Fed policy: {e}")
            return {
                'policy_stance': 'unknown',
                'reasoning': f'Error in Fed policy analysis: {e}'
            }
            
    def analyze_inflation(self) -> Dict[str, Any]:
        """
        Analyze inflation trends and expectations.
        
        Returns:
            Dictionary of inflation analysis
        """
        try:
            # Get relevant indicators
            cpi = self.indicators.get('CPI', None)
            pce = self.indicators.get('PCE', None)
            breakeven = self.indicators.get('Breakeven Inflation', None)
            
            if not cpi or not pce:
                return {
                    'inflation_trend': 'unknown',
                    'reasoning': 'Insufficient inflation data available'
                }
                
            # Calculate average inflation rate
            avg_inflation = (cpi.value + pce.value) / 2
            
            # Determine trend
            if cpi.trend == 'positive' and pce.trend == 'positive':
                trend = 'upward'
            elif cpi.trend == 'negative' and pce.trend == 'negative':
                trend = 'downward'
            else:
                trend = 'mixed'
                
            return {
                'inflation_rate': avg_inflation,
                'inflation_trend': trend,
                'breakeven_inflation': breakeven.value if breakeven else None,
                'reasoning': f"CPI trend: {cpi.trend}, PCE trend: {pce.trend}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing inflation: {e}")
            return {
                'inflation_trend': 'unknown',
                'reasoning': f'Error in inflation analysis: {e}'
            }
            
    def analyze_growth(self) -> Dict[str, Any]:
        """
        Analyze economic growth indicators.
        
        Returns:
            Dictionary of growth analysis
        """
        try:
            # Get relevant indicators
            gdp = self.indicators.get('GDP', None)
            employment = self.indicators.get('Employment', None)
            industrial_production = self.indicators.get('Industrial Production', None)
            
            if not gdp or not employment:
                return {
                    'growth_trend': 'unknown',
                    'reasoning': 'Insufficient growth data available'
                }
                
            # Calculate growth scores
            gdp_score = 1 if gdp.trend == 'positive' else 0
            employment_score = 1 if employment.trend == 'positive' else 0
            production_score = 1 if industrial_production and industrial_production.trend == 'positive' else 0
            
            # Determine overall trend
            total_score = gdp_score + employment_score + production_score
            if total_score >= 2:
                trend = 'expanding'
            elif total_score <= 1:
                trend = 'contracting'
            else:
                trend = 'stable'
                
            return {
                'growth_trend': trend,
                'reasoning': f"GDP trend: {gdp.trend}, Employment trend: {employment.trend}, Production trend: {industrial_production.trend if industrial_production else 'unknown'}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing growth: {e}")
            return {
                'growth_trend': 'unknown',
                'reasoning': f'Error in growth analysis: {e}'
            }
            
    def get_current_environment(self) -> Dict[str, Any]:
        """
        Get comprehensive economic environment analysis.
        
        Returns:
            Dictionary of economic environment analysis
        """
        try:
            fed_analysis = self.analyze_fed_policy()
            inflation_analysis = self.analyze_inflation()
            growth_analysis = self.analyze_growth()
            
            return {
                'fed_policy': fed_analysis,
                'inflation': inflation_analysis,
                'growth': growth_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting economic environment: {e}")
            return {
                'fed_policy': {'policy_stance': 'unknown', 'reasoning': f'Error: {e}'},
                'inflation': {'inflation_trend': 'unknown', 'reasoning': f'Error: {e}'},
                'growth': {'growth_trend': 'unknown', 'reasoning': f'Error: {e}'},
                'timestamp': datetime.now()
            }
