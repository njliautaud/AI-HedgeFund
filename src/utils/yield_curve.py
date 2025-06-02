import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class YieldCurveData:
    """
    Container for yield curve data.
    
    Attributes:
        maturity: Bond maturity in years
        yield_rate: Yield rate
        historical_data: Historical yield data
        metadata: Additional information
    """
    maturity: int
    yield_rate: float
    historical_data: pd.Series
    metadata: Dict[str, Any] = None


class YieldCurveAnalyzer:
    """
    Analyzes yield curve data and its implications for market conditions.
    
    Attributes:
        curve_data: Dictionary of yield curve data by maturity
        lookback_period: Number of periods to analyze trends
    """
    
    def __init__(self, lookback_period: int = 12):
        self.curve_data = {}
        self.lookback_period = lookback_period
        
    def add_yield_data(self, maturity: int, data: pd.Series, metadata: Dict[str, Any] = None):
        """
        Add yield curve data for a specific maturity.
        
        Args:
            maturity: Bond maturity in years
            data: Historical yield data
            metadata: Additional information
        """
        try:
            current_yield = data.iloc[-1]
            
            self.curve_data[maturity] = YieldCurveData(
                maturity=maturity,
                yield_rate=current_yield,
                historical_data=data,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error adding yield data for {maturity} years: {e}")
            
    def calculate_yield_curve(self) -> Dict[str, Any]:
        """
        Calculate current yield curve and its characteristics.
        
        Returns:
            Dictionary of yield curve analysis
        """
        try:
            # Sort by maturity
            sorted_maturities = sorted(self.curve_data.keys())
            yields = [self.curve_data[m].yield_rate for m in sorted_maturities]
            
            # Calculate slope
            if len(yields) >= 2:
                slope = (yields[-1] - yields[0]) / (sorted_maturities[-1] - sorted_maturities[0])
            else:
                slope = 0.0
                
            # Calculate curve shape
            if slope > 0:
                shape = 'steepening'
            elif slope < 0:
                shape = 'inverting'
            else:
                shape = 'flat'
                
            # Calculate spread between key points
            spreads = {}
            for i in range(len(sorted_maturities) - 1):
                spread = yields[i + 1] - yields[i]
                spreads[f"{sorted_maturities[i]}-{sorted_maturities[i + 1]}"] = spread
                
            return {
                'maturities': sorted_maturities,
                'yields': yields,
                'slope': slope,
                'shape': shape,
                'spreads': spreads,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating yield curve: {e}")
            return {
                'maturities': [],
                'yields': [],
                'slope': 0.0,
                'shape': 'unknown',
                'spreads': {},
                'timestamp': datetime.now()
            }
            
    def analyze_curve_inversion(self) -> Dict[str, Any]:
        """
        Analyze yield curve inversion and its implications.
        
        Returns:
            Dictionary of inversion analysis
        """
        try:
            curve = self.calculate_yield_curve()
            
            # Check for inversion
            is_inverted = curve['slope'] < 0
            
            # Calculate inversion depth
            inversion_depth = abs(curve['slope']) if is_inverted else 0.0
            
            # Determine severity
            severity = 'mild' if inversion_depth < 0.25 else 'severe'
            
            return {
                'is_inverted': is_inverted,
                'inversion_depth': inversion_depth,
                'severity': severity,
                'reasoning': f"Slope: {curve['slope']:.2f}, Shape: {curve['shape']}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing curve inversion: {e}")
            return {
                'is_inverted': False,
                'inversion_depth': 0.0,
                'severity': 'unknown',
                'reasoning': f'Error: {e}'
            }
            
    def analyze_term_premium(self) -> Dict[str, Any]:
        """
        Analyze term premium across different maturities.
        
        Returns:
            Dictionary of term premium analysis
        """
        try:
            curve = self.calculate_yield_curve()
            
            # Calculate term premium for each maturity
            term_premiums = {}
            for i in range(len(curve['maturities'])):
                maturity = curve['maturities'][i]
                yield_rate = curve['yields'][i]
                
                # Calculate premium relative to shorter maturities
                premium = yield_rate - curve['yields'][0]  # relative to shortest maturity
                term_premiums[maturity] = premium
                
            return {
                'term_premiums': term_premiums,
                'average_premium': np.mean(list(term_premiums.values())),
                'reasoning': f"Average premium: {np.mean(list(term_premiums.values())):.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing term premium: {e}")
            return {
                'term_premiums': {},
                'average_premium': 0.0,
                'reasoning': f'Error: {e}'
            }
            
    def get_current_conditions(self) -> Dict[str, Any]:
        """
        Get comprehensive yield curve analysis.
        
        Returns:
            Dictionary of yield curve conditions
        """
        try:
            curve = self.calculate_yield_curve()
            inversion = self.analyze_curve_inversion()
            premium = self.analyze_term_premium()
            
            return {
                'yield_curve': curve,
                'inversion': inversion,
                'term_premium': premium,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting yield curve conditions: {e}")
            return {
                'yield_curve': {
                    'maturities': [],
                    'yields': [],
                    'slope': 0.0,
                    'shape': 'unknown',
                    'spreads': {}
                },
                'inversion': {
                    'is_inverted': False,
                    'inversion_depth': 0.0,
                    'severity': 'unknown'
                },
                'term_premium': {
                    'term_premiums': {},
                    'average_premium': 0.0
                },
                'timestamp': datetime.now()
            }
