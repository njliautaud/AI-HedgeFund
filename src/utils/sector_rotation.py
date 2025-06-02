import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

@dataclass
class SectorPerformance:
    """
    Container for sector performance data.
    
    Attributes:
        sector: Sector name
        returns: Historical returns
        momentum: Recent momentum score
        volatility: Volatility measure
        metadata: Additional information
    """
    sector: str
    returns: pd.Series
    momentum: float
    volatility: float
    metadata: Dict[str, Any] = None


class SectorRotationModel:
    """
    Implements sector rotation strategy based on economic cycles and market conditions.
    
    Attributes:
        sectors: List of sectors to analyze
        lookback_period: Number of periods for momentum calculation
        volatility_window: Window for volatility calculation
    """
    
    def __init__(self, lookback_period: int = 12, volatility_window: int = 20):
        self.sectors = []
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        
    def add_sector(self, sector: str, returns: pd.Series, metadata: Dict[str, Any] = None):
        """
        Add sector performance data.
        
        Args:
            sector: Sector name
            returns: Historical returns
            metadata: Additional information
        """
        try:
            # Calculate momentum
            momentum = self._calculate_momentum(returns)
            
            # Calculate volatility
            volatility = self._calculate_volatility(returns)
            
            self.sectors.append(SectorPerformance(
                sector=sector,
                returns=returns,
                momentum=momentum,
                volatility=volatility,
                metadata=metadata
            ))
            
        except Exception as e:
            logger.error(f"Error adding sector {sector}: {e}")
            
    def _calculate_momentum(self, returns: pd.Series) -> float:
        """
        Calculate sector momentum.
        
        Args:
            returns: Historical returns
            
        Returns:
            Momentum score
        """
        try:
            if len(returns) < self.lookback_period:
                return 0.0
                
            # Calculate rolling returns
            rolling_returns = returns.rolling(window=self.lookback_period).mean()
            
            # Get most recent momentum
            momentum = rolling_returns.iloc[-1]
            
            # Normalize momentum
            if len(rolling_returns) > 1:
                momentum = (momentum - rolling_returns.mean()) / rolling_returns.std()
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
            
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """
        Calculate sector volatility.
        
        Args:
            returns: Historical returns
            
        Returns:
            Volatility score
        """
        try:
            if len(returns) < self.volatility_window:
                return 0.0
                
            # Calculate rolling volatility
            volatility = returns.rolling(window=self.volatility_window).std()
            
            # Get most recent volatility
            return volatility.iloc[-1]
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
            
    def analyze_sector_performance(self) -> Dict[str, Any]:
        """
        Analyze sector performance and rotation opportunities.
        
        Returns:
            Dictionary of sector analysis
        """
        try:
            if not self.sectors:
                return {
                    'top_sectors': [],
                    'bottom_sectors': [],
                    'reasoning': 'No sectors available for analysis'
                }
                
            # Sort sectors by momentum
            sorted_sectors = sorted(
                self.sectors,
                key=lambda x: x.momentum,
                reverse=True
            )
            
            # Calculate momentum and volatility rankings
            momentum_ranks = {
                s.sector: rank + 1
                for rank, s in enumerate(sorted_sectors)
            }
            
            # Sort by volatility
            sorted_by_volatility = sorted(
                self.sectors,
                key=lambda x: x.volatility
            )
            
            volatility_ranks = {
                s.sector: rank + 1
                for rank, s in enumerate(sorted_by_volatility)
            }
            
            # Calculate composite score (momentum - volatility)
            composite_scores = {
                s.sector: s.momentum - s.volatility
                for s in self.sectors
            }
            
            # Get top and bottom sectors
            n_sectors = len(self.sectors)
            top_n = max(1, n_sectors // 3)
            
            top_sectors = sorted_sectors[:top_n]
            bottom_sectors = sorted_sectors[-top_n:]
            
            return {
                'top_sectors': [
                    {
                        'sector': s.sector,
                        'momentum': s.momentum,
                        'volatility': s.volatility,
                        'composite_score': composite_scores[s.sector]
                    }
                    for s in top_sectors
                ],
                'bottom_sectors': [
                    {
                        'sector': s.sector,
                        'momentum': s.momentum,
                        'volatility': s.volatility,
                        'composite_score': composite_scores[s.sector]
                    }
                    for s in bottom_sectors
                ],
                'momentum_ranks': momentum_ranks,
                'volatility_ranks': volatility_ranks,
                'composite_scores': composite_scores,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector performance: {e}")
            return {
                'top_sectors': [],
                'bottom_sectors': [],
                'momentum_ranks': {},
                'volatility_ranks': {},
                'composite_scores': {},
                'timestamp': datetime.now()
            }
            
    def get_rotation_recommendations(self) -> Dict[str, Any]:
        """
        Generate sector rotation recommendations based on current market conditions.
        
        Returns:
            Dictionary of rotation recommendations
        """
        try:
            analysis = self.analyze_sector_performance()
            
            # Determine rotation strategy
            if not analysis['top_sectors']:
                return {
                    'strategy': 'hold',
                    'reasoning': 'Insufficient data for rotation'
                }
                
            # Calculate average momentum and volatility
            avg_momentum = np.mean([s['momentum'] for s in analysis['top_sectors']])
            avg_volatility = np.mean([s['volatility'] for s in analysis['top_sectors']])
            
            # Determine market regime
            if avg_momentum > 0 and avg_volatility < 0.1:
                regime = 'bull_market'
            elif avg_momentum < 0 and avg_volatility > 0.2:
                regime = 'bear_market'
            else:
                regime = 'range_bound'
                
            # Generate recommendations
            recommendations = []
            for sector in analysis['top_sectors']:
                weight = 1.0 / len(analysis['top_sectors'])
                recommendations.append({
                    'sector': sector['sector'],
                    'weight': weight,
                    'reasoning': f"Strong momentum ({sector['momentum']:.2f}) and low volatility ({sector['volatility']:.2f})"
                })
                
            return {
                'strategy': regime,
                'recommendations': recommendations,
                'market_regime': regime,
                'reasoning': f"Market regime: {regime}, Average momentum: {avg_momentum:.2f}, Average volatility: {avg_volatility:.2f}"
            }
            
        except Exception as e:
            logger.error(f"Error generating rotation recommendations: {e}")
            return {
                'strategy': 'hold',
                'recommendations': [],
                'market_regime': 'unknown',
                'reasoning': f'Error: {e}'
            }
