#!/usr/bin/env python3
"""
Ray Dalio Agent - Implementation of Ray Dalio's All Weather portfolio strategy

This agent implements Ray Dalio's All Weather approach:
- Focus on balancing risk across different economic environments
- Asset allocation based on economic regimes (growth, inflation, etc.)
- Risk parity principles for portfolio construction
- Emphasis on uncorrelated asset classes
- Volatility targeting for risk management
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
from ..utils.risk_metrics import calculate_correlation_matrix, calculate_volatility


@dataclass
class EconomicRegime:
    """Represents an economic regime for the All Weather strategy."""
    name: str  # e.g., "Rising Growth", "Rising Inflation", etc.
    probability: float  # Probability of this regime (0.0 to 1.0)
    indicators: Dict[str, float]  # Economic indicators and their values
    optimal_allocation: Dict[str, float]  # Optimal asset allocation for this regime


@dataclass
class DalioAnalysis:
    """Detailed analysis results for Ray Dalio's All Weather strategy."""
    timestamp: datetime
    current_regime: EconomicRegime
    regime_probabilities: Dict[str, float]
    asset_correlations: Dict[str, Dict[str, float]]
    asset_volatilities: Dict[str, float]
    recommended_allocation: Dict[str, float]
    risk_contribution: Dict[str, float]
    confidence: float
    reasoning: List[str]


class RayDalioAgent(Agent):
    """
    Agent implementation of Ray Dalio's All Weather portfolio strategy.
    
    Key principles:
    1. Economic regime identification
    2. Risk parity portfolio construction
    3. Volatility targeting
    4. Correlation analysis
    5. Balanced risk exposure across asset classes
    """
    
    def __init__(self, config: Dict, data_manager):
        """
        Initialize the Ray Dalio Agent.
        
        Args:
            config: Configuration dictionary
            data_manager: Data manager instance for accessing market data
        """
        super().__init__("ray_dalio", config)
        
        self.data_manager = data_manager
        
        # Define economic regimes
        self.regimes = {
            "rising_growth": {
                "description": "Economic growth increasing",
                "allocation": {
                    "equities": 0.40,
                    "long_term_bonds": 0.15,
                    "intermediate_bonds": 0.10,
                    "commodities": 0.20,
                    "gold": 0.15
                }
            },
            "falling_growth": {
                "description": "Economic growth decreasing",
                "allocation": {
                    "equities": 0.15,
                    "long_term_bonds": 0.40,
                    "intermediate_bonds": 0.20,
                    "commodities": 0.10,
                    "gold": 0.15
                }
            },
            "rising_inflation": {
                "description": "Inflation increasing",
                "allocation": {
                    "equities": 0.15,
                    "long_term_bonds": 0.10,
                    "intermediate_bonds": 0.10,
                    "commodities": 0.40,
                    "gold": 0.25
                }
            },
            "falling_inflation": {
                "description": "Inflation decreasing",
                "allocation": {
                    "equities": 0.25,
                    "long_term_bonds": 0.25,
                    "intermediate_bonds": 0.20,
                    "commodities": 0.15,
                    "gold": 0.15
                }
            }
        }
        
        # Configuration parameters
        self.volatility_target = config.get('volatility_target', 0.10)  # 10% annualized volatility target
        self.lookback_days = config.get('lookback_days', 252)  # 1 year of trading days
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)  # 5% threshold for rebalancing
        
        # Asset class mappings (tickers representing each asset class)
        self.asset_class_tickers = config.get('asset_class_tickers', {
            "equities": ["SPY", "QQQ", "IWM"],
            "long_term_bonds": ["TLT", "GOVT"],
            "intermediate_bonds": ["IEF", "MBB"],
            "commodities": ["DBC", "GSG"],
            "gold": ["GLD", "IAU"]
        })
        
        # Current portfolio allocation
        self.current_allocation = {}
        
        # Last analysis result
        self.last_analysis = None
        
        logger.info("Ray Dalio Agent initialized")
    
    async def process(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Process market data and generate investment signals.
        
        Args:
            inputs: Dictionary containing market data and economic indicators
            
        Returns:
            An AgentAction with investment signals
        """
        await self.update_state("ANALYZING")
        
        market_data = inputs.get("market_data", {})
        economic_data = inputs.get("economic_data", {})
        current_allocation = inputs.get("current_allocation", self.current_allocation)
        
        # Update current allocation if provided
        if current_allocation:
            self.current_allocation = current_allocation
        
        try:
            # Perform All Weather analysis
            analysis = await self.analyze_market_conditions(market_data, economic_data)
            
            if analysis:
                self.last_analysis = analysis
                
                # Generate rebalancing signals if needed
                signals = self._generate_rebalancing_signals(analysis)
                
                action = AgentAction(
                    action_type="SIGNAL",
                    content={
                        "signals": signals,
                        "agent_id": self.agent_id,
                        "timestamp": datetime.now().isoformat(),
                        "analysis": {
                            "current_regime": analysis.current_regime.name,
                            "regime_probabilities": analysis.regime_probabilities,
                            "recommended_allocation": analysis.recommended_allocation,
                            "confidence": analysis.confidence
                        }
                    }
                )
                
                await self.update_state("IDLE", action)
                return action
            else:
                logger.warning("Ray Dalio analysis failed to produce results")
                
        except Exception as e:
            logger.error(f"Error in Ray Dalio analysis: {e}")
        
        # Return empty signals if analysis fails
        action = AgentAction(
            action_type="SIGNAL",
            content={
                "signals": [],
                "agent_id": self.agent_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Analysis failed to produce actionable signals"
            }
        )
        
        await self.update_state("IDLE", action)
        return action
    
    async def analyze_market_conditions(self, market_data: Dict, economic_data: Dict) -> Optional[DalioAnalysis]:
        """
        Analyze market conditions using Ray Dalio's All Weather approach.
        
        Args:
            market_data: Dictionary of market data
            economic_data: Dictionary of economic indicators
            
        Returns:
            DalioAnalysis object or None if analysis fails
        """
        try:
            # Identify current economic regime
            regime_probabilities = self._identify_economic_regime(economic_data)
            current_regime_name = max(regime_probabilities, key=regime_probabilities.get)
            current_regime = EconomicRegime(
                name=current_regime_name,
                probability=regime_probabilities[current_regime_name],
                indicators=economic_data,
                optimal_allocation=self.regimes[current_regime_name]["allocation"]
            )
            
            # Get price data for all asset classes
            asset_price_data = {}
            all_tickers = [ticker for tickers in self.asset_class_tickers.values() for ticker in tickers]
            
            for ticker in all_tickers:
                price_data = await self.data_manager.get_price_history(ticker, days=self.lookback_days)
                if not price_data.empty:
                    asset_price_data[ticker] = price_data
            
            # Calculate asset class returns and statistics
            asset_class_returns = self._calculate_asset_class_returns(asset_price_data)
            
            # Calculate correlation matrix
            correlation_matrix = calculate_correlation_matrix(asset_class_returns)
            
            # Calculate volatilities
            volatilities = {asset: calculate_volatility(returns) for asset, returns in asset_class_returns.items()}
            
            # Construct risk parity portfolio
            risk_parity_allocation = self._construct_risk_parity_portfolio(
                asset_class_returns, correlation_matrix, volatilities
            )
            
            # Blend regime-based allocation with risk parity allocation
            regime_confidence = current_regime.probability
            recommended_allocation = {}
            
            for asset in self.regimes[current_regime_name]["allocation"]:
                regime_weight = self.regimes[current_regime_name]["allocation"][asset]
                risk_parity_weight = risk_parity_allocation.get(asset, 0.0)
                
                # Blend based on regime confidence
                recommended_allocation[asset] = (
                    regime_weight * regime_confidence + 
                    risk_parity_weight * (1 - regime_confidence)
                )
            
            # Calculate risk contribution
            risk_contribution = self._calculate_risk_contribution(
                recommended_allocation, correlation_matrix, volatilities
            )
            
            # Generate reasoning
            reasoning = self._generate_analysis_reasoning(
                current_regime, regime_probabilities, volatilities, correlation_matrix,
                recommended_allocation, risk_contribution
            )
            
            return DalioAnalysis(
                timestamp=datetime.now(),
                current_regime=current_regime,
                regime_probabilities=regime_probabilities,
                asset_correlations=correlation_matrix,
                asset_volatilities=volatilities,
                recommended_allocation=recommended_allocation,
                risk_contribution=risk_contribution,
                confidence=regime_confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error in All Weather analysis: {e}")
            return None
    
    def _identify_economic_regime(self, economic_data: Dict) -> Dict[str, float]:
        """
        Identify the current economic regime based on economic indicators.
        
        Args:
            economic_data: Dictionary of economic indicators
            
        Returns:
            Dictionary mapping regime names to probabilities
        """
        # Extract key economic indicators
        gdp_growth = economic_data.get('gdp_growth', 0.0)
        inflation = economic_data.get('inflation', 0.0)
        inflation_change = economic_data.get('inflation_change', 0.0)
        gdp_growth_change = economic_data.get('gdp_growth_change', 0.0)
        
        # Calculate regime probabilities
        regime_probabilities = {
            "rising_growth": 0.25,  # Base probability
            "falling_growth": 0.25,
            "rising_inflation": 0.25,
            "falling_inflation": 0.25
        }
        
        # Adjust based on GDP growth trend
        if gdp_growth_change > 0:
            regime_probabilities["rising_growth"] += 0.2
            regime_probabilities["falling_growth"] -= 0.2
        else:
            regime_probabilities["rising_growth"] -= 0.2
            regime_probabilities["falling_growth"] += 0.2
        
        # Adjust based on inflation trend
        if inflation_change > 0:
            regime_probabilities["rising_inflation"] += 0.2
            regime_probabilities["falling_inflation"] -= 0.2
        else:
            regime_probabilities["rising_inflation"] -= 0.2
            regime_probabilities["falling_inflation"] += 0.2
        
        # Ensure probabilities are between 0 and 1
        for regime in regime_probabilities:
            regime_probabilities[regime] = max(0.0, min(1.0, regime_probabilities[regime]))
        
        # Normalize probabilities to sum to 1
        total_probability = sum(regime_probabilities.values())
        if total_probability > 0:
            for regime in regime_probabilities:
                regime_probabilities[regime] /= total_probability
        
        return regime_probabilities
    
    def _calculate_asset_class_returns(self, asset_price_data: Dict) -> Dict[str, pd.Series]:
        """
        Calculate returns for each asset class by averaging constituent tickers.
        
        Args:
            asset_price_data: Dictionary mapping tickers to price DataFrames
            
        Returns:
            Dictionary mapping asset classes to return Series
        """
        asset_class_returns = {}
        
        for asset_class, tickers in self.asset_class_tickers.items():
            # Get returns for each ticker in this asset class
            ticker_returns = []
            
            for ticker in tickers:
                if ticker in asset_price_data:
                    # Calculate daily returns
                    prices = asset_price_data[ticker]['close']
                    returns = prices.pct_change().dropna()
                    ticker_returns.append(returns)
            
            if ticker_returns:
                # Average returns across tickers in this asset class
                asset_class_returns[asset_class] = pd.concat(ticker_returns, axis=1).mean(axis=1)
        
        return asset_class_returns
    
    def _construct_risk_parity_portfolio(self, asset_returns: Dict[str, pd.Series],
                                       correlation_matrix: Dict[str, Dict[str, float]],
                                       volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        Construct a risk parity portfolio where each asset contributes equally to risk.
        
        Args:
            asset_returns: Dictionary mapping asset classes to return Series
            correlation_matrix: Asset correlation matrix
            volatilities: Asset volatilities
            
        Returns:
            Dictionary mapping asset classes to allocation weights
        """
        # Simple risk parity implementation
        # In a full implementation, this would use numerical optimization
        
        # Inverse volatility weighting as a simple approximation of risk parity
        inv_vols = {asset: 1.0 / vol if vol > 0 else 0.0 for asset, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol > 0:
            allocation = {asset: inv_vol / total_inv_vol for asset, inv_vol in inv_vols.items()}
        else:
            # Equal weight fallback
            num_assets = len(volatilities)
            allocation = {asset: 1.0 / num_assets for asset in volatilities}
        
        return allocation
    
    def _calculate_risk_contribution(self, allocation: Dict[str, float],
                                   correlation_matrix: Dict[str, Dict[str, float]],
                                   volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate the risk contribution of each asset in the portfolio.
        
        Args:
            allocation: Asset allocation weights
            correlation_matrix: Asset correlation matrix
            volatilities: Asset volatilities
            
        Returns:
            Dictionary mapping asset classes to risk contribution percentages
        """
        risk_contribution = {}
        total_risk = 0.0
        
        # Calculate marginal risk contribution for each asset
        for asset_i in allocation:
            contrib_i = 0.0
            
            for asset_j in allocation:
                # Get correlation between assets i and j
                corr_ij = correlation_matrix.get(asset_i, {}).get(asset_j, 0.0)
                
                # Calculate contribution to portfolio variance
                contrib_i += allocation[asset_i] * allocation[asset_j] * \
                            volatilities[asset_i] * volatilities[asset_j] * corr_ij
            
            risk_contribution[asset_i] = contrib_i
            total_risk += contrib_i
        
        # Normalize to percentages
        if total_risk > 0:
            for asset in risk_contribution:
                risk_contribution[asset] = risk_contribution[asset] / total_risk
        
        return risk_contribution
    
    def _generate_rebalancing_signals(self, analysis: DalioAnalysis) -> List[Dict]:
        """
        Generate rebalancing signals based on the analysis.
        
        Args:
            analysis: DalioAnalysis object
            
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        # Only generate signals if we have current allocation data
        if not self.current_allocation:
            return signals
        
        # Check if rebalancing is needed
        for asset, target_weight in analysis.recommended_allocation.items():
            current_weight = self.current_allocation.get(asset, 0.0)
            
            # Calculate deviation from target
            deviation = abs(current_weight - target_weight)
            
            # Generate signal if deviation exceeds threshold
            if deviation > self.rebalance_threshold:
                action = "BUY" if target_weight > current_weight else "SELL"
                
                # Get representative ticker for this asset class
                tickers = self.asset_class_tickers.get(asset, [])
                if not tickers:
                    continue
                
                # Use first ticker as representative
                ticker = tickers[0]
                
                # Calculate allocation change
                allocation_change = abs(target_weight - current_weight)
                
                signals.append({
                    "agent_id": self.agent_id,
                    "symbol": ticker,
                    "asset_class": asset,
                    "action": action,
                    "allocation_change": allocation_change,
                    "current_weight": current_weight,
                    "target_weight": target_weight,
                    "confidence": analysis.confidence,
                    "reasoning": f"Rebalancing {asset} from {current_weight:.2%} to {target_weight:.2%} due to {analysis.current_regime.name} regime",
                    "timestamp": datetime.now().isoformat()
                })
        
        return signals
    
    def _generate_analysis_reasoning(self, current_regime: EconomicRegime,
                                   regime_probabilities: Dict[str, float],
                                   volatilities: Dict[str, float],
                                   correlation_matrix: Dict[str, Dict[str, float]],
                                   recommended_allocation: Dict[str, float],
                                   risk_contribution: Dict[str, float]) -> List[str]:
        """
        Generate reasoning for the analysis.
        
        Args:
            current_regime: Current economic regime
            regime_probabilities: Regime probabilities
            volatilities: Asset volatilities
            correlation_matrix: Asset correlation matrix
            recommended_allocation: Recommended asset allocation
            risk_contribution: Risk contribution of each asset
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Regime identification reasoning
        reasoning.append(f"Current economic regime identified as {current_regime.name} with {current_regime.probability:.1%} confidence")
        
        # Add regime probabilities
        regime_prob_str = ", ".join([f"{regime}: {prob:.1%}" for regime, prob in regime_probabilities.items()])
        reasoning.append(f"Regime probabilities: {regime_prob_str}")
        
        # Volatility insights
        high_vol_assets = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)[:2]
        low_vol_assets = sorted(volatilities.items(), key=lambda x: x[1])[:2]
        
        reasoning.append(f"Highest volatility assets: {high_vol_assets[0][0]} ({high_vol_assets[0][1]:.1%}), {high_vol_assets[1][0]} ({high_vol_assets[1][1]:.1%})")
        reasoning.append(f"Lowest volatility assets: {low_vol_assets[0][0]} ({low_vol_assets[0][1]:.1%}), {low_vol_assets[1][0]} ({low_vol_assets[1][1]:.1%})")
        
        # Correlation insights
        min_corr = 1.0
        min_corr_pair = (None, None)
        
        for asset_i in correlation_matrix:
            for asset_j in correlation_matrix[asset_i]:
                if asset_i != asset_j and correlation_matrix[asset_i][asset_j] < min_corr:
                    min_corr = correlation_matrix[asset_i][asset_j]
                    min_corr_pair = (asset_i, asset_j)
        
        if min_corr_pair[0] and min_corr_pair[1]:
            reasoning.append(f"Lowest correlation between {min_corr_pair[0]} and {min_corr_pair[1]} ({min_corr:.2f})")
        
        # Allocation reasoning
        top_allocations = sorted(recommended_allocation.items(), key=lambda x: x[1], reverse=True)[:3]
        allocation_str = ", ".join([f"{asset}: {weight:.1%}" for asset, weight in top_allocations])
        reasoning.append(f"Top allocations: {allocation_str}")
        
        # Risk contribution reasoning
        risk_str = ", ".join([f"{asset}: {risk:.1%}" for asset, risk in risk_contribution.items()])
        reasoning.append(f"Risk contribution: {risk_str}")
        
        return reasoning