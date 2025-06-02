#!/usr/bin/env python3
"""
Correlation Utilities - Helper functions for correlation and volatility calculations

Provides standalone utility functions for calculating correlation matrices and volatilities
for use in various agents and portfolio optimization strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from loguru import logger


class CorrelationAnalyzer:
    """
    Analyzes correlations between assets and portfolios.
    
    Provides methods for calculating correlation matrices,
    diversification scores, and risk metrics.
    """
    
    def __init__(self, min_obs: int = 20, window: int = 60):
        """
        Initialize correlation analyzer.
        
        Args:
            min_obs: Minimum number of observations required
            window: Rolling window size for calculations
        """
        self.min_obs = min_obs
        self.window = window
        logger.info("Correlation analyzer initialized")

    def calculate_correlation_matrix(self, asset_returns: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix for a dictionary of asset returns.
        
        Args:
            asset_returns: Dictionary mapping asset names to return Series
            
        Returns:
            Dictionary of dictionaries representing the correlation matrix
        """
        try:
            # Convert dictionary of Series to DataFrame
            returns_df = pd.DataFrame(asset_returns).dropna()
            
            if len(returns_df) < self.min_obs:
                logger.warning(f"Insufficient data for correlation calculation (only {len(returns_df)} observations)")
                return {asset: {} for asset in asset_returns.keys()}
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Convert to dictionary format
            result = {}
            for asset1 in corr_matrix.columns:
                result[asset1] = {}
                for asset2 in corr_matrix.columns:
                    result[asset1][asset2] = corr_matrix.loc[asset1, asset2]
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return {asset: {} for asset in asset_returns.keys()}

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all available correlation indicators to a DataFrame.
        
        Args:
            df: Input DataFrame with asset returns
            
        Returns:
            DataFrame with added correlation indicators
        """
        try:
            # Calculate pairwise correlations
            corr_matrix = df.corr()
            
            # Add correlation columns
            for asset1 in df.columns:
                for asset2 in df.columns:
                    if asset1 != asset2:
                        col_name = f"corr_{asset1}_{asset2}"
                        df[col_name] = corr_matrix.loc[asset1, asset2]
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding correlation indicators: {e}")
            return df

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate various risk metrics for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            metrics = {
                'volatility': returns.std() * np.sqrt(252),  # Annualized volatility
                'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
                'max_drawdown': self.calculate_max_drawdown(returns),
                'value_at_risk': self.calculate_value_at_risk(returns)
            }
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'value_at_risk': 0.0
            }

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown for a return series.
        
        Args:
            returns: Series of returns
            
        Returns:
            Maximum drawdown as a percentage
        """
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            return drawdown.min()
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_value_at_risk(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR) for a return series.
        
        Args:
            returns: Series of returns
            confidence: Confidence level for VaR calculation
            
        Returns:
            Value at Risk as a percentage
        """
        try:
            if len(returns) < 20:
                logger.warning("Insufficient data for VaR calculation")
                return 0.0
                
            return returns.quantile(1 - confidence)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0

    def calculate_rolling_correlation(self, asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        """
        Calculate rolling correlation between two assets.
        
        Args:
            asset1: First asset returns series
            asset2: Second asset returns series
            
        Returns:
            Series of rolling correlation values
        """
        try:
            if len(asset1) < self.window or len(asset2) < self.window:
                logger.warning("Insufficient data for rolling correlation calculation")
                return pd.Series()
            
            return asset1.rolling(window=self.window).corr(asset2)
            
        except Exception as e:
            logger.error(f"Error calculating rolling correlation: {e}")
            return pd.Series()

    def score_diversification(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate diversification score based on correlation matrix.
        
        Args:
            correlation_matrix: Dictionary of correlation values
            
        Returns:
            Diversification score (0-100)
        """
        try:
            # Calculate average correlation
            all_correlations = []
            for asset1, values in correlation_matrix.items():
                for asset2, corr in values.items():
                    if asset1 != asset2:
                        all_correlations.append(corr)
            
            if not all_correlations:
                return 0
            
            avg_corr = np.mean(all_correlations)
            
            # Convert to score (lower correlation = better diversification)
            score = 100 * (1 - avg_corr)
            return min(max(score, 0), 100)
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0

def calculate_correlation_matrix(asset_returns: Dict[str, pd.Series]) -> Dict[str, Dict[str, float]]:
    """
    Calculate correlation matrix for a dictionary of asset returns.
    
    Args:
        asset_returns: Dictionary mapping asset names to return Series
            
    Returns:
        Dictionary of dictionaries representing the correlation matrix
    """
    try:
        # Convert dictionary of Series to DataFrame
        returns_df = pd.DataFrame(asset_returns).dropna()
        
        if len(returns_df) < 20:
            logger.warning(f"Insufficient data for correlation calculation (only {len(returns_df)} observations)")
            assets = list(asset_returns.keys())
            return {asset: {other: 1.0 if asset == other else 0.0 for other in assets} for asset in assets}
        
        # Calculate correlation matrix
        corr_matrix_df = returns_df.corr(method='pearson')
        
        # Convert DataFrame to dictionary of dictionaries
        corr_matrix_dict = {}
        for asset1 in corr_matrix_df.index:
            corr_matrix_dict[asset1] = {}
            for asset2 in corr_matrix_df.columns:
                corr_matrix_dict[asset1][asset2] = corr_matrix_df.loc[asset1, asset2]
        
        return corr_matrix_dict
        
    except Exception as e:
        logger.error(f"Error calculating correlation matrix: {e}")
        # Return empty correlation matrix
        assets = list(asset_returns.keys())
        return {asset: {other: 1.0 if asset == other else 0.0 for other in assets} for asset in assets}

def calculate_volatility(returns: pd.Series, annualize: bool = True, trading_days: int = 252) -> float:
    """
    Calculate volatility (standard deviation) of returns.
    
    Args:
        returns: Series of asset returns
        annualize: Whether to annualize the volatility
        trading_days: Number of trading days in a year for annualization
        
    Returns:
        Volatility as a float
    """
    try:
        if returns.empty or len(returns) < 2:
            logger.warning("Insufficient data for volatility calculation")
            return 0.0
        
        # Calculate standard deviation
        volatility = returns.std()
        
        # Annualize if requested
        if annualize:
            volatility = volatility * np.sqrt(trading_days)
        
        return volatility
        
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.0


def calculate_portfolio_volatility(
    weights: Dict[str, float], 
    asset_returns: Dict[str, pd.Series],
    correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculate portfolio volatility using weights, returns, and correlation matrix.
    
    Args:
        weights: Dictionary mapping assets to weights
        asset_returns: Dictionary mapping assets to return Series
        correlation_matrix: Pre-computed correlation matrix (optional)
        annualize: Whether to annualize the volatility
        trading_days: Number of trading days in a year for annualization
        
    Returns:
        Portfolio volatility as a float
    """
    try:
        # Get list of assets with both weights and returns
        assets = [asset for asset in weights if asset in asset_returns]
        
        if not assets:
            logger.warning("No matching assets between weights and returns")
            return 0.0
        
        # Calculate correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = calculate_correlation_matrix(asset_returns)
        
        # Calculate asset volatilities
        volatilities = {asset: calculate_volatility(asset_returns[asset], False) for asset in assets}
        
        # Calculate portfolio variance
        portfolio_variance = 0.0
        for asset_i in assets:
            for asset_j in assets:
                # Get correlation between assets i and j
                corr_ij = correlation_matrix.get(asset_i, {}).get(asset_j, 0.0)
                
                # Calculate contribution to portfolio variance
                contribution = weights[asset_i] * weights[asset_j] * \
                               volatilities[asset_i] * volatilities[asset_j] * corr_ij
                
                portfolio_variance += contribution
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Annualize if requested
        if annualize:
            portfolio_volatility = portfolio_volatility * np.sqrt(trading_days)
        
        return portfolio_volatility
        
    except Exception as e:
        logger.error(f"Error calculating portfolio volatility: {e}")
        return 0.0