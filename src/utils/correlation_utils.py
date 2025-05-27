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
            logger.warning("Insufficient data for correlation calculation (less than 20 observations)")
            # Return empty correlation matrix
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