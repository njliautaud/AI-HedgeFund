#!/usr/bin/env python3
"""
Risk Metrics Calculator - Comprehensive Risk Assessment

Calculates various risk metrics:
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Sharpe Ratio
- Sortino Ratio
- Beta and Alpha
- Volatility measures
- Correlation analysis
- Risk attribution
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'cvar_99': self.cvar_99,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'beta': self.beta,
            'alpha': self.alpha,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis
        }


@dataclass
class RiskDecomposition:
    """Risk decomposition by position."""
    position_risk: Dict[str, float]
    marginal_var: Dict[str, float]
    component_var: Dict[str, float]
    diversification_ratio: float
    concentration_risk: float


class RiskCalculator:
    """
    Comprehensive risk metrics calculator.
    
    Provides various risk measures for portfolio analysis:
    - Downside risk measures (VaR, CVaR)
    - Performance ratios (Sharpe, Sortino, Calmar)
    - Market risk measures (Beta, Alpha)
    - Distribution statistics
    - Risk decomposition
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        logger.info(f"RiskCalculator initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def calculate_portfolio_risk(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a return series.
        
        Args:
            returns: Portfolio return series
            benchmark_returns: Benchmark return series for relative metrics
            confidence_levels: Confidence levels for VaR/CVaR calculation
        
        Returns:
            RiskMetrics object with calculated metrics
        """
        try:
            if len(returns) < 10:
                logger.warning("Insufficient data for reliable risk calculation")
                return RiskMetrics()
            
            # Clean data
            returns = returns.dropna()
            
            # Basic statistics
            volatility = self._calculate_volatility(returns)
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)
            
            # Downside risk measures
            var_95, var_99 = self._calculate_var(returns, confidence_levels)
            cvar_95, cvar_99 = self._calculate_cvar(returns, confidence_levels)
            
            # Drawdown measures
            max_drawdown = self._calculate_max_drawdown(returns)
            
            # Performance ratios
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
            
            # Market risk measures (if benchmark provided)
            beta = alpha = tracking_error = information_ratio = 0.0
            if benchmark_returns is not None and len(benchmark_returns) > 0:
                beta, alpha = self._calculate_beta_alpha(returns, benchmark_returns)
                tracking_error = self._calculate_tracking_error(returns, benchmark_returns)
                information_ratio = self._calculate_information_ratio(returns, benchmark_returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return RiskMetrics()
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(252)
    
    def _calculate_skewness(self, returns: pd.Series) -> float:
        """Calculate skewness of returns."""
        return stats.skew(returns.dropna())
    
    def _calculate_kurtosis(self, returns: pd.Series) -> float:
        """Calculate excess kurtosis of returns."""
        return stats.kurtosis(returns.dropna())
    
    def _calculate_var(self, returns: pd.Series, confidence_levels: List[float]) -> Tuple[float, float]:
        """
        Calculate Value at Risk using historical simulation.
        
        Returns VaR at 95% and 99% confidence levels.
        """
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) < 20:
                return 0.0, 0.0
            
            # Historical VaR
            var_95 = -np.percentile(clean_returns, (1 - 0.95) * 100)
            var_99 = -np.percentile(clean_returns, (1 - 0.99) * 100)
            
            return var_95, var_99
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0, 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence_levels: List[float]) -> Tuple[float, float]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Returns CVaR at 95% and 99% confidence levels.
        """
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) < 20:
                return 0.0, 0.0
            
            # CVaR = average of returns below VaR threshold
            var_95_threshold = np.percentile(clean_returns, 5)
            var_99_threshold = np.percentile(clean_returns, 1)
            
            cvar_95 = -clean_returns[clean_returns <= var_95_threshold].mean()
            cvar_99 = -clean_returns[clean_returns <= var_99_threshold].mean()
            
            return cvar_95, cvar_99
            
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0, 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        """
        try:
            # Calculate cumulative returns
            cumulative = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            # Return maximum drawdown (most negative value)
            return abs(drawdown.min())
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.
        """
        try:
            excess_returns = returns - self.risk_free_rate / 252
            
            if excess_returns.std() == 0:
                return 0.0
            
            return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        """
        try:
            excess_returns = returns - self.risk_free_rate / 252
            
            # Downside deviation (only negative returns)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                return 0.0
            
            downside_deviation = downside_returns.std() * np.sqrt(252)
            
            return (excess_returns.mean() * 252) / downside_deviation
            
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        """
        try:
            if max_drawdown == 0:
                return 0.0
            
            annual_return = returns.mean() * 252
            return annual_return / max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _calculate_beta_alpha(self, returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate beta and alpha relative to benchmark.
        """
        try:
            # Align series
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            
            if len(aligned_data) < 20:
                return 0.0, 0.0
            
            portfolio_returns = aligned_data.iloc[:, 0]
            benchmark_returns = aligned_data.iloc[:, 1]
            
            # Calculate beta using linear regression
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            if benchmark_variance == 0:
                return 0.0, 0.0
            
            beta = covariance / benchmark_variance
            
            # Calculate alpha
            portfolio_mean = portfolio_returns.mean() * 252
            benchmark_mean = benchmark_returns.mean() * 252
            alpha = portfolio_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
            
            return beta, alpha
            
        except Exception as e:
            logger.error(f"Error calculating beta/alpha: {e}")
            return 0.0, 0.0
    
    def _calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error (volatility of excess returns).
        """
        try:
            # Align series
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return 0.0
            
            portfolio_returns = aligned_data.iloc[:, 0]
            benchmark_returns = aligned_data.iloc[:, 1]
            
            # Excess returns
            excess_returns = portfolio_returns - benchmark_returns
            
            # Annualized tracking error
            return excess_returns.std() * np.sqrt(252)
            
        except Exception as e:
            logger.error(f"Error calculating tracking error: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio (excess return / tracking error).
        """
        try:
            # Align series
            aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
            
            if len(aligned_data) < 10:
                return 0.0
            
            portfolio_returns = aligned_data.iloc[:, 0]
            benchmark_returns = aligned_data.iloc[:, 1]
            
            # Excess returns
            excess_returns = portfolio_returns - benchmark_returns
            
            if excess_returns.std() == 0:
                return 0.0
            
            # Information ratio
            excess_return_annual = excess_returns.mean() * 252
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            return excess_return_annual / tracking_error
            
        except Exception as e:
            logger.error(f"Error calculating information ratio: {e}")
            return 0.0
    
    def calculate_position_risk(
        self,
        weights: Dict[str, float],
        returns_data: Dict[str, pd.Series],
        covariance_matrix: Optional[np.ndarray] = None
    ) -> RiskDecomposition:
        """
        Calculate risk decomposition by position.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns for each position
            covariance_matrix: Precomputed covariance matrix
        
        Returns:
            RiskDecomposition object
        """
        try:
            symbols = list(weights.keys())
            weight_vector = np.array([weights[symbol] for symbol in symbols])
            
            # Build returns matrix
            returns_df = pd.DataFrame(returns_data)[symbols].dropna()
            
            if len(returns_df) < 20:
                logger.warning("Insufficient data for position risk calculation")
                return RiskDecomposition({}, {}, {}, 0.0, 0.0)
            
            # Calculate covariance matrix if not provided
            if covariance_matrix is None:
                covariance_matrix = returns_df.cov().values * 252  # Annualized
            
            # Portfolio variance
            portfolio_variance = np.dot(weight_vector, np.dot(covariance_matrix, weight_vector))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Individual position volatilities
            position_volatilities = np.sqrt(np.diag(covariance_matrix))
            position_risk = {symbol: vol for symbol, vol in zip(symbols, position_volatilities)}
            
            # Marginal VaR (contribution to portfolio risk)
            marginal_var = np.dot(covariance_matrix, weight_vector) / portfolio_volatility
            marginal_var_dict = {symbol: mv for symbol, mv in zip(symbols, marginal_var)}
            
            # Component VaR (marginal VaR * weight)
            component_var = weight_vector * marginal_var
            component_var_dict = {symbol: cv for symbol, cv in zip(symbols, component_var)}
            
            # Diversification ratio
            weighted_avg_vol = np.dot(weight_vector, position_volatilities)
            diversification_ratio = weighted_avg_vol / portfolio_volatility if portfolio_volatility > 0 else 1.0
            
            # Concentration risk (Herfindahl index)
            concentration_risk = np.sum(weight_vector ** 2)
            
            return RiskDecomposition(
                position_risk=position_risk,
                marginal_var=marginal_var_dict,
                component_var=component_var_dict,
                diversification_ratio=diversification_ratio,
                concentration_risk=concentration_risk
            )
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return RiskDecomposition({}, {}, {}, 0.0, 0.0)
    
    def calculate_var_monte_carlo(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
        num_simulations: int = 10000,
        time_horizon: int = 1
    ) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level for VaR
            num_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
        
        Returns:
            VaR estimate
        """
        try:
            clean_returns = returns.dropna()
            
            if len(clean_returns) < 30:
                return 0.0
            
            # Estimate parameters
            mean_return = clean_returns.mean()
            std_return = clean_returns.std()
            
            # Monte Carlo simulation
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(
                mean_return * time_horizon,
                std_return * np.sqrt(time_horizon),
                num_simulations
            )
            
            # Calculate VaR
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
            return var
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo VaR: {e}")
            return 0.0
    
    def calculate_stress_test(
        self,
        weights: Dict[str, float],
        returns_data: Dict[str, pd.Series],
        stress_scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance under stress scenarios.
        
        Args:
            weights: Portfolio weights
            returns_data: Historical returns data
            stress_scenarios: Stress scenarios {scenario_name: {symbol: shock}}
        
        Returns:
            Portfolio returns under each scenario
        """
        try:
            scenario_results = {}
            
            for scenario_name, shocks in stress_scenarios.items():
                portfolio_shock = 0.0
                
                for symbol, weight in weights.items():
                    if symbol in shocks:
                        portfolio_shock += weight * shocks[symbol]
                
                scenario_results[scenario_name] = portfolio_shock
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Error calculating stress test: {e}")
            return {}
    
    def calculate_correlation_matrix(
        self,
        returns_data: Dict[str, pd.Series],
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio positions.
        
        Args:
            returns_data: Returns data for each position
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Correlation matrix
        """
        try:
            returns_df = pd.DataFrame(returns_data).dropna()
            
            if len(returns_df) < 20:
                logger.warning("Insufficient data for correlation calculation")
                return pd.DataFrame()
            
            correlation_matrix = returns_df.corr(method=method)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def calculate_rolling_risk(
        self,
        returns: pd.Series,
        window: int = 252,
        metrics: List[str] = ['volatility', 'var_95', 'sharpe']
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Args:
            returns: Return series
            window: Rolling window size
            metrics: List of metrics to calculate
        
        Returns:
            DataFrame with rolling metrics
        """
        try:
            rolling_metrics = pd.DataFrame(index=returns.index)
            
            if 'volatility' in metrics:
                rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
            
            if 'var_95' in metrics:
                rolling_metrics['var_95'] = -returns.rolling(window).quantile(0.05)
            
            if 'sharpe' in metrics:
                excess_returns = returns - self.risk_free_rate / 252
                rolling_mean = excess_returns.rolling(window).mean() * 252
                rolling_std = excess_returns.rolling(window).std() * np.sqrt(252)
                rolling_metrics['sharpe'] = rolling_mean / rolling_std
            
            if 'max_drawdown' in metrics:
                def rolling_max_drawdown(series):
                    cumulative = (1 + series).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    return abs(drawdown.min())
                
                rolling_metrics['max_drawdown'] = returns.rolling(window).apply(
                    rolling_max_drawdown, raw=False
                )
            
            return rolling_metrics.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating rolling risk: {e}")
            return pd.DataFrame()