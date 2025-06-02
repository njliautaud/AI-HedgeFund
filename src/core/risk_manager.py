import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass
from scipy import stats

from src.utils.risk_metrics import RiskMetrics
from src.utils.correlation_utils import CorrelationAnalyzer
from src.utils.transaction_costs import TransactionCost

@dataclass
class RiskParameters:
    """
    Container for risk management parameters.
    
    Attributes:
        max_position_size: Maximum position size as % of portfolio
        max_drawdown: Maximum allowed drawdown
        max_volatility: Maximum allowed portfolio volatility
        value_at_risk: Maximum allowed VaR
        correlation_threshold: Maximum allowed asset correlation
    """
    max_position_size: float = 0.05
    max_drawdown: float = 0.20
    max_volatility: float = 0.20
    value_at_risk: float = 0.05
    correlation_threshold: float = 0.80


class RiskManager:
    """
    Manages portfolio risk across multiple dimensions.
    
    Attributes:
        risk_params: Risk parameters
        correlation_analyzer: Correlation analysis tool
        lookback_period: Period for risk calculations
    """
    
    def __init__(self, config: Dict, data_manager, risk_params: RiskParameters = None, lookback_period: int = 252):
        self.config = config
        self.data_manager = data_manager
        self.risk_params = risk_params or RiskParameters()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.lookback_period = lookback_period
        self.portfolio_metrics = {}
        
    async def initialize(self):
        """Initialize the risk manager."""
        logger.info("Initializing Risk Manager...")
        # Initialize any required resources or connections
        pass
        
    async def start_monitoring(self, portfolio_manager, check_interval: int = 300):
        """
        Start continuous risk monitoring of the portfolio.
        
        Args:
            portfolio_manager: PortfolioManager instance to monitor
            check_interval: Time in seconds between risk checks
        """
        logger.info("Starting Risk Manager monitoring...")
        self.monitoring = True
        
        while self.monitoring:
            try:
                # Get current portfolio state
                portfolio = await portfolio_manager.get_portfolio()
                
                # Calculate portfolio risk metrics
                risk_metrics = self.calculate_portfolio_risk(portfolio['returns'])
                
                # Check for risk threshold breaches
                await self._check_risk_thresholds(risk_metrics, portfolio)
                
                # Store metrics for historical analysis
                self.portfolio_metrics[datetime.now()] = risk_metrics
                
                # Wait for next check
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                logger.info("Risk monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)  # Wait before retrying on error
    
    async def stop_monitoring(self):
        """Stop any ongoing monitoring tasks."""
        logger.info("Stopping Risk Manager monitoring...")
        self.monitoring = False
        
    async def _check_risk_thresholds(self, risk_metrics: Dict, portfolio: Dict):
        """
        Check risk metrics against thresholds and trigger alerts if needed.
        
        Args:
            risk_metrics: Dictionary of calculated risk metrics
            portfolio: Current portfolio state
        """
        alerts = []
        
        # Check position sizes
        for symbol, position in portfolio['positions'].items():
            position_size = position['value'] / portfolio['total_value']
            if position_size > self.risk_params.max_position_size:
                alerts.append(f"Position size for {symbol} ({position_size:.2%}) exceeds maximum allowed ({self.risk_params.max_position_size:.2%})")
        
        # Check drawdown
        if risk_metrics['max_drawdown'] < -self.risk_params.max_drawdown:
            alerts.append(f"Maximum drawdown ({abs(risk_metrics['max_drawdown']):.2%}) exceeds threshold ({self.risk_params.max_drawdown:.2%})")
        
        # Check volatility
        if risk_metrics['volatility'] > self.risk_params.max_volatility:
            alerts.append(f"Portfolio volatility ({risk_metrics['volatility']:.2%}) exceeds maximum allowed ({self.risk_params.max_volatility:.2%})")
        
        # Check VaR
        if risk_metrics['var_95'] < -self.risk_params.value_at_risk:
            alerts.append(f"95% Value at Risk ({abs(risk_metrics['var_95']):.2%}) exceeds threshold ({self.risk_params.value_at_risk:.2%})")
        
        # Log alerts if any
        for alert in alerts:
            logger.warning(f"RISK ALERT: {alert}")
            
        # Here you could also implement additional actions like:
        # - Sending email/SMS alerts
        # - Triggering automated risk reduction
        # - Notifying portfolio manager
        
        return alerts
        
    def calculate_portfolio_risk(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio risk metrics.
        
        Args:
            portfolio_returns: Series of portfolio returns
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if len(portfolio_returns) < self.lookback_period:
                logger.warning("Insufficient data for risk calculation")
                return self._default_risk_metrics()
                
            # Calculate basic statistics
            mean_return = portfolio_returns.mean()
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Calculate drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.cummax()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Calculate VaR
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)
            
            # Calculate Sharpe ratio
            risk_free_rate = 0.02  # 2% annual risk-free rate
            sharpe_ratio = (mean_return * 252 - risk_free_rate) / volatility
            
            # Create risk metrics object
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99
            )
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self._default_risk_metrics()
            
    def _default_risk_metrics(self) -> Dict[str, Any]:
        """Return default risk metrics when calculation fails."""
        return {
            'volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'risk_metrics': RiskMetrics(),
            'timestamp': datetime.now()
        }
        
    def analyze_position_risk(self, position_size: float, portfolio_value: float) -> Dict[str, Any]:
        """
        Analyze risk of a single position.
        
        Args:
            position_size: Size of the position
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary of position risk analysis
        """
        try:
            position_pct = position_size / portfolio_value
            
            # Check against risk parameters
            is_concentrated = position_pct > self.risk_params.max_position_size
            
            # Calculate potential impact
            impact = {
                'position_size': position_size,
                'portfolio_pct': position_pct,
                'is_concentrated': is_concentrated,
                'max_allowed_pct': self.risk_params.max_position_size
            }
            
            return impact
            
        except Exception as e:
            logger.error(f"Error analyzing position risk: {e}")
            return {
                'position_size': 0.0,
                'portfolio_pct': 0.0,
                'is_concentrated': False,
                'max_allowed_pct': self.risk_params.max_position_size
            }
            
    def assess_diversification(self, portfolio_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Assess portfolio diversification.
        
        Args:
            portfolio_weights: Dictionary of asset weights
            
        Returns:
            Dictionary of diversification analysis
        """
        try:
            if not portfolio_weights:
                return {
                    'diversification_score': 0.0,
                    'reasoning': 'No portfolio weights provided'
                }
                
            # Calculate correlation matrix
            correlation_matrix = self.correlation_analyzer.calculate_correlation_matrix(
                {asset: pd.Series([weight]) for asset, weight in portfolio_weights.items()}
            )
            
            # Calculate diversification score
            diversification_score = self.correlation_analyzer.score_diversification(correlation_matrix)
            
            return {
                'diversification_score': diversification_score,
                'correlation_matrix': correlation_matrix,
                'is_overconcentrated': diversification_score < 50,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error assessing diversification: {e}")
            return {
                'diversification_score': 0.0,
                'correlation_matrix': {},
                'is_overconcentrated': True,
                'timestamp': datetime.now()
            }
            
    def get_risk_limits(self) -> Dict[str, float]:
        """
        Get current risk limits.
        
        Returns:
            Dictionary of risk limits
        """
        return {
            'max_position_size': self.risk_params.max_position_size,
            'max_drawdown': self.risk_params.max_drawdown,
            'max_volatility': self.risk_params.max_volatility,
            'value_at_risk': self.risk_params.value_at_risk,
            'correlation_threshold': self.risk_params.correlation_threshold
        }
        
    def check_risk_violations(self, portfolio_returns: pd.Series) -> Dict[str, Any]:
        """
        Check for risk violations.
        
        Args:
            portfolio_returns: Series of portfolio returns
            
        Returns:
            Dictionary of risk violations
        """
        try:
            risk_metrics = self.calculate_portfolio_risk(portfolio_returns)
            
            violations = {
                'volatility': risk_metrics['volatility'] > self.risk_params.max_volatility,
                'drawdown': abs(risk_metrics['max_drawdown']) > self.risk_params.max_drawdown,
                'VaR_95': abs(risk_metrics['risk_metrics'].var_95) > self.risk_params.value_at_risk,
                'VaR_99': abs(risk_metrics['risk_metrics'].var_99) > self.risk_params.value_at_risk
            }
            
            return {
                'violations': violations,
                'risk_metrics': risk_metrics,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error checking risk violations: {e}")
            return {
                'violations': {},
                'risk_metrics': {},
                'timestamp': datetime.now()
            }
