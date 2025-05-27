#!/usr/bin/env python3
"""
Portfolio Optimizer - Modern Portfolio Theory Implementation

Implements various portfolio optimization techniques:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Black-Litterman
- Minimum Variance
- Maximum Sharpe Ratio
- Custom constraint handling
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Results from portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    optimization_method: str
    success: bool
    message: str
    iterations: int = 0


class PortfolioOptimizer:
    """
    Advanced portfolio optimization using multiple methodologies.
    
    Supports:
    - Mean-variance optimization
    - Risk parity
    - Minimum variance
    - Maximum Sharpe ratio
    - Custom constraints
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_method = config.get('optimization', {}).get('method', 'mean_variance')
        self.risk_free_rate = config.get('optimization', {}).get('risk_free_rate', 0.02)
        self.max_iterations = config.get('optimization', {}).get('max_iterations', 1000)
        
        # Optimization parameters
        self.regularization = 1e-5
        self.tolerance = 1e-8
        
        logger.info(f"PortfolioOptimizer initialized with method: {self.optimization_method}")
    
    def optimize(
        self,
        expected_returns: Dict[str, float],
        risk_matrix: np.ndarray,
        constraints: Dict[str, Any],
        current_weights: Optional[Dict[str, float]] = None,
        method: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio allocation.
        
        Args:
            expected_returns: Expected returns for each asset
            risk_matrix: Covariance matrix
            constraints: Portfolio constraints
            current_weights: Current portfolio weights
            method: Optimization method override
        
        Returns:
            Optimized weights dictionary
        """
        try:
            # Use specified method or default
            opt_method = method or self.optimization_method
            
            # Prepare data
            symbols = list(expected_returns.keys())
            returns_vector = np.array([expected_returns[symbol] for symbol in symbols])
            
            # Ensure risk matrix matches symbols
            if risk_matrix.shape[0] != len(symbols):
                logger.warning(f"Risk matrix size {risk_matrix.shape} doesn't match symbols {len(symbols)}")
                risk_matrix = self._create_default_risk_matrix(len(symbols))
            
            # Add regularization to ensure positive definite
            risk_matrix = risk_matrix + np.eye(len(symbols)) * self.regularization
            
            # Run optimization based on method
            if opt_method == 'mean_variance':
                result = self._mean_variance_optimization(
                    symbols, returns_vector, risk_matrix, constraints, current_weights
                )
            elif opt_method == 'risk_parity':
                result = self._risk_parity_optimization(
                    symbols, risk_matrix, constraints
                )
            elif opt_method == 'min_variance':
                result = self._minimum_variance_optimization(
                    symbols, risk_matrix, constraints
                )
            elif opt_method == 'max_sharpe':
                result = self._maximum_sharpe_optimization(
                    symbols, returns_vector, risk_matrix, constraints
                )
            elif opt_method == 'black_litterman':
                result = self._black_litterman_optimization(
                    symbols, returns_vector, risk_matrix, constraints, current_weights
                )
            else:
                logger.warning(f"Unknown optimization method: {opt_method}, using mean_variance")
                result = self._mean_variance_optimization(
                    symbols, returns_vector, risk_matrix, constraints, current_weights
                )
            
            if result.success:
                logger.info(f"Optimization successful: {result.optimization_method}, "
                          f"Expected Return: {result.expected_return:.2%}, "
                          f"Risk: {result.expected_risk:.2%}, "
                          f"Sharpe: {result.sharpe_ratio:.3f}")
                return result.weights
            else:
                logger.error(f"Optimization failed: {result.message}")
                return self._fallback_allocation(symbols, constraints)
                
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return self._fallback_allocation(list(expected_returns.keys()), constraints)
    
    def _mean_variance_optimization(
        self,
        symbols: List[str],
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any],
        current_weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Mean-variance optimization (Markowitz).
        
        Maximizes: expected_return - (risk_aversion * variance)
        """
        n_assets = len(symbols)
        risk_aversion = self.config.get('optimization', {}).get('risk_aversion', 1.0)
        
        # Objective function: maximize utility (return - risk penalty)
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints and bounds
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        x0 = self._get_initial_weights(n_assets, current_weights, symbols)
        
        # Optimize
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.x)}
                
                # Calculate metrics
                portfolio_return = np.dot(result.x, returns)
                portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_method='mean_variance',
                    success=True,
                    message='Optimization converged',
                    iterations=result.nit
                )
            else:
                return OptimizationResult(
                    weights={},
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    optimization_method='mean_variance',
                    success=False,
                    message=f"Optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='mean_variance',
                success=False,
                message=f"Optimization error: {str(e)}"
            )
    
    def _risk_parity_optimization(
        self,
        symbols: List[str],
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Risk parity optimization - equal risk contribution.
        """
        n_assets = len(symbols)
        
        def risk_budget_objective(weights):
            """Minimize sum of squared differences in risk contributions."""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution
            target_risk = portfolio_vol / n_assets
            
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints and bounds
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Initial guess - equal weights
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                risk_budget_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.x)}
                
                # Calculate metrics
                portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=0,  # Not optimizing for return
                    expected_risk=portfolio_risk,
                    sharpe_ratio=0,
                    optimization_method='risk_parity',
                    success=True,
                    message='Risk parity optimization converged',
                    iterations=result.nit
                )
            else:
                return OptimizationResult(
                    weights={},
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    optimization_method='risk_parity',
                    success=False,
                    message=f"Risk parity optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='risk_parity',
                success=False,
                message=f"Risk parity error: {str(e)}"
            )
    
    def _minimum_variance_optimization(
        self,
        symbols: List[str],
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Minimum variance optimization.
        """
        n_assets = len(symbols)
        
        # Objective: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints and bounds
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.x)}
                
                # Calculate metrics
                portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=0,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=0,
                    optimization_method='min_variance',
                    success=True,
                    message='Minimum variance optimization converged',
                    iterations=result.nit
                )
            else:
                return OptimizationResult(
                    weights={},
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    optimization_method='min_variance',
                    success=False,
                    message=f"Minimum variance optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='min_variance',
                success=False,
                message=f"Minimum variance error: {str(e)}"
            )
    
    def _maximum_sharpe_optimization(
        self,
        symbols: List[str],
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any]
    ) -> OptimizationResult:
        """
        Maximum Sharpe ratio optimization.
        """
        n_assets = len(symbols)
        
        # Objective: maximize Sharpe ratio (minimize negative Sharpe)
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            
            if portfolio_std == 0:
                return 1e6  # Large penalty for zero variance
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints and bounds
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.x)}
                
                # Calculate metrics
                portfolio_return = np.dot(result.x, returns)
                portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_method='max_sharpe',
                    success=True,
                    message='Maximum Sharpe optimization converged',
                    iterations=result.nit
                )
            else:
                return OptimizationResult(
                    weights={},
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    optimization_method='max_sharpe',
                    success=False,
                    message=f"Maximum Sharpe optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='max_sharpe',
                success=False,
                message=f"Maximum Sharpe error: {str(e)}"
            )
    
    def _black_litterman_optimization(
        self,
        symbols: List[str],
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any],
        current_weights: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """
        Black-Litterman optimization with market equilibrium.
        """
        n_assets = len(symbols)
        
        # Market capitalization weights (proxy using equal weights)
        if current_weights:
            market_weights = np.array([current_weights.get(symbol, 1/n_assets) for symbol in symbols])
        else:
            market_weights = np.ones(n_assets) / n_assets
        
        # Risk aversion parameter
        risk_aversion = self.config.get('optimization', {}).get('risk_aversion', 3.0)
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        # Confidence in views (tau parameter)
        tau = 0.025
        
        # Black-Litterman expected returns (no views for simplicity)
        # In practice, you would incorporate analyst views here
        bl_returns = pi  # Using equilibrium returns
        
        # Mean-variance optimization with BL returns
        def objective(weights):
            portfolio_return = np.dot(weights, bl_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_variance)
        
        # Constraints and bounds
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Initial guess
        x0 = market_weights
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                weights_dict = {symbol: weight for symbol, weight in zip(symbols, result.x)}
                
                # Calculate metrics
                portfolio_return = np.dot(result.x, bl_returns)
                portfolio_risk = np.sqrt(np.dot(result.x, np.dot(cov_matrix, result.x)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                
                return OptimizationResult(
                    weights=weights_dict,
                    expected_return=portfolio_return,
                    expected_risk=portfolio_risk,
                    sharpe_ratio=sharpe_ratio,
                    optimization_method='black_litterman',
                    success=True,
                    message='Black-Litterman optimization converged',
                    iterations=result.nit
                )
            else:
                return OptimizationResult(
                    weights={},
                    expected_return=0,
                    expected_risk=0,
                    sharpe_ratio=0,
                    optimization_method='black_litterman',
                    success=False,
                    message=f"Black-Litterman optimization failed: {result.message}"
                )
                
        except Exception as e:
            return OptimizationResult(
                weights={},
                expected_return=0,
                expected_risk=0,
                sharpe_ratio=0,
                optimization_method='black_litterman',
                success=False,
                message=f"Black-Litterman error: {str(e)}"
            )
    
    def _build_constraints(self, n_assets: int, constraints: Dict[str, Any]) -> Tuple[Bounds, List]:
        """
        Build optimization constraints and bounds.
        """
        # Individual asset bounds
        lower_bounds = np.zeros(n_assets)
        upper_bounds = np.ones(n_assets)
        
        # Apply position size constraints
        max_position_size = constraints.get('max_position_size', 1.0)
        upper_bounds = np.minimum(upper_bounds, max_position_size)
        
        bounds = Bounds(lower_bounds, upper_bounds)
        
        # Linear constraints
        linear_constraints = []
        
        # Budget constraint: sum of weights = 1 - cash_reserve
        cash_reserve = constraints.get('min_cash_reserve', 0.0)
        target_investment = 1.0 - cash_reserve
        
        # Equality constraint: sum(weights) = target_investment
        A_eq = np.ones((1, n_assets))
        b_eq = np.array([target_investment])
        
        linear_constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - target_investment
        })
        
        return bounds, linear_constraints
    
    def _get_initial_weights(
        self,
        n_assets: int,
        current_weights: Optional[Dict[str, float]],
        symbols: List[str]
    ) -> np.ndarray:
        """
        Get initial weights for optimization.
        """
        if current_weights:
            # Use current weights as starting point
            weights = np.array([current_weights.get(symbol, 0) for symbol in symbols])
            
            # Normalize if needed
            total_weight = np.sum(weights)
            if total_weight > 0:
                weights = weights / total_weight * 0.95  # Leave some cash
            else:
                weights = np.ones(n_assets) / n_assets * 0.95
        else:
            # Equal weights
            weights = np.ones(n_assets) / n_assets * 0.95
        
        return weights
    
    def _create_default_risk_matrix(self, n_assets: int) -> np.ndarray:
        """
        Create default risk matrix when data is insufficient.
        """
        # Create correlation matrix with moderate correlation
        correlation = np.full((n_assets, n_assets), 0.3)
        np.fill_diagonal(correlation, 1.0)
        
        # Assume 20% volatility for all assets
        volatilities = np.full(n_assets, 0.20)
        
        # Convert to covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation
        
        return cov_matrix
    
    def _fallback_allocation(self, symbols: List[str], constraints: Dict[str, Any]) -> Dict[str, float]:
        """
        Fallback allocation when optimization fails.
        """
        logger.warning("Using fallback equal-weight allocation")
        
        n_assets = len(symbols)
        cash_reserve = constraints.get('min_cash_reserve', 0.05)
        max_position_size = constraints.get('max_position_size', 0.20)
        
        # Equal weights with constraints
        target_investment = 1.0 - cash_reserve
        equal_weight = target_investment / n_assets
        
        # Apply position size limit
        weight_per_asset = min(equal_weight, max_position_size)
        
        return {symbol: weight_per_asset for symbol in symbols}
    
    def calculate_efficient_frontier(
        self,
        expected_returns: Dict[str, float],
        risk_matrix: np.ndarray,
        constraints: Dict[str, Any],
        n_points: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Calculate efficient frontier points.
        """
        symbols = list(expected_returns.keys())
        returns_vector = np.array([expected_returns[symbol] for symbol in symbols])
        
        # Find minimum and maximum return portfolios
        min_return = np.min(returns_vector)
        max_return = np.max(returns_vector)
        
        target_returns = np.linspace(min_return, max_return, n_points)
        frontier_points = []
        
        for target_return in target_returns:
            try:
                # Optimize for minimum variance given target return
                weights = self._optimize_for_target_return(
                    symbols, returns_vector, risk_matrix, constraints, target_return
                )
                
                if weights:
                    portfolio_return = np.dot(list(weights.values()), returns_vector)
                    portfolio_risk = np.sqrt(np.dot(list(weights.values()), 
                                                  np.dot(risk_matrix, list(weights.values()))))
                    sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
                    
                    frontier_points.append({
                        'return': portfolio_return,
                        'risk': portfolio_risk,
                        'sharpe_ratio': sharpe_ratio,
                        'weights': weights
                    })
                    
            except Exception as e:
                logger.debug(f"Error calculating frontier point for return {target_return}: {e}")
                continue
        
        return frontier_points
    
    def _optimize_for_target_return(
        self,
        symbols: List[str],
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: Dict[str, Any],
        target_return: float
    ) -> Optional[Dict[str, float]]:
        """
        Optimize for minimum variance given target return.
        """
        n_assets = len(symbols)
        
        # Objective: minimize variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        bounds, linear_constraints = self._build_constraints(n_assets, constraints)
        
        # Add return constraint
        linear_constraints.append({
            'type': 'eq',
            'fun': lambda x: np.dot(x, returns) - target_return
        })
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=linear_constraints,
                options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
            )
            
            if result.success:
                return {symbol: weight for symbol, weight in zip(symbols, result.x)}
            else:
                return None
                
        except Exception:
            return None