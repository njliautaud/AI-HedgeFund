#!/usr/bin/env python3
"""
Portfolio Manager - Portfolio Construction and Management

Handles:
- Portfolio construction and optimization
- Position sizing and allocation
- Risk management and monitoring
- Rebalancing and execution
- Performance tracking
- Transaction cost analysis
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass, field
from scipy.optimize import minimize
import json

from src.core.strategy_manager import StrategySignal
from src.utils.portfolio_optimizer import PortfolioOptimizer
from src.utils.risk_metrics import RiskCalculator
from src.utils.transaction_costs import TransactionCostModel


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def return_pct(self) -> float:
        if self.avg_cost > 0:
            return (self.current_price - self.avg_cost) / self.avg_cost
        return 0.0


@dataclass
class Portfolio:
    """Portfolio state and metrics."""
    positions: Dict[str, Position] = field(default_factory=dict)
    cash: float = 100000.0
    total_value: float = 100000.0
    total_pnl: float = 0.0
    daily_returns: List[float] = field(default_factory=list)
    benchmark_returns: List[float] = field(default_factory=list)
    last_rebalance: datetime = field(default_factory=datetime.now)
    inception_date: datetime = field(default_factory=datetime.now)
    
    @property
    def invested_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def cash_weight(self) -> float:
        return self.cash / self.total_value if self.total_value > 0 else 0
    
    @property
    def num_positions(self) -> int:
        return len([pos for pos in self.positions.values() if pos.quantity != 0])


@dataclass
class Trade:
    """Represents a trade to be executed."""
    symbol: str
    action: str  # 'BUY', 'SELL'
    quantity: float
    target_weight: float
    current_weight: float
    priority: float
    reasoning: str
    strategy_source: str
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class PortfolioManager:
    """
    Manages portfolio construction, optimization, and execution.
    
    Key responsibilities:
    1. Aggregate signals from multiple strategies
    2. Optimize portfolio allocation
    3. Manage risk constraints
    4. Execute trades and rebalancing
    5. Track performance and attribution
    """
    
    def __init__(self, config: Dict, data_manager, risk_manager):
        self.config = config
        self.data_manager = data_manager
        self.risk_manager = risk_manager
        
        # Initialize portfolio
        initial_capital = config.get('portfolio', {}).get('initial_capital', 100000)
        self.portfolio = Portfolio(cash=initial_capital, total_value=initial_capital)
        
        # Portfolio constraints
        self.constraints = {
            'max_position_size': config.get('risk_management', {}).get('max_position_size', 0.05),
            'max_sector_exposure': config.get('risk_management', {}).get('max_sector_exposure', 0.25),
            'min_cash_reserve': config.get('portfolio', {}).get('min_cash_reserve', 0.05),
            'max_turnover': 0.50,  # 50% max monthly turnover
            'max_positions': 50
        }
        
        # Initialize optimizers and calculators
        self.optimizer = PortfolioOptimizer(config)
        self.risk_calculator = RiskCalculator()
        self.transaction_cost_model = TransactionCostModel(config)
        
        # Performance tracking
        self.performance_history = []
        self.trade_history = []
        
        # Rebalancing settings
        self.rebalance_frequency = config.get('portfolio', {}).get('rebalance_frequency', 'weekly')
        self.rebalance_threshold = 0.05  # 5% weight deviation triggers rebalance
        
        logger.info(f"PortfolioManager initialized with ${initial_capital:,.0f} capital")
    
    async def update_portfolio(self, signals: List[StrategySignal]):
        """Update portfolio based on new strategy signals."""
        logger.info(f"Processing {len(signals)} strategy signals")
        
        try:
            # 1. Update current positions with latest prices
            await self._update_position_prices()
            
            # 2. Aggregate and optimize signals
            target_weights = await self._optimize_portfolio(signals)
            
            # 3. Generate trades to reach target allocation
            trades = await self._generate_trades(target_weights)
            
            # 4. Apply risk management filters
            filtered_trades = self.risk_manager.filter_trades(trades, self.portfolio)
            
            # 5. Execute trades
            executed_trades = await self._execute_trades(filtered_trades)
            
            # 6. Update portfolio state
            await self._update_portfolio_state(executed_trades)
            
            # 7. Record performance
            await self._record_performance()
            
            logger.info(f"Portfolio update completed. Executed {len(executed_trades)} trades")
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            raise
    
    async def _update_position_prices(self):
        """Update current prices for all positions."""
        for symbol, position in self.portfolio.positions.items():
            try:
                # Get current price from data manager
                symbol_data = self.data_manager.market_data.get(symbol, {})
                price_data = symbol_data.get('price_data')
                
                if price_data is not None and not price_data.empty:
                    current_price = price_data['Close'].iloc[-1]
                    
                    # Update position
                    position.current_price = current_price
                    position.market_value = position.quantity * current_price
                    position.unrealized_pnl = (current_price - position.avg_cost) * position.quantity
                    position.last_updated = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")
        
        # Update portfolio totals
        self.portfolio.total_value = self.portfolio.cash + self.portfolio.invested_value
        
        # Update position weights
        for position in self.portfolio.positions.values():
            position.weight = position.market_value / self.portfolio.total_value if self.portfolio.total_value > 0 else 0
    
    async def _optimize_portfolio(self, signals: List[StrategySignal]) -> Dict[str, float]:
        """Optimize portfolio allocation based on strategy signals."""
        try:
            # 1. Aggregate signals by symbol
            aggregated_signals = self._aggregate_signals(signals)
            
            # 2. Get expected returns and risk estimates
            expected_returns = await self._estimate_expected_returns(aggregated_signals)
            risk_estimates = await self._estimate_risk_matrix(list(aggregated_signals.keys()))
            
            # 3. Run portfolio optimization
            target_weights = self.optimizer.optimize(
                expected_returns=expected_returns,
                risk_matrix=risk_estimates,
                constraints=self.constraints,
                current_weights=self._get_current_weights()
            )
            
            return target_weights
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            # Fallback: use signal weights directly with constraints
            return self._apply_simple_allocation(signals)
    
    def _aggregate_signals(self, signals: List[StrategySignal]) -> Dict[str, Dict[str, Any]]:
        """Aggregate multiple signals for the same symbol."""
        aggregated = {}
        
        for signal in signals:
            symbol = signal.symbol
            
            if symbol not in aggregated:
                aggregated[symbol] = {
                    'total_weight': 0.0,
                    'weighted_confidence': 0.0,
                    'signals': [],
                    'actions': []
                }
            
            # Weight by confidence
            weight_contribution = signal.target_weight * signal.confidence
            aggregated[symbol]['total_weight'] += weight_contribution
            aggregated[symbol]['weighted_confidence'] += signal.confidence
            aggregated[symbol]['signals'].append(signal)
            aggregated[symbol]['actions'].append(signal.action)
        
        # Normalize confidence scores
        for symbol_data in aggregated.values():
            num_signals = len(symbol_data['signals'])
            if num_signals > 0:
                symbol_data['weighted_confidence'] /= num_signals
        
        return aggregated
    
    async def _estimate_expected_returns(self, aggregated_signals: Dict[str, Dict]) -> Dict[str, float]:
        """Estimate expected returns for symbols."""
        expected_returns = {}
        
        for symbol, signal_data in aggregated_signals.items():
            try:
                # Base expected return on signal strength and historical performance
                confidence = signal_data['weighted_confidence']
                
                # Get historical returns
                symbol_data = self.data_manager.market_data.get(symbol, {})
                price_data = symbol_data.get('price_data')
                
                if price_data is not None and not price_data.empty:
                    # Calculate historical return (annualized)
                    returns = price_data['Close'].pct_change().dropna()
                    historical_return = returns.mean() * 252  # Annualized
                    
                    # Adjust based on signal confidence
                    expected_return = historical_return * (0.5 + confidence * 0.5)
                    expected_returns[symbol] = expected_return
                else:
                    expected_returns[symbol] = 0.05  # Default 5% expected return
                    
            except Exception as e:
                logger.error(f"Error estimating returns for {symbol}: {e}")
                expected_returns[symbol] = 0.05
        
        return expected_returns
    
    async def _estimate_risk_matrix(self, symbols: List[str]) -> np.ndarray:
        """Estimate covariance matrix for symbols."""
        try:
            returns_data = []
            valid_symbols = []
            
            for symbol in symbols:
                symbol_data = self.data_manager.market_data.get(symbol, {})
                price_data = symbol_data.get('price_data')
                
                if price_data is not None and not price_data.empty:
                    returns = price_data['Close'].pct_change().dropna()
                    if len(returns) > 50:  # Minimum data requirement
                        returns_data.append(returns)
                        valid_symbols.append(symbol)
            
            if len(returns_data) < 2:
                # Return identity matrix if insufficient data
                return np.eye(len(symbols)) * 0.04  # 20% volatility assumption
            
            # Align returns data
            returns_df = pd.concat(returns_data, axis=1, keys=valid_symbols)
            returns_df = returns_df.dropna()
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov() * 252
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return cov_matrix.values
            
        except Exception as e:
            logger.error(f"Error estimating risk matrix: {e}")
            return np.eye(len(symbols)) * 0.04
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights."""
        weights = {}
        for symbol, position in self.portfolio.positions.items():
            weights[symbol] = position.weight
        return weights
    
    def _apply_simple_allocation(self, signals: List[StrategySignal]) -> Dict[str, float]:
        """Apply simple allocation when optimization fails."""
        target_weights = {}
        total_weight = 0
        
        # Sum all target weights
        for signal in signals:
            if signal.action == 'BUY':
                target_weights[signal.symbol] = signal.target_weight
                total_weight += signal.target_weight
        
        # Normalize to respect cash reserve
        max_invested = 1 - self.constraints['min_cash_reserve']
        if total_weight > max_invested:
            scale_factor = max_invested / total_weight
            for symbol in target_weights:
                target_weights[symbol] *= scale_factor
        
        # Apply position size constraints
        max_position = self.constraints['max_position_size']
        for symbol in target_weights:
            target_weights[symbol] = min(target_weights[symbol], max_position)
        
        return target_weights
    
    async def _generate_trades(self, target_weights: Dict[str, float]) -> List[Trade]:
        """Generate trades to reach target allocation."""
        trades = []
        current_weights = self._get_current_weights()
        
        # Calculate required trades
        all_symbols = set(target_weights.keys()) | set(current_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Only trade if difference is significant
            if abs(weight_diff) > self.rebalance_threshold:
                # Calculate quantity to trade
                target_value = target_weight * self.portfolio.total_value
                current_value = current_weight * self.portfolio.total_value
                trade_value = target_value - current_value
                
                # Get current price
                current_price = self._get_current_price(symbol)
                if current_price > 0:
                    quantity = trade_value / current_price
                    action = 'BUY' if quantity > 0 else 'SELL'
                    
                    # Estimate transaction costs
                    estimated_cost = self.transaction_cost_model.estimate_cost(
                        symbol, abs(quantity), current_price
                    )
                    
                    trade = Trade(
                        symbol=symbol,
                        action=action,
                        quantity=abs(quantity),
                        target_weight=target_weight,
                        current_weight=current_weight,
                        priority=abs(weight_diff),
                        reasoning=f"Rebalance: {current_weight:.2%} -> {target_weight:.2%}",
                        strategy_source="portfolio_optimization",
                        estimated_cost=estimated_cost
                    )
                    
                    trades.append(trade)
        
        # Sort by priority (largest weight differences first)
        trades.sort(key=lambda x: x.priority, reverse=True)
        
        return trades
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            symbol_data = self.data_manager.market_data.get(symbol, {})
            price_data = symbol_data.get('price_data')
            
            if price_data is not None and not price_data.empty:
                return price_data['Close'].iloc[-1]
            
            return 0.0
            
        except Exception:
            return 0.0
    
    async def _execute_trades(self, trades: List[Trade]) -> List[Trade]:
        """Execute trades (simulated execution)."""
        executed_trades = []
        
        for trade in trades:
            try:
                # Check if we have enough cash for buys
                if trade.action == 'BUY':
                    required_cash = trade.quantity * self._get_current_price(trade.symbol)
                    if required_cash > self.portfolio.cash:
                        logger.warning(f"Insufficient cash for {trade.symbol} trade")
                        continue
                
                # Execute the trade
                success = await self._execute_single_trade(trade)
                
                if success:
                    executed_trades.append(trade)
                    self.trade_history.append(trade)
                    
            except Exception as e:
                logger.error(f"Error executing trade for {trade.symbol}: {e}")
        
        return executed_trades
    
    async def _execute_single_trade(self, trade: Trade) -> bool:
        """Execute a single trade."""
        try:
            symbol = trade.symbol
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                return False
            
            # Update or create position
            if symbol not in self.portfolio.positions:
                self.portfolio.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=0,
                    avg_cost=0,
                    current_price=current_price,
                    market_value=0,
                    weight=0,
                    unrealized_pnl=0
                )
            
            position = self.portfolio.positions[symbol]
            
            if trade.action == 'BUY':
                # Calculate new average cost
                total_cost = position.quantity * position.avg_cost + trade.quantity * current_price
                new_quantity = position.quantity + trade.quantity
                
                position.avg_cost = total_cost / new_quantity if new_quantity > 0 else current_price
                position.quantity = new_quantity
                
                # Update cash
                trade_cost = trade.quantity * current_price + trade.estimated_cost
                self.portfolio.cash -= trade_cost
                
            else:  # SELL
                # Realize P&L
                realized_pnl = (current_price - position.avg_cost) * trade.quantity
                position.realized_pnl += realized_pnl
                position.quantity -= trade.quantity
                
                # Update cash
                trade_proceeds = trade.quantity * current_price - trade.estimated_cost
                self.portfolio.cash += trade_proceeds
                
                # Remove position if fully sold
                if position.quantity <= 0:
                    del self.portfolio.positions[symbol]
            
            logger.debug(f"Executed {trade.action} {trade.quantity:.2f} shares of {symbol} at ${current_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def _update_portfolio_state(self, executed_trades: List[Trade]):
        """Update portfolio state after trade execution."""
        # Update position values
        await self._update_position_prices()
        
        # Calculate total P&L
        total_realized_pnl = sum(pos.realized_pnl for pos in self.portfolio.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.portfolio.positions.values())
        self.portfolio.total_pnl = total_realized_pnl + total_unrealized_pnl
    
    async def _record_performance(self):
        """Record daily performance metrics."""
        try:
            # Calculate daily return
            if len(self.performance_history) > 0:
                prev_value = self.performance_history[-1]['total_value']
                daily_return = (self.portfolio.total_value - prev_value) / prev_value
            else:
                daily_return = 0.0
            
            self.portfolio.daily_returns.append(daily_return)
            
            # Record performance snapshot
            performance_record = {
                'date': datetime.now(),
                'total_value': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'invested_value': self.portfolio.invested_value,
                'total_pnl': self.portfolio.total_pnl,
                'daily_return': daily_return,
                'num_positions': self.portfolio.num_positions,
                'cash_weight': self.portfolio.cash_weight
            }
            
            self.performance_history.append(performance_record)
            
            # Keep only last 252 days (1 year)
            if len(self.performance_history) > 252:
                self.performance_history = self.performance_history[-252:]
                self.portfolio.daily_returns = self.portfolio.daily_returns[-252:]
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def get_portfolio(self) -> Portfolio:
        """Get the current portfolio state."""
        return self.portfolio

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        try:
            current_value = self.portfolio.total_value
            cash = self.portfolio.cash
            invested = current_value - cash
            
            return {
                'total_value': current_value,
                'cash': cash,
                'invested': invested,
                'positions': len(self.portfolio.positions),
                'total_pnl': self.portfolio.total_pnl,
                'return_pct': (current_value - self.portfolio.inception_date) / self.portfolio.inception_date,
                'max_drawdown': self._calculate_max_drawdown(),
                'top_positions': self._get_top_positions(),
                'sector_allocation': self._get_sector_allocation()
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            raise

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if len(self.performance_history) < 2:
            return 0.0
            
        values = [record['total_value'] for record in self.performance_history]
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

    def _get_top_positions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N positions by weight."""
        positions = [
            {
                'symbol': pos.symbol,
                'weight': pos.weight,
                'market_value': pos.market_value,
                'pnl': pos.total_pnl,
                'return_pct': pos.return_pct
            }
            for pos in self.portfolio.positions.values()
        ]
        
        positions.sort(key=lambda x: x['weight'], reverse=True)
        return positions[:n]
    
    def _get_sector_allocation(self) -> Dict[str, float]:
        """Get allocation by sector (simplified)."""
        # This would map symbols to sectors
        # For now, return a placeholder
        return {'Technology': 0.3, 'Healthcare': 0.2, 'Financials': 0.15, 'Other': 0.35}
    
    async def save_state(self):
        """Save portfolio state to disk."""
        try:
            state = {
                'portfolio': {
                    'positions': {symbol: {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_cost': pos.avg_cost,
                        'realized_pnl': pos.realized_pnl,
                        'entry_date': pos.entry_date.isoformat()
                    } for symbol, pos in self.portfolio.positions.items()},
                    'cash': self.portfolio.cash,
                    'inception_date': self.portfolio.inception_date.isoformat()
                },
                'performance_history': self.performance_history,
                'trade_history': [{
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'quantity': trade.quantity,
                    'timestamp': trade.timestamp.isoformat()
                } for trade in self.trade_history[-100:]]  # Last 100 trades
            }
            
            with open('data/portfolio_state.json', 'w') as f:
                json.dump(state, f, default=str, indent=2)
            
            logger.info("Portfolio state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    async def load_state(self):
        """Load portfolio state from disk."""
        try:
            with open('data/portfolio_state.json', 'r') as f:
                state = json.load(f)
            
            # Restore portfolio positions
            for symbol, pos_data in state['portfolio']['positions'].items():
                position = Position(
                    symbol=pos_data['symbol'],
                    quantity=pos_data['quantity'],
                    avg_cost=pos_data['avg_cost'],
                    current_price=0,  # Will be updated
                    market_value=0,   # Will be calculated
                    weight=0,         # Will be calculated
                    unrealized_pnl=0, # Will be calculated
                    realized_pnl=pos_data['realized_pnl'],
                    entry_date=datetime.fromisoformat(pos_data['entry_date'])
                )
                self.portfolio.positions[symbol] = position
            
            self.portfolio.cash = state['portfolio']['cash']
            self.portfolio.inception_date = datetime.fromisoformat(state['portfolio']['inception_date'])
            
            # Update current prices and values
            await self._update_position_prices()
            
            logger.info("Portfolio state loaded successfully")
            
        except FileNotFoundError:
            logger.info("No saved portfolio state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")
    
    def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        # Check time-based rebalancing
        time_since_rebalance = datetime.now() - self.portfolio.last_rebalance
        
        if self.rebalance_frequency == 'daily' and time_since_rebalance.days >= 1:
            return True
        elif self.rebalance_frequency == 'weekly' and time_since_rebalance.days >= 7:
            return True
        elif self.rebalance_frequency == 'monthly' and time_since_rebalance.days >= 30:
            return True
        
        # Check drift-based rebalancing
        # (Would compare current weights to target weights)
        
        return False