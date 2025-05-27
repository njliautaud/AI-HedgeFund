#!/usr/bin/env python3
"""
Strategy Manager - Investment Strategy Orchestration

Implements AI models for legendary investors:
- Warren Buffett (Value Investing)
- Peter Lynch (Growth at Reasonable Price)
- Benjamin Graham (Deep Value)
- Ray Dalio (All Weather)
- Custom momentum and mean reversion strategies
- Economic macro strategy
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass

from ..strategies.warren_buffett_ai import WarrenBuffettAI
from ..strategies.peter_lynch_ai import PeterLynchAI
from ..strategies.benjamin_graham_ai import BenjaminGrahamAI
from ..strategies.ray_dalio_ai import RayDalioAI
from ..strategies.momentum_strategy import MomentumStrategy
from ..strategies.mean_reversion_strategy import MeanReversionStrategy
from ..strategies.economic_macro_strategy import EconomicMacroStrategy
from ..utils.signal_aggregator import SignalAggregator


@dataclass
class StrategySignal:
    """Represents a trading signal from a strategy."""
    strategy_name: str
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    target_weight: float  # Portfolio weight recommendation
    reasoning: str
    timestamp: datetime
    metadata: Dict[str, Any] = None


class StrategyManager:
    """
    Manages and orchestrates all investment strategies.
    
    Coordinates multiple AI-driven investment strategies,
    aggregates their signals, and provides unified recommendations.
    """
    
    def __init__(self, config: Dict, data_manager):
        self.config = config
        self.data_manager = data_manager
        self.signal_aggregator = SignalAggregator(config)
        
        # Initialize all strategies
        self.strategies = self._initialize_strategies()
        
        # Strategy performance tracking
        self.strategy_performance = {}
        
        logger.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize all investment strategy AI models."""
        strategies = {}
        strategy_configs = self.config.get('investment_strategies', {})
        
        # Warren Buffett Value Strategy
        if 'warren_buffett' in strategy_configs:
            strategies['warren_buffett'] = WarrenBuffettAI(
                strategy_configs['warren_buffett'],
                self.data_manager
            )
        
        # Peter Lynch Growth Strategy
        if 'peter_lynch' in strategy_configs:
            strategies['peter_lynch'] = PeterLynchAI(
                strategy_configs['peter_lynch'],
                self.data_manager
            )
        
        # Benjamin Graham Deep Value Strategy
        if 'benjamin_graham' in strategy_configs:
            strategies['benjamin_graham'] = BenjaminGrahamAI(
                strategy_configs['benjamin_graham'],
                self.data_manager
            )
        
        # Ray Dalio All Weather Strategy
        if 'ray_dalio' in strategy_configs:
            strategies['ray_dalio'] = RayDalioAI(
                strategy_configs['ray_dalio'],
                self.data_manager
            )
        
        # Momentum Strategy
        if 'momentum_strategy' in strategy_configs:
            strategies['momentum_strategy'] = MomentumStrategy(
                strategy_configs['momentum_strategy'],
                self.data_manager
            )
        
        # Mean Reversion Strategy
        if 'mean_reversion' in strategy_configs:
            strategies['mean_reversion'] = MeanReversionStrategy(
                strategy_configs['mean_reversion'],
                self.data_manager
            )
        
        # Economic Macro Strategy
        if 'economic_macro' in strategy_configs:
            strategies['economic_macro'] = EconomicMacroStrategy(
                strategy_configs['economic_macro'],
                self.data_manager
            )
        
        return strategies
    
    async def run_all_strategies(self, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Run all strategies and collect their signals."""
        logger.info("Running all investment strategies...")
        
        all_signals = []
        
        # Run strategies concurrently
        tasks = []
        for strategy_name, strategy in self.strategies.items():
            task = self._run_single_strategy(strategy_name, strategy, market_data)
            tasks.append(task)
        
        # Collect results
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for strategy_name, result in zip(self.strategies.keys(), strategy_results):
            if isinstance(result, Exception):
                logger.error(f"Strategy {strategy_name} failed: {result}")
            else:
                all_signals.extend(result)
                logger.debug(f"Strategy {strategy_name} generated {len(result)} signals")
        
        # Aggregate and rank signals
        aggregated_signals = self.signal_aggregator.aggregate_signals(all_signals)
        
        logger.info(f"Generated {len(aggregated_signals)} aggregated signals")
        return aggregated_signals
    
    async def _run_single_strategy(self, strategy_name: str, strategy, market_data: Dict[str, Any]) -> List[StrategySignal]:
        """Run a single strategy and return its signals."""
        try:
            signals = await strategy.generate_signals(market_data)
            
            # Update performance tracking
            self._update_strategy_performance(strategy_name, signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error running strategy {strategy_name}: {e}")
            return []
    
    def _update_strategy_performance(self, strategy_name: str, signals: List[StrategySignal]):
        """Update performance metrics for a strategy."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_signals': 0,
                'avg_confidence': 0.0,
                'last_run': None,
                'success_rate': 0.0
            }
        
        perf = self.strategy_performance[strategy_name]
        perf['total_signals'] += len(signals)
        perf['last_run'] = datetime.now()
        
        if signals:
            confidences = [signal.confidence for signal in signals]
            perf['avg_confidence'] = np.mean(confidences)
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all strategies."""
        return self.strategy_performance.copy()
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get the configured weights for each strategy."""
        weights = {}
        strategy_configs = self.config.get('investment_strategies', {})
        
        for strategy_name in self.strategies.keys():
            weights[strategy_name] = strategy_configs.get(strategy_name, {}).get('weight', 0.1)
        
        return weights
    
    async def get_strategy_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Get detailed recommendations for a specific symbol from all strategies."""
        market_data = await self.data_manager.get_current_market_data()
        
        recommendations = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                rec = await strategy.analyze_symbol(symbol, market_data)
                recommendations[strategy_name] = rec
            except Exception as e:
                logger.error(f"Error getting recommendation from {strategy_name} for {symbol}: {e}")
                recommendations[strategy_name] = None
        
        return recommendations
    
    async def backtest_strategy(self, strategy_name: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Backtest a specific strategy over a date range."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        strategy = self.strategies[strategy_name]
        
        # Load historical data for the period
        historical_data = await self._load_historical_data(start_date, end_date)
        
        # Run backtest
        backtest_results = await strategy.backtest(historical_data, start_date, end_date)
        
        return backtest_results
    
    async def _load_historical_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Load historical market data for backtesting."""
        # This would load historical data for the specified period
        # Implementation depends on data storage and availability
        pass
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names."""
        return list(self.strategies.keys())
    
    def enable_strategy(self, strategy_name: str):
        """Enable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = True
            logger.info(f"Enabled strategy: {strategy_name}")
    
    def disable_strategy(self, strategy_name: str):
        """Disable a strategy."""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].enabled = False
            logger.info(f"Disabled strategy: {strategy_name}")
    
    def update_strategy_weight(self, strategy_name: str, new_weight: float):
        """Update the weight of a strategy."""
        if strategy_name in self.config['investment_strategies']:
            self.config['investment_strategies'][strategy_name]['weight'] = new_weight
            logger.info(f"Updated {strategy_name} weight to {new_weight}")
    
    async def optimize_strategy_weights(self, lookback_days: int = 252) -> Dict[str, float]:
        """Optimize strategy weights based on historical performance."""
        logger.info(f"Optimizing strategy weights based on {lookback_days} days of data")
        
        # This would implement portfolio optimization
        # For now, return current weights
        return self.get_strategy_weights()
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get a summary of all strategies and their current status."""
        summary = {
            'total_strategies': len(self.strategies),
            'active_strategies': len([s for s in self.strategies.values() if getattr(s, 'enabled', True)]),
            'strategy_details': {},
            'last_updated': datetime.now()
        }
        
        for strategy_name, strategy in self.strategies.items():
            summary['strategy_details'][strategy_name] = {
                'enabled': getattr(strategy, 'enabled', True),
                'weight': self.config.get('investment_strategies', {}).get(strategy_name, {}).get('weight', 0.1),
                'description': self.config.get('investment_strategies', {}).get(strategy_name, {}).get('description', ''),
                'performance': self.strategy_performance.get(strategy_name, {})
            }
        
        return summary