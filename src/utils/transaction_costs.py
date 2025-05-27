#!/usr/bin/env python3
"""
Transaction Cost Model - Trading Cost Estimation and Analysis

Models various transaction costs:
- Bid-ask spreads
- Market impact
- Commission fees
- Slippage
- Timing costs
- Opportunity costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class TransactionCost:
    """Breakdown of transaction costs."""
    symbol: str
    quantity: float
    price: float
    commission: float = 0.0
    bid_ask_spread: float = 0.0
    market_impact: float = 0.0
    slippage: float = 0.0
    timing_cost: float = 0.0
    total_cost: float = 0.0
    cost_bps: float = 0.0  # Cost in basis points
    
    def __post_init__(self):
        self.total_cost = (
            self.commission + 
            self.bid_ask_spread + 
            self.market_impact + 
            self.slippage + 
            self.timing_cost
        )
        
        notional = abs(self.quantity * self.price)
        self.cost_bps = (self.total_cost / notional * 10000) if notional > 0 else 0


@dataclass
class MarketImpactModel:
    """Market impact model parameters."""
    temporary_impact_coeff: float = 0.5
    permanent_impact_coeff: float = 0.1
    liquidity_factor: float = 1.0
    volatility_factor: float = 1.0
    size_factor: float = 1.0


class TransactionCostModel:
    """
    Comprehensive transaction cost model.
    
    Estimates various components of trading costs:
    1. Fixed costs (commissions)
    2. Proportional costs (bid-ask spreads)
    3. Market impact (temporary and permanent)
    4. Slippage and timing costs
    
    Uses industry-standard models and empirical relationships.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Commission structure
        self.commission_structure = config.get('transaction_costs', {}).get('commission', {
            'type': 'per_share',  # 'per_share', 'percentage', 'fixed'
            'rate': 0.005,  # $0.005 per share or 0.5% or $5 fixed
            'minimum': 1.0,  # Minimum commission
            'maximum': 50.0   # Maximum commission
        })
        
        # Spread model parameters
        self.spread_model = config.get('transaction_costs', {}).get('spread', {
            'base_spread_bps': 5.0,  # Base spread in basis points
            'volatility_multiplier': 2.0,
            'liquidity_adjustment': True
        })
        
        # Market impact model
        self.impact_model = MarketImpactModel(
            temporary_impact_coeff=config.get('transaction_costs', {}).get('market_impact', {}).get('temporary_coeff', 0.5),
            permanent_impact_coeff=config.get('transaction_costs', {}).get('market_impact', {}).get('permanent_coeff', 0.1),
            liquidity_factor=1.0,
            volatility_factor=1.0,
            size_factor=1.0
        )
        
        # Slippage parameters
        self.slippage_model = config.get('transaction_costs', {}).get('slippage', {
            'base_slippage_bps': 2.0,
            'size_multiplier': 1.5,
            'volatility_multiplier': 1.0
        })
        
        # Market data for cost estimation
        self.market_data_cache = {}
        
        logger.info("TransactionCostModel initialized")
    
    def estimate_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_data: Optional[Dict] = None,
        trade_urgency: str = 'normal'  # 'low', 'normal', 'high'
    ) -> float:
        """
        Estimate total transaction cost for a trade.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares
            price: Current price
            market_data: Market data for the symbol
            trade_urgency: Urgency level affecting costs
        
        Returns:
            Estimated total cost in dollars
        """
        try:
            cost_breakdown = self.estimate_cost_breakdown(
                symbol, quantity, price, market_data, trade_urgency
            )
            return cost_breakdown.total_cost
            
        except Exception as e:
            logger.error(f"Error estimating transaction cost for {symbol}: {e}")
            # Fallback: simple percentage cost
            return abs(quantity * price) * 0.001  # 10 bps
    
    def estimate_cost_breakdown(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_data: Optional[Dict] = None,
        trade_urgency: str = 'normal'
    ) -> TransactionCost:
        """
        Estimate detailed transaction cost breakdown.
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive for buy, negative for sell)
            price: Current price
            market_data: Market data including volume, volatility, etc.
            trade_urgency: Trade urgency level
        
        Returns:
            TransactionCost object with detailed breakdown
        """
        try:
            notional = abs(quantity * price)
            
            # Get or estimate market characteristics
            market_chars = self._get_market_characteristics(symbol, market_data)
            
            # Calculate each cost component
            commission = self._calculate_commission(quantity, price)
            bid_ask_spread = self._calculate_bid_ask_cost(
                symbol, quantity, price, market_chars, trade_urgency
            )
            market_impact = self._calculate_market_impact(
                symbol, quantity, price, market_chars, trade_urgency
            )
            slippage = self._calculate_slippage(
                symbol, quantity, price, market_chars, trade_urgency
            )
            timing_cost = self._calculate_timing_cost(
                symbol, quantity, price, market_chars, trade_urgency
            )
            
            return TransactionCost(
                symbol=symbol,
                quantity=quantity,
                price=price,
                commission=commission,
                bid_ask_spread=bid_ask_spread,
                market_impact=market_impact,
                slippage=slippage,
                timing_cost=timing_cost
            )
            
        except Exception as e:
            logger.error(f"Error calculating cost breakdown for {symbol}: {e}")
            # Return minimal cost estimate
            return TransactionCost(
                symbol=symbol,
                quantity=quantity,
                price=price,
                commission=max(abs(quantity) * 0.005, 1.0),
                bid_ask_spread=abs(quantity * price) * 0.0005
            )
    
    def _get_market_characteristics(self, symbol: str, market_data: Optional[Dict]) -> Dict:
        """
        Get or estimate market characteristics for cost modeling.
        """
        if market_data:
            return {
                'volatility': market_data.get('volatility', 0.20),
                'avg_volume': market_data.get('avg_volume', 1000000),
                'market_cap': market_data.get('market_cap', 1e9),
                'bid_ask_spread': market_data.get('bid_ask_spread', 0.01),
                'price': market_data.get('price', 100.0)
            }
        
        # Use cached data or defaults
        if symbol in self.market_data_cache:
            return self.market_data_cache[symbol]
        
        # Default characteristics based on symbol type
        if symbol in ['SPY', 'QQQ', 'IWM', 'VTI']:  # Large ETFs
            chars = {
                'volatility': 0.15,
                'avg_volume': 50000000,
                'market_cap': 1e11,
                'bid_ask_spread': 0.01,
                'price': 300.0
            }
        elif len(symbol) <= 4:  # Assume large cap stock
            chars = {
                'volatility': 0.25,
                'avg_volume': 5000000,
                'market_cap': 1e10,
                'bid_ask_spread': 0.02,
                'price': 150.0
            }
        else:  # Assume smaller/less liquid
            chars = {
                'volatility': 0.35,
                'avg_volume': 500000,
                'market_cap': 1e9,
                'bid_ask_spread': 0.05,
                'price': 50.0
            }
        
        self.market_data_cache[symbol] = chars
        return chars
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate commission cost.
        """
        commission_type = self.commission_structure['type']
        rate = self.commission_structure['rate']
        minimum = self.commission_structure['minimum']
        maximum = self.commission_structure['maximum']
        
        if commission_type == 'per_share':
            commission = abs(quantity) * rate
        elif commission_type == 'percentage':
            commission = abs(quantity * price) * rate
        else:  # fixed
            commission = rate
        
        # Apply min/max constraints
        commission = max(commission, minimum)
        commission = min(commission, maximum)
        
        return commission
    
    def _calculate_bid_ask_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_chars: Dict,
        trade_urgency: str
    ) -> float:
        """
        Calculate bid-ask spread cost.
        """
        base_spread_bps = self.spread_model['base_spread_bps']
        volatility_mult = self.spread_model['volatility_multiplier']
        
        # Adjust spread based on market characteristics
        volatility = market_chars['volatility']
        spread_bps = base_spread_bps * (1 + volatility_mult * (volatility - 0.20))
        
        # Adjust for liquidity
        if self.spread_model['liquidity_adjustment']:
            avg_volume = market_chars['avg_volume']
            liquidity_factor = max(0.5, min(2.0, 1000000 / avg_volume))
            spread_bps *= liquidity_factor
        
        # Adjust for urgency
        urgency_multiplier = {'low': 0.5, 'normal': 1.0, 'high': 1.5}[trade_urgency]
        spread_bps *= urgency_multiplier
        
        # Calculate cost (half spread for market orders)
        spread_cost = abs(quantity * price) * (spread_bps / 10000) * 0.5
        
        return spread_cost
    
    def _calculate_market_impact(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_chars: Dict,
        trade_urgency: str
    ) -> float:
        """
        Calculate market impact cost using square-root model.
        """
        # Trade size as percentage of average daily volume
        avg_volume = market_chars['avg_volume']
        participation_rate = abs(quantity) / avg_volume
        
        # Volatility adjustment
        volatility = market_chars['volatility']
        
        # Square-root impact model: Impact ∝ σ * sqrt(participation_rate)
        temporary_impact = (
            self.impact_model.temporary_impact_coeff * 
            volatility * 
            np.sqrt(participation_rate)
        )
        
        permanent_impact = (
            self.impact_model.permanent_impact_coeff * 
            volatility * 
            participation_rate
        )
        
        # Adjust for urgency
        urgency_multiplier = {'low': 0.7, 'normal': 1.0, 'high': 1.8}[trade_urgency]
        total_impact = (temporary_impact + permanent_impact) * urgency_multiplier
        
        # Convert to dollar cost
        impact_cost = abs(quantity * price) * total_impact
        
        return impact_cost
    
    def _calculate_slippage(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_chars: Dict,
        trade_urgency: str
    ) -> float:
        """
        Calculate slippage cost.
        """
        base_slippage_bps = self.slippage_model['base_slippage_bps']
        size_mult = self.slippage_model['size_multiplier']
        vol_mult = self.slippage_model['volatility_multiplier']
        
        # Size adjustment
        avg_volume = market_chars['avg_volume']
        size_factor = (abs(quantity) / avg_volume) ** 0.5
        
        # Volatility adjustment
        volatility = market_chars['volatility']
        vol_factor = volatility / 0.20  # Normalize to 20% vol
        
        # Calculate slippage
        slippage_bps = base_slippage_bps * (1 + size_mult * size_factor) * (1 + vol_mult * (vol_factor - 1))
        
        # Adjust for urgency
        urgency_multiplier = {'low': 0.5, 'normal': 1.0, 'high': 2.0}[trade_urgency]
        slippage_bps *= urgency_multiplier
        
        # Convert to dollar cost
        slippage_cost = abs(quantity * price) * (slippage_bps / 10000)
        
        return slippage_cost
    
    def _calculate_timing_cost(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_chars: Dict,
        trade_urgency: str
    ) -> float:
        """
        Calculate timing/delay cost.
        """
        # Timing cost depends on how long it takes to execute
        volatility = market_chars['volatility']
        
        # Estimate execution time based on trade size and urgency
        avg_volume = market_chars['avg_volume']
        participation_rate = abs(quantity) / avg_volume
        
        if trade_urgency == 'high':
            execution_time_hours = 0.1  # 6 minutes
        elif trade_urgency == 'normal':
            execution_time_hours = min(1.0, participation_rate * 8)  # Up to 1 hour
        else:  # low urgency
            execution_time_hours = min(4.0, participation_rate * 24)  # Up to 4 hours
        
        # Timing cost = volatility * sqrt(time) * notional
        timing_cost_rate = volatility * np.sqrt(execution_time_hours / (252 * 6.5))  # Annualized to hours
        timing_cost = abs(quantity * price) * timing_cost_rate * 0.5  # 50% of potential move
        
        return timing_cost
    
    def estimate_portfolio_turnover_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Estimate total cost of portfolio rebalancing.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            portfolio_value: Total portfolio value
            prices: Current prices for each symbol
        
        Returns:
            Dictionary with cost breakdown and turnover metrics
        """
        try:
            total_cost = 0.0
            turnover = 0.0
            trade_costs = {}
            
            # Calculate trades needed
            all_symbols = set(current_weights.keys()) | set(target_weights.keys())
            
            for symbol in all_symbols:
                current_weight = current_weights.get(symbol, 0.0)
                target_weight = target_weights.get(symbol, 0.0)
                weight_change = target_weight - current_weight
                
                if abs(weight_change) > 0.001:  # 0.1% threshold
                    # Calculate trade size
                    trade_value = abs(weight_change) * portfolio_value
                    price = prices.get(symbol, 100.0)  # Default price
                    quantity = trade_value / price
                    
                    # Estimate cost
                    cost = self.estimate_cost(symbol, quantity, price)
                    
                    total_cost += cost
                    turnover += abs(weight_change)
                    trade_costs[symbol] = {
                        'weight_change': weight_change,
                        'trade_value': trade_value,
                        'cost': cost,
                        'cost_bps': (cost / trade_value * 10000) if trade_value > 0 else 0
                    }
            
            return {
                'total_cost': total_cost,
                'total_cost_bps': (total_cost / portfolio_value * 10000) if portfolio_value > 0 else 0,
                'turnover': turnover,
                'num_trades': len(trade_costs),
                'trade_costs': trade_costs,
                'avg_cost_bps': np.mean([trade['cost_bps'] for trade in trade_costs.values()]) if trade_costs else 0
            }
            
        except Exception as e:
            logger.error(f"Error estimating portfolio turnover cost: {e}")
            return {
                'total_cost': 0.0,
                'total_cost_bps': 0.0,
                'turnover': 0.0,
                'num_trades': 0,
                'trade_costs': {},
                'avg_cost_bps': 0.0
            }
    
    def optimize_execution_strategy(
        self,
        symbol: str,
        target_quantity: float,
        price: float,
        time_horizon_hours: float = 4.0,
        market_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Optimize trade execution strategy to minimize costs.
        
        Args:
            symbol: Trading symbol
            target_quantity: Total quantity to trade
            price: Current price
            time_horizon_hours: Available time for execution
            market_data: Market data for the symbol
        
        Returns:
            Optimal execution strategy
        """
        try:
            market_chars = self._get_market_characteristics(symbol, market_data)
            avg_volume = market_chars['avg_volume']
            
            # Calculate optimal participation rate (typically 10-20% of volume)
            daily_volume = avg_volume
            hourly_volume = daily_volume / 6.5  # Trading hours
            
            # Target participation rate (balance between impact and timing risk)
            target_participation = min(0.20, max(0.05, abs(target_quantity) / daily_volume))
            
            # Calculate number of slices
            optimal_slice_size = target_participation * hourly_volume
            num_slices = max(1, int(abs(target_quantity) / optimal_slice_size))
            
            # Adjust for time constraint
            max_slices = max(1, int(time_horizon_hours))
            num_slices = min(num_slices, max_slices)
            
            slice_size = abs(target_quantity) / num_slices
            slice_interval_hours = time_horizon_hours / num_slices
            
            # Estimate total cost for this strategy
            total_cost = 0.0
            for i in range(num_slices):
                slice_quantity = slice_size if target_quantity > 0 else -slice_size
                urgency = 'normal' if slice_interval_hours > 0.5 else 'high'
                
                cost = self.estimate_cost(symbol, slice_quantity, price, market_data, urgency)
                total_cost += cost
            
            return {
                'strategy': 'TWAP',  # Time-Weighted Average Price
                'num_slices': num_slices,
                'slice_size': slice_size,
                'slice_interval_hours': slice_interval_hours,
                'participation_rate': target_participation,
                'estimated_total_cost': total_cost,
                'cost_bps': (total_cost / abs(target_quantity * price) * 10000) if target_quantity != 0 else 0,
                'execution_time_hours': time_horizon_hours
            }
            
        except Exception as e:
            logger.error(f"Error optimizing execution strategy for {symbol}: {e}")
            return {
                'strategy': 'MARKET',
                'num_slices': 1,
                'slice_size': abs(target_quantity),
                'estimated_total_cost': abs(target_quantity * price) * 0.01,  # 1% fallback
                'cost_bps': 100
            }
    
    def update_market_data(self, symbol: str, market_data: Dict):
        """
        Update cached market data for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Updated market characteristics
        """
        self.market_data_cache[symbol] = market_data
        logger.debug(f"Updated market data for {symbol}")
    
    def get_cost_summary(self, trades: List[TransactionCost]) -> Dict[str, Any]:
        """
        Generate summary statistics for a list of trades.
        
        Args:
            trades: List of TransactionCost objects
        
        Returns:
            Summary statistics
        """
        if not trades:
            return {}
        
        total_notional = sum(abs(trade.quantity * trade.price) for trade in trades)
        total_cost = sum(trade.total_cost for trade in trades)
        
        cost_components = {
            'commission': sum(trade.commission for trade in trades),
            'bid_ask_spread': sum(trade.bid_ask_spread for trade in trades),
            'market_impact': sum(trade.market_impact for trade in trades),
            'slippage': sum(trade.slippage for trade in trades),
            'timing_cost': sum(trade.timing_cost for trade in trades)
        }
        
        return {
            'num_trades': len(trades),
            'total_notional': total_notional,
            'total_cost': total_cost,
            'total_cost_bps': (total_cost / total_notional * 10000) if total_notional > 0 else 0,
            'avg_cost_bps': np.mean([trade.cost_bps for trade in trades]),
            'cost_components': cost_components,
            'cost_components_bps': {
                component: (cost / total_notional * 10000) if total_notional > 0 else 0
                for component, cost in cost_components.items()
            }
        }