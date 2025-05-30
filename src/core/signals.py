#!/usr/bin/env python3
"""
Defines the StrategySignal dataclass used by various strategies and the strategy manager.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

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