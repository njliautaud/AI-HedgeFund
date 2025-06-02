import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from loguru import logger

from ..core.signals import StrategySignal
from datetime import datetime

@dataclass
class SignalAggregator:
    """
    Aggregates and combines signals from multiple strategies.
    
    Attributes:
        signals: List of StrategySignal objects to aggregate
        weights: Optional weights for each signal
        method: Aggregation method ('weighted', 'simple', 'voting')
    """
    
    def __init__(self, method: str = 'weighted'):
        self.signals = []
        self.weights = None
        self.method = method
        
    def add_signal(self, signal: StrategySignal, weight: float = 1.0):
        """Add a signal with optional weight."""
        self.signals.append(signal)
        if self.weights is None:
            self.weights = np.array([weight])
        else:
            self.weights = np.append(self.weights, weight)
            
    def aggregate(self) -> StrategySignal:
        """Aggregate all signals using the specified method."""
        try:
            if not self.signals:
                raise ValueError("No signals to aggregate")
                
            if self.method == 'weighted':
                return self._weighted_aggregation()
            elif self.method == 'simple':
                return self._simple_aggregation()
            elif self.method == 'voting':
                return self._voting_aggregation()
            else:
                raise ValueError(f"Unknown aggregation method: {self.method}")
                
        except Exception as e:
            logger.error(f"Error aggregating signals: {e}")
            return StrategySignal(
                strategy_name="Aggregator",
                symbol="",
                action="HOLD",
                confidence=0.0,
                target_weight=0.0,
                reasoning="Error during signal aggregation",
                timestamp=datetime.now(),
                metadata=None
            )
            
    def _weighted_aggregation(self) -> StrategySignal:
        """Weighted average of signals."""
        total_weight = np.sum(self.weights)
        normalized_weights = self.weights / total_weight
        
        # Convert actions to numerical values
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        actions = np.array([action_map[s.action] for s in self.signals])
        
        weighted_actions = np.array([a * w for a, w in zip(actions, normalized_weights)])
        weighted_confidences = np.array([s.confidence * w for s, w in zip(self.signals, normalized_weights)])
        
        combined_action = float(np.sum(weighted_actions))
        combined_confidence = float(np.sum(weighted_confidences))
        
        # Convert back to action string
        action_map_inv = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        combined_action_str = action_map_inv[int(np.sign(combined_action))]
        
        # Calculate average target weight
        target_weights = np.array([s.target_weight for s in self.signals])
        combined_weight = float(np.mean(target_weights))
        
        # Combine reasoning
        reasonings = [s.reasoning for s in self.signals]
        combined_reasoning = "\n".join(reasonings)
        
        return StrategySignal(
            strategy_name="Aggregator",
            symbol="",
            action=combined_action_str,
            confidence=combined_confidence,
            target_weight=combined_weight,
            reasoning=combined_reasoning,
            timestamp=datetime.now(),
            metadata=None
        )
        
    def _simple_aggregation(self) -> StrategySignal:
        """Simple average of signals."""
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        actions = np.array([action_map[s.action] for s in self.signals])
        confidences = np.array([s.confidence for s in self.signals])
        target_weights = np.array([s.target_weight for s in self.signals])
        
        combined_action = float(np.mean(actions))
        combined_confidence = float(np.mean(confidences))
        combined_weight = float(np.mean(target_weights))
        
        action_map_inv = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        combined_action_str = action_map_inv[int(np.sign(combined_action))]
        
        reasonings = [s.reasoning for s in self.signals]
        combined_reasoning = "\n".join(reasonings)
        
        return StrategySignal(
            strategy_name="Aggregator",
            symbol="",
            action=combined_action_str,
            confidence=combined_confidence,
            target_weight=combined_weight,
            reasoning=combined_reasoning,
            timestamp=datetime.now(),
            metadata=None
        )
        
    def _voting_aggregation(self) -> StrategySignal:
        """Majority vote for signal direction."""
        action_map = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
        actions = np.array([action_map[s.action] for s in self.signals])
        confidences = np.array([s.confidence for s in self.signals])
        target_weights = np.array([s.target_weight for s in self.signals])
        
        # Determine majority direction
        positive_votes = np.sum(actions > 0)
        negative_votes = np.sum(actions < 0)
        
        if positive_votes > negative_votes:
            combined_action = 1
        elif negative_votes > positive_votes:
            combined_action = -1
        else:
            combined_action = 0
        
        combined_confidence = float(np.mean(confidences))
        combined_weight = float(np.mean(target_weights))
        
        action_map_inv = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}
        combined_action_str = action_map_inv[combined_action]
        
        reasonings = [s.reasoning for s in self.signals]
        combined_reasoning = "\n".join(reasonings)
        
        return StrategySignal(
            strategy_name="Aggregator",
            symbol="",
            action=combined_action_str,
            confidence=combined_confidence,
            target_weight=combined_weight,
            reasoning=combined_reasoning,
            timestamp=datetime.now(),
            metadata=None
        )
