#!/usr/bin/env python3
"""
Agent Base - Foundation for all AI Hedge Fund agents

Provides the base agent class that all strategy-specific agents
and the founding agent will inherit from.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
from dataclasses import dataclass


@dataclass
class AgentAction:
    """Represents an action taken by an agent."""
    action_type: str  # 'ANALYSIS', 'SIGNAL', 'DECISION', 'COMMUNICATION'
    content: Dict[str, Any]
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_id: str
    status: str  # 'IDLE', 'ANALYZING', 'DECIDING', 'COMMUNICATING'
    last_action: Optional[AgentAction] = None
    performance_metrics: Dict[str, Any] = None
    last_updated: datetime = datetime.now()


class Agent(ABC):
    """
    Base class for all agents in the AI Hedge Fund system.
    
    All strategy-specific agents and the founding agent will inherit
    from this base class to ensure consistent interfaces and behavior.
    """
    
    def __init__(self, agent_id: str, config: Dict):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary for this agent
        """
        self.agent_id = agent_id
        self.config = config
        self.enabled = True
        self.state = AgentState(
            agent_id=agent_id,
            status='IDLE'
        )
        self.action_history = []
        logger.info(f"Agent {agent_id} initialized")
    
    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Process inputs and generate an action.
        
        This is the main method that must be implemented by all agents.
        It takes inputs relevant to the agent's domain and produces an action.
        
        Args:
            inputs: Dictionary of inputs relevant to this agent
            
        Returns:
            An AgentAction representing the agent's response
        """
        pass
    
    async def update_state(self, status: str, action: Optional[AgentAction] = None):
        """
        Update the agent's state.
        
        Args:
            status: New status for the agent
            action: Optional action that caused the state change
        """
        self.state.status = status
        self.state.last_updated = datetime.now()
        
        if action:
            self.state.last_action = action
            self.action_history.append(action)
        
        logger.debug(f"Agent {self.agent_id} state updated to {status}")
    
    def get_state(self) -> AgentState:
        """Get the current state of the agent."""
        return self.state
    
    def get_action_history(self) -> List[AgentAction]:
        """Get the history of actions taken by this agent."""
        return self.action_history
    
    def enable(self):
        """Enable this agent."""
        self.enabled = True
        logger.info(f"Agent {self.agent_id} enabled")
    
    def disable(self):
        """Disable this agent."""
        self.enabled = False
        logger.info(f"Agent {self.agent_id} disabled")
    
    def is_enabled(self) -> bool:
        """Check if this agent is enabled."""
        return self.enabled