#!/usr/bin/env python3
"""
Agent Manager - Coordinates all agents in the AI Hedge Fund system

This module manages the lifecycle of all agents, including:
- Founding agent (central orchestrator)
- Strategy agents (Warren Buffett, Peter Lynch, etc.)
- Economic agents (Fed policy, inflation, etc.)
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger

# Import agent classes
from .agent_base import Agent, AgentAction, AgentState
from .founding_agent import FoundingAgent
from .warren_buffett_agent import WarrenBuffettAgent

# Import for future strategy agents
# from .peter_lynch_agent import PeterLynchAgent
# from .benjamin_graham_agent import BenjaminGrahamAgent
# from .ray_dalio_agent import RayDalioAgent
# from .economic_agent import EconomicAgent


class AgentManager:
    """
    Manages all agents in the AI Hedge Fund system.
    
    Responsibilities:
    1. Initialize and configure all agents
    2. Route messages between agents
    3. Coordinate agent activities
    4. Monitor agent performance
    5. Provide agent status information
    """
    
    def __init__(self, config: Dict, data_manager, portfolio_manager, risk_manager):
        """
        Initialize the Agent Manager.
        
        Args:
            config: Configuration dictionary
            data_manager: Data manager instance
            portfolio_manager: Portfolio manager instance
            risk_manager: Risk manager instance
        """
        self.config = config
        self.data_manager = data_manager
        self.portfolio_manager = portfolio_manager
        self.risk_manager = risk_manager
        
        # Initialize agent containers
        self.founding_agent = None
        self.strategy_agents = {}
        self.economic_agents = {}
        
        # Message queue for inter-agent communication
        self.message_queue = asyncio.Queue()
        
        # Agent performance metrics
        self.agent_performance = {}
        
        logger.info("Agent Manager initialized")
    
    async def initialize(self):
        """
        Initialize all agents in the system.
        """
        logger.info("Initializing agents...")
        
        # Initialize strategy agents
        await self._initialize_strategy_agents()
        
        # Initialize economic agents
        await self._initialize_economic_agents()
        
        # Initialize founding agent (must be last as it needs references to other agents)
        await self._initialize_founding_agent()
        
        logger.info(f"Agent initialization complete. {len(self.strategy_agents)} strategy agents, "
                   f"{len(self.economic_agents)} economic agents, and founding agent ready.")
    
    async def _initialize_strategy_agents(self):
        """
        Initialize all strategy agents based on configuration.
        """
        strategy_configs = self.config.get('investment_strategies', {})
        
        # Warren Buffett Agent
        if 'warren_buffett' in strategy_configs:
            self.strategy_agents['warren_buffett'] = WarrenBuffettAgent(
                strategy_configs['warren_buffett'],
                self.data_manager
            )
            logger.info("Warren Buffett Agent initialized")
        
        # Add other strategy agents as they are implemented
        # Peter Lynch Agent
        # if 'peter_lynch' in strategy_configs:
        #     self.strategy_agents['peter_lynch'] = PeterLynchAgent(
        #         strategy_configs['peter_lynch'],
        #         self.data_manager
        #     )
        #     logger.info("Peter Lynch Agent initialized")
        # 
        # # Benjamin Graham Agent
        # if 'benjamin_graham' in strategy_configs:
        #     self.strategy_agents['benjamin_graham'] = BenjaminGrahamAgent(
        #         strategy_configs['benjamin_graham'],
        #         self.data_manager
        #     )
        #     logger.info("Benjamin Graham Agent initialized")
        # 
        # # Ray Dalio Agent
        # if 'ray_dalio' in strategy_configs:
        #     self.strategy_agents['ray_dalio'] = RayDalioAgent(
        #         strategy_configs['ray_dalio'],
        #         self.data_manager
        #     )
        #     logger.info("Ray Dalio Agent initialized")
    
    async def _initialize_economic_agents(self):
        """
        Initialize all economic agents based on configuration.
        """
        economic_configs = self.config.get('economic_strategies', {})
        
        # Add economic agents as they are implemented
        # Fed Policy Agent
        # if 'fed_policy' in economic_configs:
        #     self.economic_agents['fed_policy'] = EconomicAgent(
        #         'fed_policy',
        #         economic_configs['fed_policy'],
        #         self.data_manager
        #     )
        #     logger.info("Fed Policy Agent initialized")
        # 
        # # Inflation Agent
        # if 'inflation' in economic_configs:
        #     self.economic_agents['inflation'] = EconomicAgent(
        #         'inflation',
        #         economic_configs['inflation'],
        #         self.data_manager
        #     )
        #     logger.info("Inflation Agent initialized")
    
    async def _initialize_founding_agent(self):
        """
        Initialize the founding agent.
        """
        # Combine all agents for the founding agent to oversee
        all_agents = {**self.strategy_agents, **self.economic_agents}
        
        # Initialize founding agent
        self.founding_agent = FoundingAgent(self.config, all_agents)
        logger.info("Founding Agent initialized")
        
        # Register all agents with the founding agent
        for agent_id, agent in all_agents.items():
            await self.founding_agent.register_agent(agent_id, agent)
    
    async def process_market_data(self, market_data: Dict):
        """
        Process new market data through all relevant agents.
        
        Args:
            market_data: Dictionary of market data
        
        Returns:
            List of signals from all agents
        """
        logger.info("Processing market data through agents...")
        
        # Process through strategy agents
        strategy_tasks = []
        for agent_id, agent in self.strategy_agents.items():
            if agent.is_enabled():
                task = agent.process({"market_data": market_data})
                strategy_tasks.append(task)
        
        # Process through economic agents
        economic_tasks = []
        for agent_id, agent in self.economic_agents.items():
            if agent.is_enabled():
                task = agent.process({"market_data": market_data})
                economic_tasks.append(task)
        
        # Gather all results
        strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
        economic_results = await asyncio.gather(*economic_tasks, return_exceptions=True)
        
        # Extract signals from results
        all_signals = []
        
        for result in strategy_results + economic_results:
            if isinstance(result, Exception):
                logger.error(f"Agent processing error: {result}")
                continue
                
            if result.action_type == "SIGNAL":
                all_signals.extend(result.content.get("signals", []))
        
        # Send signals to founding agent for final decision
        if all_signals:
            founding_result = await self.founding_agent.process({
                "type": "strategy_signals",
                "signals": all_signals
            })
            
            # Return the founding agent's decision
            return founding_result.content.get("decision")
        else:
            logger.info("No signals generated from agents")
            return {"action": "HOLD", "reasoning": "No signals received"}
    
    async def process_user_query(self, query: str) -> str:
        """
        Process a user query through the founding agent.
        
        Args:
            query: User's query string
            
        Returns:
            Response from the founding agent
        """
        logger.info(f"Processing user query: {query}")
        
        # Send query to founding agent
        result = await self.founding_agent.process({
            "type": "user_query",
            "query": query
        })
        
        # Return the response message
        return result.content.get("message", "No response generated")
    
    async def update_system_status(self):
        """
        Update system status and send to founding agent for analysis.
        
        This allows the founding agent to monitor performance and make adjustments.
        """
        # Gather system status information
        status = {
            "portfolio": self.portfolio_manager.get_portfolio_status(),
            "risk": self.risk_manager.get_risk_metrics(),
            "strategy_performance": self.agent_performance,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to founding agent
        result = await self.founding_agent.process({
            "type": "system_status",
            "status": status
        })
        
        # Apply any adjustments recommended by the founding agent
        adjustments = result.content.get("adjustments", {})
        
        # Update agent weights if recommended
        agent_weights = adjustments.get("agent_weights", {})
        for agent_id, weight in agent_weights.items():
            if agent_id in self.strategy_agents:
                self.strategy_agents[agent_id].config['weight'] = weight
                logger.info(f"Updated {agent_id} weight to {weight}")
        
        return adjustments
    
    async def get_agent_states(self) -> Dict[str, AgentState]:
        """
        Get the current state of all agents.
        
        Returns:
            Dictionary mapping agent IDs to their current states
        """
        states = {}
        
        # Get founding agent state
        if self.founding_agent:
            states["founding_agent"] = self.founding_agent.get_state()
        
        # Get strategy agent states
        for agent_id, agent in self.strategy_agents.items():
            states[agent_id] = agent.get_state()
        
        # Get economic agent states
        for agent_id, agent in self.economic_agents.items():
            states[agent_id] = agent.get_state()
        
        return states
    
    def enable_agent(self, agent_id: str):
        """
        Enable an agent.
        
        Args:
            agent_id: ID of the agent to enable
        """
        if agent_id in self.strategy_agents:
            self.strategy_agents[agent_id].enable()
        elif agent_id in self.economic_agents:
            self.economic_agents[agent_id].enable()
        else:
            logger.warning(f"Agent {agent_id} not found")
    
    def disable_agent(self, agent_id: str):
        """
        Disable an agent.
        
        Args:
            agent_id: ID of the agent to disable
        """
        if agent_id in self.strategy_agents:
            self.strategy_agents[agent_id].disable()
        elif agent_id in self.economic_agents:
            self.economic_agents[agent_id].disable()
        else:
            logger.warning(f"Agent {agent_id} not found")
    
    def get_active_agents(self) -> List[str]:
        """
        Get a list of all active agent IDs.
        
        Returns:
            List of active agent IDs
        """
        active_agents = []
        
        for agent_id, agent in self.strategy_agents.items():
            if agent.is_enabled():
                active_agents.append(agent_id)
        
        for agent_id, agent in self.economic_agents.items():
            if agent.is_enabled():
                active_agents.append(agent_id)
        
        return active_agents
    
    async def update_agent_performance(self, performance_data: Dict):
        """
        Update performance metrics for agents.
        
        Args:
            performance_data: Dictionary mapping agent IDs to performance metrics
        """
        for agent_id, metrics in performance_data.items():
            self.agent_performance[agent_id] = metrics
            
            # Update the agent's state with performance metrics
            if agent_id in self.strategy_agents:
                self.strategy_agents[agent_id].state.performance_metrics = metrics
            elif agent_id in self.economic_agents:
                self.economic_agents[agent_id].state.performance_metrics = metrics