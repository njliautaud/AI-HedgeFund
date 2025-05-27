#!/usr/bin/env python3
"""
Founding Agent - The central orchestrator for the AI Hedge Fund

This agent oversees all strategy-specific agents, makes final investment decisions,
and provides a conversational interface for users to interact with the system.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json
from loguru import logger

# Import the base agent class
from .agent_base import Agent, AgentAction, AgentState

# For LLM integration
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


class FoundingAgent(Agent):
    """
    The Founding Agent is the central orchestrator of the AI Hedge Fund system.
    
    It has several key responsibilities:
    1. Oversee and coordinate all strategy-specific agents
    2. Aggregate signals and make final investment decisions
    3. Provide a conversational interface for users
    4. Monitor system performance and adjust agent weights
    5. Explain investment decisions and system behavior
    """
    
    def __init__(self, config: Dict, strategy_agents: Dict[str, Agent] = None):
        """
        Initialize the Founding Agent.
        
        Args:
            config: Configuration dictionary
            strategy_agents: Dictionary of strategy agents to oversee
        """
        super().__init__("founding_agent", config)
        
        # Store references to all strategy agents
        self.strategy_agents = strategy_agents or {}
        
        # Initialize LLM for conversation
        self._initialize_llm()
        
        # Conversation history
        self.conversation_history = []
        
        # Decision history
        self.decision_history = []
        
        logger.info("Founding Agent initialized")
    
    def _initialize_llm(self):
        """
        Initialize the language model for conversation.
        
        This uses a local model for conversation capabilities.
        In production, this could be replaced with an API call to a hosted model.
        """
        try:
            # Use a smaller model that can run locally
            model_name = self.config.get("llm_model", "gpt2")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.conversation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                temperature=0.7
            )
            logger.info(f"LLM initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.conversation_pipeline = None
    
    async def process(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Process inputs and generate an action.
        
        The Founding Agent can handle different types of inputs:
        - User queries (for conversation)
        - Strategy signals (for decision making)
        - System status updates (for monitoring)
        
        Args:
            inputs: Dictionary of inputs
            
        Returns:
            An AgentAction representing the agent's response
        """
        input_type = inputs.get("type", "")
        
        if input_type == "user_query":
            return await self._handle_user_query(inputs)
        elif input_type == "strategy_signals":
            return await self._handle_strategy_signals(inputs)
        elif input_type == "system_status":
            return await self._handle_system_status(inputs)
        else:
            logger.warning(f"Unknown input type: {input_type}")
            return AgentAction(
                action_type="COMMUNICATION",
                content={
                    "message": "I don't understand that input type. Please try again."
                }
            )
    
    async def _handle_user_query(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Handle a user query through the conversational interface.
        
        Args:
            inputs: Dictionary containing the user's query
            
        Returns:
            An AgentAction with the response
        """
        query = inputs.get("query", "")
        await self.update_state("COMMUNICATING")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Process the query
        response = await self._generate_response(query)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        action = AgentAction(
            action_type="COMMUNICATION",
            content={
                "message": response,
                "query": query
            }
        )
        
        await self.update_state("IDLE", action)
        return action
    
    async def _generate_response(self, query: str) -> str:
        """
        Generate a response to the user's query using the LLM.
        
        Args:
            query: The user's query
            
        Returns:
            A string response
        """
        if not self.conversation_pipeline:
            return "I'm sorry, but my conversation capabilities are currently unavailable."
        
        # Format conversation history for context
        context = "\n".join([f"{msg['role']}: {msg['content']}" 
                         for msg in self.conversation_history[-5:]])
        
        # Prepare prompt with context and query
        prompt = f"""You are the Founding Agent of an AI Hedge Fund system that oversees multiple investment strategy agents.
        
        Previous conversation:
        {context}
        
        User: {query}
        Assistant: """
        
        try:
            # Generate response
            response = self.conversation_pipeline(prompt)[0]['generated_text']
            
            # Extract just the assistant's response
            response = response.split("Assistant: ")[-1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    async def _handle_strategy_signals(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Handle signals from strategy agents and make investment decisions.
        
        Args:
            inputs: Dictionary containing strategy signals
            
        Returns:
            An AgentAction with the decision
        """
        signals = inputs.get("signals", [])
        await self.update_state("DECIDING")
        
        # Aggregate signals and make decisions
        decision = await self._make_investment_decision(signals)
        
        action = AgentAction(
            action_type="DECISION",
            content={
                "decision": decision,
                "signals": signals
            }
        )
        
        # Record decision
        self.decision_history.append({
            "timestamp": datetime.now(),
            "signals": signals,
            "decision": decision
        })
        
        await self.update_state("IDLE", action)
        return action
    
    async def _make_investment_decision(self, signals: List[Dict]) -> Dict:
        """
        Aggregate signals from strategy agents and make a final investment decision.
        
        Args:
            signals: List of signals from strategy agents
            
        Returns:
            A dictionary representing the investment decision
        """
        # This would implement sophisticated decision logic
        # For now, we'll use a simple weighted average approach
        
        if not signals:
            return {"action": "HOLD", "reasoning": "No signals received"}
        
        # Group signals by symbol
        symbol_signals = {}
        for signal in signals:
            symbol = signal.get("symbol")
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            symbol_signals[symbol].append(signal)
        
        # Make decisions for each symbol
        decisions = []
        for symbol, sym_signals in symbol_signals.items():
            # Calculate weighted confidence
            buy_confidence = 0
            sell_confidence = 0
            total_weight = 0
            
            for signal in sym_signals:
                strategy = signal.get("strategy")
                action = signal.get("action")
                confidence = signal.get("confidence", 0)
                weight = signal.get("weight", 1)
                
                total_weight += weight
                
                if action == "BUY":
                    buy_confidence += confidence * weight
                elif action == "SELL":
                    sell_confidence += confidence * weight
            
            # Normalize
            if total_weight > 0:
                buy_confidence /= total_weight
                sell_confidence /= total_weight
            
            # Determine action
            if buy_confidence > sell_confidence and buy_confidence > 0.6:
                action = "BUY"
                confidence = buy_confidence
            elif sell_confidence > buy_confidence and sell_confidence > 0.6:
                action = "SELL"
                confidence = sell_confidence
            else:
                action = "HOLD"
                confidence = max(buy_confidence, sell_confidence)
            
            decisions.append({
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "reasoning": f"Based on {len(sym_signals)} signals with buy_confidence={buy_confidence:.2f} and sell_confidence={sell_confidence:.2f}"
            })
        
        return {
            "decisions": decisions,
            "timestamp": datetime.now().isoformat(),
            "meta": {
                "total_signals": len(signals),
                "symbols_analyzed": len(symbol_signals)
            }
        }
    
    async def _handle_system_status(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Handle system status updates and monitor performance.
        
        Args:
            inputs: Dictionary containing system status information
            
        Returns:
            An AgentAction with the monitoring result
        """
        status = inputs.get("status", {})
        await self.update_state("ANALYZING")
        
        # Analyze system status and adjust agent weights if needed
        adjustments = await self._analyze_system_performance(status)
        
        action = AgentAction(
            action_type="ANALYSIS",
            content={
                "status": status,
                "adjustments": adjustments
            }
        )
        
        await self.update_state("IDLE", action)
        return action
    
    async def _analyze_system_performance(self, status: Dict) -> Dict:
        """
        Analyze system performance and make adjustments to agent weights.
        
        Args:
            status: Dictionary containing system status information
            
        Returns:
            A dictionary of adjustments to make
        """
        # This would implement sophisticated performance analysis
        # For now, we'll return a simple analysis
        
        adjustments = {
            "agent_weights": {},
            "recommendations": []
        }
        
        # Check portfolio performance
        portfolio = status.get("portfolio", {})
        returns = portfolio.get("returns", 0)
        
        # Check strategy performance
        strategy_performance = status.get("strategy_performance", {})
        
        # Make adjustments based on performance
        for strategy, perf in strategy_performance.items():
            success_rate = perf.get("success_rate", 0)
            
            # Adjust weights based on performance
            if success_rate > 0.7:
                adjustments["agent_weights"][strategy] = 1.2  # Increase weight
                adjustments["recommendations"].append(f"Increase {strategy} weight due to high success rate")
            elif success_rate < 0.3:
                adjustments["agent_weights"][strategy] = 0.8  # Decrease weight
                adjustments["recommendations"].append(f"Decrease {strategy} weight due to low success rate")
        
        return adjustments
    
    async def register_agent(self, agent_id: str, agent: Agent):
        """
        Register a new strategy agent with the Founding Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: The Agent instance to register
        """
        self.strategy_agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
    
    async def unregister_agent(self, agent_id: str):
        """
        Unregister a strategy agent from the Founding Agent.
        
        Args:
            agent_id: Unique identifier for the agent to unregister
        """
        if agent_id in self.strategy_agents:
            del self.strategy_agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def get_agent_states(self) -> Dict[str, AgentState]:
        """
        Get the current state of all registered agents.
        
        Returns:
            A dictionary mapping agent IDs to their current states
        """
        states = {"founding_agent": self.get_state()}
        
        for agent_id, agent in self.strategy_agents.items():
            states[agent_id] = agent.get_state()
        
        return states
    
    async def explain_decision(self, decision_id: str) -> str:
        """
        Generate a natural language explanation of an investment decision.
        
        Args:
            decision_id: ID of the decision to explain
            
        Returns:
            A string explanation of the decision
        """
        # Find the decision in history
        decision = None
        for d in self.decision_history:
            if str(d.get("timestamp")) == decision_id:
                decision = d
                break
        
        if not decision:
            return "Decision not found"
        
        # Generate explanation
        explanation = "I made this decision based on the following factors:\n\n"
        
        for d in decision.get("decision", {}).get("decisions", []):
            symbol = d.get("symbol")
            action = d.get("action")
            confidence = d.get("confidence")
            reasoning = d.get("reasoning")
            
            explanation += f"- {symbol}: {action} with {confidence:.2f} confidence. {reasoning}\n"
        
        return explanation