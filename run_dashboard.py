#!/usr/bin/env python3
"""
AI Hedge Fund - Dashboard Runner

Script to run the main application dashboard.
"""

import asyncio
import sys
import os
from pathlib import Path
import yaml # Added for config loading
from loguru import logger

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))

# Import core components
from src.core.data_manager import DataManager
from src.core.portfolio_manager import PortfolioManager
from src.core.strategy_manager import StrategyManager
from src.core.ai_model_manager import AIModelManager
from src.agents.agent_manager import AgentManager
from src.core.db_manager import DatabaseManager
from src.core.risk_manager import RiskManager, RiskParameters

# Import the dashboard
from ui.dash_dashboard import DashDashboard
from utils.config_manager import ConfigManager

async def main():
    """Run the main application dashboard."""
    logger.info("Starting AI Hedge Fund Dashboard")

    # Load configuration
    config_path = PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    
    try:
        config = ConfigManager.load_config(str(config_path))
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Initialize managers
    # These might require API keys or other configs, ensure they are in config.yaml
    try:
        logger.info("Initializing DatabaseManager...")
        db_manager = DatabaseManager()
        logger.info("DatabaseManager initialized.")
        
        logger.info("Initializing DataManager...")
        data_manager = DataManager(config, db_manager)
        await data_manager.initialize()
        logger.info("DataManager initialized.")

        logger.info("Initializing AIModelManager...")
        ai_model_manager = AIModelManager(config)
        # await ai_model_manager.initialize() # If it has an async init
        logger.info("AIModelManager initialized.")

        logger.info("Initializing RiskManager...")
        risk_manager = RiskManager(
            config=config,
            data_manager=data_manager,
            risk_params=RiskParameters(
                max_position_size=0.05,
                max_drawdown=0.20,
                max_volatility=0.20,
                value_at_risk=0.05,
                correlation_threshold=0.80
            )
        )
        logger.info("RiskManager initialized.")
        
        logger.info("Initializing PortfolioManager...")
        portfolio_manager = PortfolioManager(config, data_manager, risk_manager)
        logger.info("PortfolioManager initialized.")

        logger.info("Initializing AgentManager...")
        agent_manager = AgentManager(config, ai_model_manager, data_manager, portfolio_manager)
        # Store ai_model_manager as an attribute if not already done
        if not hasattr(agent_manager, 'ai_model_manager'):
            agent_manager.ai_model_manager = ai_model_manager
        # Initialize agents if the method exists
        if hasattr(agent_manager, 'initialize_agents'):
            await agent_manager.initialize_agents()
        logger.info("AgentManager and agents initialized.")

        logger.info("Initializing StrategyManager...")
        strategy_manager = StrategyManager(config, data_manager, ai_model_manager)
        # Initialize strategies if the method exists
        if hasattr(strategy_manager, 'initialize'):
            await strategy_manager.initialize()
        logger.info("StrategyManager and strategies initialized.")
        
        # risk_manager = RiskManager(config, data_manager, portfolio_manager) # If you have a RiskManager
        # await risk_manager.initialize() # If it has an async init

    except Exception as e:
        logger.error(f"Error initializing managers: {e}", exc_info=True)
        sys.exit(1)

    # Create dashboard
    dashboard = DashDashboard(
        config=config,
        portfolio_manager=portfolio_manager,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        data_manager=data_manager,
        agent_manager=agent_manager
    )
    
    # Start dashboard
    try:
        await dashboard.start()
        dashboard_port = config.get('ui', {}).get('dashboard_port', 8050)  # Default to 8050 if not in config
        logger.info(f"Dashboard started at http://localhost:{dashboard_port}")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        if 'dashboard' in locals():
            if hasattr(dashboard, 'stop') and callable(dashboard.stop):
                await dashboard.stop()
        sys.exit(1)
    
    try:
        # Keep the script running
        while True:
            await asyncio.sleep(3600) # Sleep for a long time, or handle other tasks
    except KeyboardInterrupt:
        logger.info("Stopping dashboard and application...")
    finally:
        await dashboard.stop()
        # Add cleanup for other managers if needed
        # await strategy_manager.stop_strategies() # Example
        # await data_manager.close() # Example
        logger.info("Application shut down gracefully.")


if __name__ == '__main__':
    # Run the dashboard
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error running application: {e}", exc_info=True)
        sys.exit(1)