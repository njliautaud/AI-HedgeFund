#!/usr/bin/env python3
"""
AI Hedge Fund - Main Application Hub

A comprehensive AI-driven hedge fund infrastructure featuring:
- Multiple legendary investor AI strategies (Warren Buffett, etc.)
- Economic and macro strategy bots
- Real-time market analysis and portfolio management
- Advanced risk management and optimization
- Local AI model training and inference

Author: AI Hedge Fund Team
Version: 1.0.0
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from loguru import logger
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

# Core imports
from src.core.data_manager import DataManager
from src.core.strategy_manager import StrategyManager
from src.core.portfolio_manager import PortfolioManager
from src.core.risk_manager import RiskManager, RiskParameters
from src.core.ai_model_manager import AIModelManager
from src.core.db_manager import DatabaseManager

# UI imports
from ui.dash_dashboard import DashDashboard as Dashboard
from ui.api_server import APIServer

# Utils
from utils.config_manager import ConfigManager
from src.utils.logger_setup import setup_logging


class AIHedgeFund:
    """
    Main AI Hedge Fund application orchestrator.
    
    Coordinates all components:
    - Data ingestion and management
    - AI strategy execution
    - Portfolio optimization
    - Risk management
    - User interface
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize the AI Hedge Fund system."""
        self.config_path = config_path
        self.config = None
        self.running = False
        
        # Core components
        self.data_manager = None
        self.strategy_manager = None
        self.portfolio_manager = None
        self.risk_manager = None
        self.ai_model_manager = None
        
        # UI components
        self.dashboard = None
        self.api_server = None
        
        # Performance tracking
        self.start_time = None
        self.last_update = None
        
        logger.info("AI Hedge Fund system initializing...")
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            # Load configuration
            self.config = ConfigManager.load_config(self.config_path)
            
            # Setup logging
            setup_logging(self.config.get('logging', {}))
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize UI components
            await self._initialize_ui_components()
            
            # Load saved state
            await self._load_system_state()
            
            logger.info("AI Hedge Fund system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Hedge Fund system: {e}")
            raise
    
    async def _initialize_core_components(self):
        """Initialize core system components."""
        logger.info("Initializing core components...")
        
        # Database Manager - handles data persistence
        self.db_manager = DatabaseManager()
        
        # Data Manager - handles all market data
        self.data_manager = DataManager(self.config, self.db_manager)
        await self.data_manager.initialize()
        
        # AI Model Manager - manages all AI models
        self.ai_model_manager = AIModelManager(self.config)
        await self.ai_model_manager.initialize()
        
        # Risk Manager - handles risk assessment and limits
        self.risk_manager = RiskManager(
            config=self.config,
            data_manager=self.data_manager,
            risk_params=RiskParameters(
                max_position_size=0.05,
                max_drawdown=0.20,
                max_volatility=0.20,
                value_at_risk=0.05,
                correlation_threshold=0.80
            )
        )
        await self.risk_manager.initialize()
        
        # Portfolio Manager - handles portfolio construction and execution
        self.portfolio_manager = PortfolioManager(
            self.config, self.data_manager, self.risk_manager
        )
        await self.portfolio_manager.load_state()
        
        # Strategy Manager - coordinates all investment strategies
        self.strategy_manager = StrategyManager(
            self.config, self.data_manager, self.ai_model_manager
        )
        await self.strategy_manager.initialize()
        
        logger.info("Core components initialized")
    
    async def _initialize_ui_components(self):
        """Initialize user interface components."""
        logger.info("Initializing UI components...")
        
        # Web Dashboard
        self.dashboard = Dashboard(
            self.config,
            self.portfolio_manager,
            self.strategy_manager,
            self.risk_manager,
            self.data_manager
        )
        
        # API Server for external integrations
        self.api_server = APIServer(
            self.config,
            self.portfolio_manager,
            self.strategy_manager,
            self.risk_manager,
            self.data_manager
        )
        
        logger.info("UI components initialized")
    
    async def _load_system_state(self):
        """Load saved system state."""
        try:
            # Load portfolio state
            await self.portfolio_manager.load_state()
            
            # Load strategy states
            await self.strategy_manager.load_states()
            
            # Load AI model states
            await self.ai_model_manager.load_models()
            
            logger.info("System state loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load some system state: {e}")
    
    async def start(self):
        """Start the AI Hedge Fund system."""
        try:
            if self.running:
                logger.warning("System is already running")
                return
            
            self.running = True
            self.start_time = datetime.now()
            
            logger.info("Starting AI Hedge Fund system...")
            
            # Start core services
            await self._start_core_services()
            
            # Start UI services
            await self._start_ui_services()
            
            # Start main trading loop
            await self._start_trading_loop()
            
        except Exception as e:
            logger.error(f"Failed to start AI Hedge Fund system: {e}")
            await self.stop()
            raise
    
    async def _start_core_services(self):
        """Start core system services."""
        # Start data feeds
        await self.data_manager.start_data_feeds()
        
        # Start AI model inference services
        await self.ai_model_manager.start_inference_services()
        
        # Start risk monitoring
        await self.risk_manager.start_monitoring(self.portfolio_manager)
        
        logger.info("Core services started")
    
    async def _start_ui_services(self):
        """Start user interface services."""
        # Start web dashboard
        await self.dashboard.start()
        
        # Start API server
        await self.api_server.start()
        
        logger.info("UI services started")
        logger.info(f"Dashboard available at: http://localhost:{self.config.get('ui', {}).get('dashboard_port', 8080)}")
        logger.info(f"API server available at: http://localhost:{self.config.get('ui', {}).get('api_port', 8081)}")
    
    async def _start_trading_loop(self):
        """Start the main trading loop."""
        logger.info("Starting main trading loop...")
        
        update_interval = self.config.get('trading', {}).get('update_interval_seconds', 300)  # 5 minutes
        
        while self.running:
            try:
                loop_start = datetime.now()
                
                # Update market data
                await self.data_manager.update_all_data()
                
                # Generate strategy signals
                signals = await self.strategy_manager.generate_signals()
                
                # Update portfolio based on signals
                if signals:
                    await self.portfolio_manager.update_portfolio(signals)
                
                # Update risk metrics
                await self.risk_manager.update_risk_metrics()
                
                # Save system state periodically
                if (datetime.now() - (self.last_update or self.start_time)).seconds > 3600:  # Every hour
                    await self._save_system_state()
                    self.last_update = datetime.now()
                
                # Calculate sleep time
                loop_duration = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, update_interval - loop_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                else:
                    logger.warning(f"Trading loop took {loop_duration:.1f}s, longer than {update_interval}s interval")
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _save_system_state(self):
        """Save current system state."""
        try:
            await self.portfolio_manager.save_state()
            await self.strategy_manager.save_states()
            await self.ai_model_manager.save_models()
            
            logger.info("System state saved")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    async def stop(self):
        """Stop the AI Hedge Fund system."""
        logger.info("Stopping AI Hedge Fund system...")
        
        self.running = False
        
        try:
            # Save final state
            await self._save_system_state()
            
            # Stop UI services
            if self.dashboard:
                await self.dashboard.stop()
            if self.api_server:
                await self.api_server.stop()
            
            # Stop core services
            if self.risk_manager:
                await self.risk_manager.stop_monitoring()
            if self.ai_model_manager:
                await self.ai_model_manager.stop_inference_services()
            if self.data_manager:
                await self.data_manager.stop_data_feeds()
            
            logger.info("AI Hedge Fund system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            'running': self.running,
            'uptime_seconds': uptime,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'components': {
                'data_manager': self.data_manager is not None,
                'strategy_manager': self.strategy_manager is not None,
                'portfolio_manager': self.portfolio_manager is not None,
                'risk_manager': self.risk_manager is not None,
                'ai_model_manager': self.ai_model_manager is not None,
                'dashboard': self.dashboard is not None,
                'api_server': self.api_server is not None
            },
            'portfolio_summary': self.portfolio_manager.get_portfolio_summary() if self.portfolio_manager else {},
            'active_strategies': self.strategy_manager.get_active_strategies() if self.strategy_manager else [],
            'risk_metrics': self.risk_manager.get_current_metrics() if self.risk_manager else {}
        }


async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='AI Hedge Fund System')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--mode', choices=['run', 'backtest', 'train'], default='run', 
                       help='Operation mode')
    parser.add_argument('--strategy', help='Specific strategy to run (optional)')
    parser.add_argument('--start-date', help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=100000, 
                       help='Initial capital for backtesting')
    
    args = parser.parse_args()
    
    # Create AI Hedge Fund instance
    hedge_fund = AIHedgeFund(args.config)
    
    try:
        # Initialize system
        await hedge_fund.initialize()
        
        if args.mode == 'run':
            # Normal operation mode
            logger.info("Starting AI Hedge Fund in live trading mode")
            await hedge_fund.start()
            
        elif args.mode == 'backtest':
            # Backtesting mode
            logger.info("Starting backtesting mode")
            from backtesting.backtest_engine import BacktestEngine
            
            backtest_engine = BacktestEngine(
                hedge_fund.config,
                hedge_fund.strategy_manager,
                hedge_fund.data_manager
            )
            
            results = await backtest_engine.run_backtest(
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                strategy_filter=args.strategy
            )
            
            logger.info(f"Backtest completed. Results: {results}")
            
        elif args.mode == 'train':
            # Model training mode
            logger.info("Starting model training mode")
            await hedge_fund.ai_model_manager.train_all_models()
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        # Cleanup
        await hedge_fund.stop()


if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the application
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)