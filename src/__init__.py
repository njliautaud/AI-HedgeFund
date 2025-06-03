"""
AI Hedge Fund Source Package

This package contains the core functionality of the AI Hedge Fund system.
"""

# Import core modules for easier access
from src.core.data_manager import DataManager
from src.core.strategy_manager import StrategyManager
from src.core.portfolio_manager import PortfolioManager
from src.core.risk_manager import RiskManager, RiskParameters
from src.core.ai_model_manager import AIModelManager
from src.core.db_manager import DatabaseManager

# Import utility modules
from utils.config_manager import ConfigManager
# Import logger setup directly from the module path
from src.utils.logger_setup import setup_logging, get_logger
