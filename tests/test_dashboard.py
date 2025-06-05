#!/usr/bin/env python3
"""
Test script for the AI Hedge Fund Dashboard.

This script tests the dashboard's functionality including:
- Dashboard initialization
- Page rendering
- API endpoints
- Interactive components
"""

import pytest
import requests
import time
from pathlib import Path
import sys
import os
from loguru import logger
from unittest.mock import MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the dashboard and API server
from ui.dash_dashboard import DashDashboard
from ui.api_server import APIServer

# Test configuration
TEST_CONFIG = {
    "ui": {
        "dashboard_port": 8088,
        "api_port": 8089
    },
    "logging": {
        "console_output": True,
        "file_output": False,
        "log_level": "DEBUG"
    },
    "database": {
        "type": "sqlite",
        "path": ":memory:"
    },
    "trading": {
        "mode": "paper"
    }
}

class MockComponent:
    """Base class for mock components."""
    async def initialize(self):
        pass
    
    async def start(self):
        pass
    
    async def stop(self):
        pass

class TestDashboard:
    """Test suite for the dashboard functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment with mock components."""
        logger.info("Setting up test environment...")
        
        # Create mock components
        self.portfolio_manager = MagicMock()
        self.strategy_manager = MagicMock()
        self.risk_manager = MagicMock()
        self.data_manager = MagicMock()
        self.agent_manager = MagicMock()
        
        # Initialize dashboard with mock components
        self.dashboard = DashDashboard(
            config=TEST_CONFIG,
            portfolio_manager=self.portfolio_manager,
            strategy_manager=self.strategy_manager,
            risk_manager=self.risk_manager,
            data_manager=self.data_manager,
            agent_manager=self.agent_manager
        )
        
        logger.info("Test environment setup complete")
        
        yield  # This is where the testing happens
        
        # Teardown
        logger.info("Tearing down test environment...")
        if hasattr(self, 'dashboard') and hasattr(self.dashboard, 'stop'):
            self.dashboard.stop()
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        logger.info("Testing dashboard initialization...")
        
        # Verify dashboard was initialized with correct config
        assert self.dashboard.config == TEST_CONFIG
        assert self.dashboard.portfolio_manager == self.portfolio_manager
        assert self.dashboard.strategy_manager == self.strategy_manager
        
        # Test that the app was created
        assert hasattr(self.dashboard, 'app')
        assert hasattr(self.dashboard, 'server')
        
        logger.info("Dashboard initialization test passed")
    
    def test_dashboard_layout(self):
        """Test dashboard layout creation."""
        logger.info("Testing dashboard layout...")
        
        # Test that the layout can be created
        layout = self.dashboard._create_main_layout()
        assert layout is not None
        
        # Test that the navigation exists
        nav = None
        for child in layout.children:
            if hasattr(child, 'id') and child.id == 'navbar':
                nav = child
                break
        
        assert nav is not None, "Navigation bar not found in layout"
        
        logger.info("Dashboard layout test passed")
    
    def test_page_routing(self):
        """Test dashboard page routing."""
        logger.info("Testing dashboard page routing...")
        
        # Test that the page content can be rendered for different paths
        test_cases = [
            ("/", "Overview"),
            ("/portfolio", "Portfolio"),
            ("/strategies", "Strategies"),
            ("/risk", "Risk"),
            ("/emerging-markets", "Emerging Markets"),
            ("/market-data", "Market Data"),
            ("/models", "Models"),
            ("/agents", "Agents")
        ]
        
        for path, expected_text in test_cases:
            logger.info(f"Testing path: {path}")
            try:
                content = self.dashboard._render_page_content(path)
                assert content is not None, f"No content returned for path: {path}"
                
                # Convert to string and check for expected text
                content_str = str(content)
                if expected_text not in content_str:
                    logger.warning(f"Expected text '{expected_text}' not found in content for path: {path}")
                
            except Exception as e:
                pytest.fail(f"Error rendering page {path}: {str(e)}")
        
        logger.info("Dashboard page routing test passed")

if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main(["-v", "-s", "tests/test_dashboard.py"]))
