"""
Functional tests for the AI Hedge Fund Dashboard.

These tests verify the end-to-end functionality of the dashboard.
"""

import pytest
import time
import json
import requests
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Test configuration
DASHBOARD_URL = "http://localhost:8088"
API_URL = "http://localhost:8089"

class TestDashboardFunctional:
    """Functional tests for the dashboard."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        self.session = requests.Session()
        self.dashboard_url = DASHBOARD_URL
        self.api_url = API_URL
        
        # Wait for services to start
        self._wait_for_service(self.dashboard_url)
        self._wait_for_service(f"{self.api_url}/")
        
        yield  # Test runs here
        
        # Cleanup
        self.session.close()
    
    def _wait_for_service(self, url, max_attempts=30, delay=1):
        """Wait for a service to become available."""
        for _ in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code < 500:  # Accept any non-5xx response
                    return True
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(delay)
        raise RuntimeError(f"Service not available: {url}")
    
    def test_dashboard_homepage(self):
        """Test that the dashboard homepage loads."""
        response = self.session.get(f"{self.dashboard_url}/")
        assert response.status_code == 200
        assert "AI Hedge Fund" in response.text
    
    def test_api_endpoints(self):
        """Test that API endpoints are accessible."""
        # Test API root
        response = self.session.get(f"{self.api_url}/")
        assert response.status_code == 200
        
        # Test API version
        response = self.session.get(f"{self.api_url}/api/version")
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
    
    def test_portfolio_page(self):
        """Test the portfolio page."""
        response = self.session.get(f"{self.dashboard_url}/portfolio")
        assert response.status_code == 200
        assert "Portfolio" in response.text
    
    def test_strategies_page(self):
        """Test the strategies page."""
        response = self.session.get(f"{self.dashboard_url}/strategies")
        assert response.status_code == 200
        assert "Strategies" in response.text
    
    def test_risk_page(self):
        """Test the risk page."""
        response = self.session.get(f"{self.dashboard_url}/risk")
        assert response.status_code == 200
        assert "Risk" in response.text
    
    def test_emerging_markets_page(self):
        """Test the emerging markets page."""
        response = self.session.get(f"{self.dashboard_url}/emerging-markets")
        assert response.status_code == 200
        assert "Emerging Markets" in response.text
    
    def test_market_data_page(self):
        """Test the market data page."""
        response = self.session.get(f"{self.dashboard_url}/market-data")
        assert response.status_code == 200
        assert "Market Data" in response.text
    
    def test_models_page(self):
        """Test the models page."""
        response = self.session.get(f"{self.dashboard_url}/models")
        assert response.status_code == 200
        assert "Models" in response.text
    
    def test_agents_page(self):
        """Test the agents page."""
        response = self.session.get(f"{self.dashboard_url}/agents")
        assert response.status_code == 200
        assert "Agents" in response.text

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_dashboard_functional.py"])
