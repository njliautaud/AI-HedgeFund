"""
API Endpoint Tests for AI Hedge Fund Dashboard

Tests the functionality of all API endpoints used by the dashboard.
"""

import pytest
import json
import time
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Test configuration
API_BASE_URL = "http://localhost:8089/api/v1"
TEST_TIMEOUT = 10  # seconds

class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test client and wait for API to be ready."""
        self.base_url = API_BASE_URL
        self.wait_for_api()
    
    def wait_for_api(self):
        """Wait for the API to become available."""
        import requests
        start_time = time.time()
        while time.time() - start_time < TEST_TIMEOUT:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    return
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(0.5)
        raise RuntimeError("API not available")
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        import requests
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_portfolio_endpoints(self):
        """Test portfolio-related endpoints."""
        import requests
        
        # Test getting portfolio summary
        response = requests.get(f"{self.base_url}/portfolio/summary")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "total_value" in data
        
        # Test getting portfolio holdings
        response = requests.get(f"{self.base_url}/portfolio/holdings")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_market_data_endpoints(self):
        """Test market data endpoints."""
        import requests
        
        # Test getting market indices
        response = requests.get(f"{self.base_url}/market/indices")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # Test getting stock data
        response = requests.get(f"{self.base_url}/market/stock/AAPL")
        assert response.status_code in [200, 404]  # 404 if no data available
    
    def test_risk_endpoints(self):
        """Test risk management endpoints."""
        import requests
        
        # Test getting risk metrics
        response = requests.get(f"{self.base_url}/risk/metrics")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        
        # Test getting risk alerts
        response = requests.get(f"{self.base_url}/risk/alerts")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_strategy_endpoints(self):
        """Test strategy-related endpoints."""
        import requests
        
        # Test getting strategy list
        response = requests.get(f"{self.base_url}/strategies")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are strategies, test getting details for the first one
            strategy_id = data[0]["id"]
            response = requests.get(f"{self.base_url}/strategies/{strategy_id}")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, dict)
            assert data["id"] == strategy_id

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_api_endpoints.py"])
