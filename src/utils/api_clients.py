"""
API Clients - Provides clients for various financial data APIs.

This module contains classes for interacting with different financial data APIs
like Yahoo Finance, Alpha Vantage, FRED, and others.
"""

import asyncio
from typing import Dict, Any, Optional, List
import aiohttp
import pandas as pd
import yfinance as yf
from alpha_vantage.async_support.timeseries import TimeSeries
from fredapi import Fred
import finnhub
from loguru import logger


class APIClientFactory:
    """Factory class for creating API clients."""
    
    @staticmethod
    def get_client(api_name: str, api_key: Optional[str] = None, **kwargs):
        """Get an API client by name.
        
        Args:
            api_name: Name of the API client to create
            api_key: API key if required
            **kwargs: Additional arguments to pass to the client
            
        Returns:
            An instance of the requested API client
            
        Raises:
            ValueError: If the API name is not recognized
        """
        clients = {
            'yfinance': YFinanceClient,
            'alpha_vantage': AlphaVantageClient,
            'fred': FREDClient,
            'finnhub': FinnhubClient
        }
        
        client_class = clients.get(api_name.lower())
        if not client_class:
            raise ValueError(f"Unsupported API: {api_name}")
            
        return client_class(api_key, **kwargs)


class BaseAPIClient:
    """Base class for API clients."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the API client.
        
        Args:
            api_key: API key if required
        """
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get(self, url: str, params: Optional[Dict] = None, **kwargs):
        """Make a GET request.
        
        Args:
            url: URL to request
            params: Query parameters
            **kwargs: Additional arguments to pass to the session
            
        Returns:
            Response data
        """
        if not self.session:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, **kwargs) as response:
                    return await self._process_response(response)
        else:
            async with self.session.get(url, params=params, **kwargs) as response:
                return await self._process_response(response)
    
    async def _process_response(self, response):
        """Process the API response.
        
        Args:
            response: aiohttp.ClientResponse object
            
        Returns:
            Response data
            
        Raises:
            Exception: If the request failed
        """
        if response.status != 200:
            error_text = await response.text()
            raise Exception(f"API request failed with status {response.status}: {error_text}")
        
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            return await response.json()
        return await response.text()


class YFinanceClient(BaseAPIClient):
    """Client for Yahoo Finance API."""
    
    async def get_historical_data(self, symbol: str, period: str = "1y", interval: str = "1d"):
        """Get historical market data.
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical data
        """
        ticker = yf.Ticker(symbol)
        return ticker.history(period=period, interval=interval)
    
    async def get_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        ticker = yf.Ticker(symbol)
        return ticker.info


class AlphaVantageClient(BaseAPIClient):
    """Client for Alpha Vantage API."""
    
    def __init__(self, api_key: str):
        """Initialize the Alpha Vantage client.
        
        Args:
            api_key: Alpha Vantage API key
        """
        super().__init__(api_key)
        self.client = TimeSeries(key=api_key)
    
    async def get_historical_data(self, symbol: str, output_size: str = 'compact'):
        """Get historical stock data.
        
        Args:
            symbol: Stock symbol
            output_size: 'compact' (last 100 data points) or 'full' (20+ years)
            
        Returns:
            Tuple of (data, metadata)
        """
        loop = asyncio.get_event_loop()
        data, meta_data = await loop.run_in_executor(
            None,
            lambda: self.client.get_daily(symbol=symbol, outputsize=output_size)
        )
        return data, meta_data


class FREDClient:
    """Client for FRED (Federal Reserve Economic Data) API."""
    
    def __init__(self, api_key: str):
        """Initialize the FRED client.
        
        Args:
            api_key: FRED API key
        """
        self.client = Fred(api_key=api_key)
    
    def get_series(self, series_id: str, **kwargs) -> pd.DataFrame:
        """Get economic data series.
        
        Args:
            series_id: FRED series ID
            **kwargs: Additional arguments to pass to the FRED API
            
        Returns:
            DataFrame with the economic data
        """
        return self.client.get_series(series_id, **kwargs)
    
    def search(self, text: str, **kwargs) -> pd.DataFrame:
        """Search for economic data series.
        
        Args:
            text: Search text
            **kwargs: Additional arguments to pass to the FRED API
            
        Returns:
            DataFrame with search results
        """
        return self.client.search(text, **kwargs)


class FinnhubClient:
    """Client for Finnhub API."""
    
    def __init__(self, api_key: str):
        """Initialize the Finnhub client.
        
        Args:
            api_key: Finnhub API key
        """
        self.client = finnhub.Client(api_key=api_key)
    
    def get_company_news(self, symbol: str, _from: str, to: str) -> List[Dict]:
        """Get company news.
        
        Args:
            symbol: Stock symbol
            _from: Start date (YYYY-MM-DD)
            to: End date (YYYY-MM-DD)
            
        Returns:
            List of news articles
        """
        return self.client.company_news(symbol, _from, to)
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company profile data
        """
        return self.client.company_profile2(symbol=symbol)
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict]:
        """Get analyst recommendation trends.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of recommendation trends
        """
        return self.client.recommendation_trends(symbol)
