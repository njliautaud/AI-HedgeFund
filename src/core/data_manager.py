#!/usr/bin/env python3
"""
Data Manager - Centralized data collection and processing

Handles:
- Stock market data (prices, fundamentals, technical indicators)
- Economic indicators (Fed data, macro metrics)
- Sentiment analysis (news, social media)
- Data caching and storage
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

import yfinance as yf
from fredapi import Fred
import finnhub
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ..utils.cache_manager import CacheManager
from ..utils.technical_indicators import TechnicalIndicators


class DataManager:
    """
    Centralized data management system for the AI hedge fund.
    
    Collects and processes:
    - Market data (stocks, ETFs, indices)
    - Economic indicators
    - Fundamental data
    - Sentiment data
    - Technical indicators
    """
    
    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.cache_manager = CacheManager(config)
        self.technical_indicators = TechnicalIndicators()
        
        # Initialize API clients
        self._initialize_apis()
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Data storage
        self.market_data = {}
        self.economic_data = {}
        self.sentiment_data = {}
        self.fundamental_data = {}
        
        logger.info("DataManager initialized")
    
    def _initialize_apis(self):
        """Initialize all API clients."""
        api_keys = self.config.get('api_keys', {})
        
        # FRED API for economic data
        if api_keys.get('fred'):
            self.fred = Fred(api_key=api_keys['fred'])
        else:
            logger.warning("FRED API key not found")
            self.fred = None
        
        # Finnhub API
        if api_keys.get('finnhub'):
            self.finnhub_client = finnhub.Client(api_key=api_keys['finnhub'])
        else:
            logger.warning("Finnhub API key not found")
            self.finnhub_client = None
        
        # Alpha Vantage API
        if api_keys.get('alpha_vantage'):
            self.av_timeseries = TimeSeries(key=api_keys['alpha_vantage'])
            self.av_fundamentals = FundamentalData(key=api_keys['alpha_vantage'])
        else:
            logger.warning("Alpha Vantage API key not found")
            self.av_timeseries = None
            self.av_fundamentals = None
    
    async def initialize(self):
        """Initialize data sources and load initial data."""
        logger.info("Initializing data sources...")
        
        # Load watchlist
        self.watchlist = await self._load_watchlist()
        
        # Load initial market data
        await self.load_market_data()
        
        logger.info(f"Loaded data for {len(self.watchlist)} symbols")
    
    async def _load_watchlist(self) -> List[str]:
        """Load the list of stocks/ETFs to monitor."""
        # Default watchlist - can be expanded
        default_watchlist = [
            # Major indices ETFs
            'SPY', 'QQQ', 'IWM', 'VTI', 'VXUS',
            # Sector ETFs
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLU', 'XLB', 'XLRE', 'XLP', 'XLY',
            # Individual stocks - Blue chips
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE',
            # Value stocks (Buffett style)
            'KO', 'PEP', 'WMT', 'CVX', 'XOM', 'T', 'VZ', 'IBM', 'GE', 'F',
            # Growth stocks
            'NFLX', 'CRM', 'ZOOM', 'SHOP', 'SQ', 'ROKU', 'TWLO', 'OKTA',
            # Commodities
            'GLD', 'SLV', 'USO', 'UNG', 'DBA'
        ]
        
        # Load from database if available
        try:
            db_watchlist = await self.db_manager.get_watchlist()
            if db_watchlist:
                return db_watchlist
        except Exception as e:
            logger.warning(f"Could not load watchlist from database: {e}")
        
        return default_watchlist
    
    async def load_market_data(self, symbols: Optional[List[str]] = None):
        """Load market data for specified symbols."""
        if symbols is None:
            symbols = self.watchlist
        
        logger.info(f"Loading market data for {len(symbols)} symbols...")
        
        # Use asyncio to load data concurrently
        tasks = [self._load_symbol_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = 0
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load data for {symbol}: {result}")
            else:
                successful += 1
        
        logger.info(f"Successfully loaded data for {successful}/{len(symbols)} symbols")
    
    async def _load_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Load comprehensive data for a single symbol."""
        try:
            # Check cache first
            cached_data = await self.cache_manager.get(f"market_data_{symbol}")
            if cached_data:
                self.market_data[symbol] = cached_data
                return cached_data
            
            # Load fresh data
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            hist_data = ticker.history(period="2y", interval="1d")
            
            # Get fundamental data
            info = ticker.info
            
            # Calculate technical indicators
            technical_data = self.technical_indicators.calculate_all(hist_data)
            
            # Combine all data
            symbol_data = {
                'symbol': symbol,
                'price_data': hist_data,
                'fundamentals': info,
                'technical_indicators': technical_data,
                'last_updated': datetime.now()
            }
            
            # Store in memory and cache
            self.market_data[symbol] = symbol_data
            await self.cache_manager.set(f"market_data_{symbol}", symbol_data)
            
            return symbol_data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise
    
    async def load_economic_data(self):
        """Load economic indicators from FRED."""
        if not self.fred:
            logger.warning("FRED API not available, skipping economic data")
            return
        
        logger.info("Loading economic indicators...")
        
        indicators = self.config.get('economic_indicators', {}).get('fed_data', [])
        
        for indicator in indicators:
            try:
                # Check cache
                cached_data = await self.cache_manager.get(f"economic_{indicator}")
                if cached_data:
                    self.economic_data[indicator] = cached_data
                    continue
                
                # Load from FRED
                data = self.fred.get_series(
                    indicator, 
                    start=datetime.now() - timedelta(days=365*5)  # 5 years
                )
                
                self.economic_data[indicator] = {
                    'data': data,
                    'last_updated': datetime.now()
                }
                
                # Cache the data
                await self.cache_manager.set(f"economic_{indicator}", self.economic_data[indicator])
                
                logger.debug(f"Loaded economic indicator: {indicator}")
                
            except Exception as e:
                logger.error(f"Error loading economic indicator {indicator}: {e}")
    
    async def initialize_sentiment_analysis(self):
        """Initialize sentiment analysis data sources."""
        logger.info("Initializing sentiment analysis...")
        
        # Load VIX data for fear/greed analysis
        await self._load_vix_data()
        
        # Load news sentiment for major symbols
        await self._load_news_sentiment()
    
    async def _load_vix_data(self):
        """Load VIX (volatility index) data."""
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1y")
            
            self.sentiment_data['vix'] = {
                'data': vix_data,
                'current_level': vix_data['Close'].iloc[-1],
                'last_updated': datetime.now()
            }
            
            logger.debug(f"VIX current level: {self.sentiment_data['vix']['current_level']:.2f}")
            
        except Exception as e:
            logger.error(f"Error loading VIX data: {e}")
    
    async def _load_news_sentiment(self):
        """Load news sentiment for major symbols."""
        if not self.finnhub_client:
            logger.warning("Finnhub API not available, skipping news sentiment")
            return
        
        major_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        for symbol in major_symbols:
            try:
                # Get recent news
                news = self.finnhub_client.company_news(
                    symbol, 
                    _from=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    to=datetime.now().strftime('%Y-%m-%d')
                )
                
                # Analyze sentiment
                sentiments = []
                for article in news[:10]:  # Analyze top 10 articles
                    if article.get('headline'):
                        sentiment = self.sentiment_analyzer.polarity_scores(article['headline'])
                        sentiments.append(sentiment['compound'])
                
                avg_sentiment = np.mean(sentiments) if sentiments else 0
                
                self.sentiment_data[symbol] = {
                    'news_sentiment': avg_sentiment,
                    'news_count': len(news),
                    'last_updated': datetime.now()
                }
                
                logger.debug(f"{symbol} news sentiment: {avg_sentiment:.3f}")
                
            except Exception as e:
                logger.error(f"Error loading news sentiment for {symbol}: {e}")
    
    async def get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data for all symbols."""
        # Refresh data if needed
        await self._refresh_stale_data()
        
        return {
            'market_data': self.market_data,
            'economic_data': self.economic_data,
            'sentiment_data': self.sentiment_data,
            'timestamp': datetime.now()
        }
    
    async def _refresh_stale_data(self):
        """Refresh data that is older than the configured interval."""
        refresh_interval = self.config.get('system', {}).get('data_refresh_interval', 300)
        cutoff_time = datetime.now() - timedelta(seconds=refresh_interval)
        
        # Check which symbols need refresh
        symbols_to_refresh = []
        for symbol, data in self.market_data.items():
            if data.get('last_updated', datetime.min) < cutoff_time:
                symbols_to_refresh.append(symbol)
        
        if symbols_to_refresh:
            logger.info(f"Refreshing data for {len(symbols_to_refresh)} symbols")
            await self.load_market_data(symbols_to_refresh)
    
    def get_fundamental_metrics(self, symbol: str) -> Dict[str, float]:
        """Extract key fundamental metrics for a symbol."""
        if symbol not in self.market_data:
            return {}
        
        fundamentals = self.market_data[symbol].get('fundamentals', {})
        
        return {
            'pe_ratio': fundamentals.get('trailingPE', 0),
            'price_to_book': fundamentals.get('priceToBook', 0),
            'debt_to_equity': fundamentals.get('debtToEquity', 0),
            'roe': fundamentals.get('returnOnEquity', 0),
            'revenue_growth': fundamentals.get('revenueGrowth', 0),
            'earnings_growth': fundamentals.get('earningsGrowth', 0),
            'free_cash_flow': fundamentals.get('freeCashflow', 0),
            'current_ratio': fundamentals.get('currentRatio', 0),
            'market_cap': fundamentals.get('marketCap', 0),
            'enterprise_value': fundamentals.get('enterpriseValue', 0),
            'peg_ratio': fundamentals.get('pegRatio', 0)
        }
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, float]:
        """Get technical indicators for a symbol."""
        if symbol not in self.market_data:
            return {}
        
        return self.market_data[symbol].get('technical_indicators', {})
    
    def get_economic_indicator(self, indicator: str) -> Optional[pd.Series]:
        """Get specific economic indicator data."""
        return self.economic_data.get(indicator, {}).get('data')
    
    def get_sentiment_score(self, symbol: str) -> float:
        """Get sentiment score for a symbol."""
        return self.sentiment_data.get(symbol, {}).get('news_sentiment', 0.0)
    
    async def add_symbol_to_watchlist(self, symbol: str):
        """Add a new symbol to the watchlist."""
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            await self._load_symbol_data(symbol)
            await self.db_manager.update_watchlist(self.watchlist)
            logger.info(f"Added {symbol} to watchlist")
    
    async def remove_symbol_from_watchlist(self, symbol: str):
        """Remove a symbol from the watchlist."""
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            if symbol in self.market_data:
                del self.market_data[symbol]
            await self.db_manager.update_watchlist(self.watchlist)
            logger.info(f"Removed {symbol} from watchlist")