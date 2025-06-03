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

from .base_manager import BaseManager
from ..utils.cache_manager import CacheManager
from ..utils.api_clients import APIClientFactory
from ..utils.technical_indicators import TechnicalAnalyzer


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
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Initialize API clients
        self._initialize_apis()
        
        # Sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Data storage
        self.market_data = {}
        self.economic_data = {}
        self.sentiment_data = {}
        self.fundamental_data = {}
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Data feed management
        self.data_feeds = {}
        self.running = False
        self.loop = None
        self.tasks = []
        
        logger.info("DataManager initialized")
    
    def _initialize_apis(self):
        """Initialize all API clients."""
        api_keys = self.config.get('api_keys', {})
        
        # Use yfinance as primary data source
        self.yfinance = yf
        
        # FRED API for economic data (disabled if no key)
        if api_keys.get('fred'):
            self.fred = Fred(api_key=api_keys['fred'])
        else:
            logger.info("Using yfinance for economic data")
            self.fred = None
        
        # Finnhub API (disabled if no key)
        if api_keys.get('finnhub'):
            self.finnhub_client = finnhub.Client(api_key=api_keys['finnhub'])
        else:
            logger.info("Using yfinance for real-time data")
            self.finnhub_client = None
        
        # Alpha Vantage API (disabled if no key)
        if api_keys.get('alpha_vantage'):
            self.av_timeseries = TimeSeries(key=api_keys['alpha_vantage'])
            self.av_fundamentals = FundamentalData(key=api_keys['alpha_vantage'])
        else:
            logger.info("Using yfinance for fundamental data")
            self.av_timeseries = None
            self.av_fundamentals = None
    
    async def start_data_feeds(self):
        """Start real-time data feeds for market data and other sources."""
        if self.running:
            logger.warning("Data feeds already running")
            return
            
        logger.info("Starting data feeds...")
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # Start market data feed
        market_feed_task = self.loop.create_task(self._market_data_feed())
        self.tasks.append(market_feed_task)
        
        # Start news feed if configured
        if self.config.get('data_sources', {}).get('news', {}).get('enabled', False):
            news_feed_task = self.loop.create_task(self._news_feed())
            self.tasks.append(news_feed_task)
            
        logger.info("Data feeds started")
    
    async def stop_data_feeds(self):
        """Stop all running data feeds."""
        if not self.running:
            return
            
        logger.info("Stopping data feeds...")
        self.running = False
        
        # Cancel all running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        self.tasks = []
        logger.info("Data feeds stopped")
    
    async def _load_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Load market data for a single symbol with proper caching and rate limiting."""
        try:
            # Check cache first
            cached_data = self.cache_manager.get(f"market_data:{symbol}")
            if cached_data:
                return cached_data
            
            # Rate limiting: wait before making API call
            await asyncio.sleep(1)  # Wait 1 second between requests
            
            # Use yfinance as primary data source
            ticker = yf.Ticker(symbol)
            
            # Get historical data
            data = ticker.history(period='1d')
            
            if not data.empty:
                # Store at least the price data
                result = {
                    'price_data': data
                }
                
                try:
                    # Attempt to get additional metrics
                    # Get financial data
                    financials = ticker.financials
                    balance = ticker.balance_sheet
                    cashflow = ticker.cashflow
                    
                    # Get key statistics
                    stats = ticker.stats()
                    
                    # Add additional data to result
                    result.update({
                        'financials': financials,
                        'balance_sheet': balance,
                        'cashflow': cashflow,
                        'stats': stats
                    })
                except Exception as e:
                    logger.error(f"Error loading additional data for {symbol}: {e}")
                
                # Store in cache with appropriate TTL
                cache_ttl = self.config.get('data_sources', {}).get('market_data', {}).get('ttl', 3600)
                self.cache_manager.set(f"market_data:{symbol}", result, ttl=cache_ttl)
                return result
            else:
                logger.warning(f"No data received for {symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None

    async def _market_data_feed(self):
        """Background task to fetch and process real-time market data."""
        update_interval = self.config.get('data_sources', {}).get('market_data', {}).get('update_interval', 300)
        retry_interval = self.config.get('data_sources', {}).get('market_data', {}).get('retry_interval', 10)
        max_retries = self.config.get('data_sources', {}).get('market_data', {}).get('max_retries', 5)
        
        retry_count = 0
        
        while self.running:
            try:
                symbols = self.watchlist
                if not symbols:
                    logger.warning("No symbols configured for market data feed")
                    break
                
                for symbol in symbols:
                    try:
                        # Add delay between requests to avoid rate limiting
                        await asyncio.sleep(1)
                        
                        # Try to load data with retries
                        data = None
                        for attempt in range(max_retries):
                            try:
                                data = await self._load_symbol_data(symbol)
                                if data is not None:
                                    break
                                await asyncio.sleep(retry_interval)
                            except Exception as e:
                                logger.error(f"Attempt {attempt + 1}/{max_retries} failed for {symbol}: {e}")
                                if attempt < max_retries - 1:
                                    await asyncio.sleep(retry_interval * (attempt + 1))  # Exponential backoff
                        
                        if data is not None:
                            # Extract price data
                            price_data = data['price_data']
                            if not price_data.empty:
                                self.market_data[symbol] = {
                                    'price': price_data['Close'].iloc[-1],
                                    'volume': price_data['Volume'].iloc[-1],
                                    'high': price_data['High'].iloc[-1],
                                    'low': price_data['Low'].iloc[-1]
                                }
                    except Exception as e:
                        logger.error(f"Error processing market data for {symbol}: {e}")
                        # Reset retry interval for each symbol
                        retry_interval = 10
                
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                logger.info("Market data feed cancelled")
                break
            except Exception as e:
                logger.error(f"Error in market data feed: {e}")
                # Exponential backoff for general errors
                await asyncio.sleep(retry_interval)
                retry_interval = min(retry_interval * 2, 300)  # Cap at 5 minutes
    
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

    async def load_economic_data(self, indicators: List[str] = None):
        """Load economic indicator data."""
        if indicators is None:
            indicators = self.config.get('economic_indicators', {}).get('fred_data', [])
        
        for indicator in indicators:
            try:
                # Check cache
                cached_data = await self.cache_manager.get(f"economic_{indicator}")
                if cached_data:
                    self.economic_data[indicator] = cached_data
                    continue  # Continue to next indicator if we have cached data

                # Load from FRED
                data = self.fred.get_series(
                    indicator,
                    start=datetime.now() - timedelta(days=365*5)  # 5 years
                )

                # Calculate technical indicators
                technical_data = self.technical_analyzer.calculate_all(data)

                self.economic_data[indicator] = {
                    'data': data,
                    'technical_indicators': technical_data,
                    'last_updated': datetime.now()
                }

                logger.info(f"Successfully loaded {indicator} data")
            except Exception as e:
                logger.error(f"Error loading {indicator}: {e}")
                continue  # Continue to next indicator even if one fails

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
            
            if not vix_data.empty:
                self.vix_data = {
                    'data': vix_data,
                    'last_updated': datetime.now()
                }
                logger.info("Successfully loaded VIX data")
            else:
                logger.warning("No VIX data received")
        except Exception as e:
            logger.error(f"Error loading VIX data: {e}")

    async def _load_news_sentiment(self):
        """Load news sentiment data for major symbols."""
        try:
            symbols = self.config.get('sentiment_analysis', {}).get('symbols', [])
            if not symbols:
                logger.warning("No symbols configured for news sentiment analysis")
                return
            
            for symbol in symbols:
                try:
                    # Load news data using yfinance
                    ticker = yf.Ticker(symbol)
                    news = ticker.news
                    
                    if news:
                        # Calculate sentiment scores
                        sentiment_scores = []
                        for article in news:
                            try:
                                text = article.get('title', '') + ' ' + article.get('summary', '')
                                if text:
                                    sentiment = self.sentiment_analyzer.analyze(text)
                                    sentiment_scores.append(sentiment)
                            except Exception as e:
                                logger.error(f"Error analyzing news article: {e}")
                        
                        if sentiment_scores:
                            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                            self.news_sentiment[symbol] = {
                                'sentiment': avg_sentiment,
                                'last_updated': datetime.now()
                            }
                            logger.info(f"Successfully loaded news sentiment for {symbol}")
                    else:
                        logger.warning(f"No news found for {symbol}")
                except Exception as e:
                    logger.error(f"Error loading news sentiment for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error in news sentiment loading: {e}")
            
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