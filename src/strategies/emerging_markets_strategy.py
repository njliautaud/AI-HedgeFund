#!/usr/bin/env python3
"""
Emerging Markets Strategy

Implements analysis of emerging market opportunities by:
- Scraping research reports from top consulting firms (McKinsey, BCG, Bain)
- Analyzing market size and growth potential
- Identifying gaps in existing solutions
- Evaluating investment potential based on market metrics
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urljoin
import re

from ..core.signals import StrategySignal

@dataclass
class MarketOpportunity:
    """Represents an emerging market investment opportunity with comprehensive analysis."""
    # Core identification (required fields)
    market_name: str
    region: str
    sector: str
    description: str
    
    # Source tracking (required)
    source_urls: List[Dict[str, str]]  # List of dicts with 'url' and 'source_name'
    last_updated: datetime
    
    # Quantitative metrics (required)
    market_size: float  # In billions USD
    growth_rate: float  # Annual growth rate percentage
    
    # Qualitative analysis (required)
    investment_thesis: str
    key_drivers: List[Dict[str, str]]  # List of dicts with 'driver' and 'impact'
    risks: List[Dict[str, str]]  # List of dicts with 'risk' and 'mitigation'
    competitive_landscape: str
    regulatory_environment: str
    
    # Optional fields with defaults (must come after all required fields)
    market_share: Optional[float] = None  # Percentage of total market
    pe_ratio: Optional[float] = None
    volatility: Optional[float] = None  # Standard deviation of returns
    
    # Calculated fields
    market_score: float = field(init=False)  # 0-100 score based on potential
    sentiment_score: float = field(init=False)  # -1 to 1 sentiment analysis
    
    def __post_init__(self):
        """Calculate market scores based on multiple factors."""
        # Size component (0-40 points)
        size_score = min(40, (self.market_size ** 0.5) * 2)
        
        # Growth component (0-30 points)
        growth_score = min(30, self.growth_rate * 1.5)
        
        # Volatility adjustment (-10 to +10 points)
        vol_adjustment = 0
        if self.volatility is not None:
            # Lower volatility is better, so we subtract from the score
            vol_adjustment = min(10, max(-10, (0.3 - self.volatility) * 20))
        
        self.market_score = size_score + growth_score + vol_adjustment
        
        # Simple sentiment analysis (placeholder - can be enhanced with NLP)
        positive_terms = sum(1 for d in self.key_drivers if any(word in d['impact'].lower() 
                                                              for word in ['growth', 'opportunity', 'expansion', 'innovation']))
        negative_terms = sum(1 for r in self.risks if any(word in r['risk'].lower() 
                                                         for word in ['risk', 'volatility', 'uncertainty', 'regulation', 'competition']))
        
        total_terms = max(1, len(self.key_drivers) + len(self.risks))
        self.sentiment_score = (positive_terms - negative_terms) / total_terms
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['last_updated'] = self.last_updated.isoformat()
        return result

class EmergingMarketsStrategy:
    """
    Identifies and evaluates emerging market opportunities.
    Uses web scraping and analysis to find high-potential markets.
    """
    
    def __init__(self, config: Dict):
        """Initialize the strategy with configuration."""
        self.config = config
        self.market_opportunities: List[MarketOpportunity] = []
        self.last_updated: Optional[datetime] = None
        self.research_sources = [
            {
                'name': 'McKinsey',
                'url': 'https://www.mckinsey.com/featured-insights',
                'base_url': 'https://www.mckinsey.com',
                'search_terms': ['emerging markets', 'market trends', 'industry trends']
            },
            {
                'name': 'BCG',
                'url': 'https://www.bcg.com/publications',
                'base_url': 'https://www.bcg.com',
                'search_terms': ['emerging markets', 'market trends']
            },
            {
                'name': 'Bain',
                'url': 'https://www.bain.com/insights/',
                'base_url': 'https://www.bain.com',
                'search_terms': ['emerging markets', 'industry trends']
            }
        ]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    async def update_market_data(self) -> None:
        """Update market data by scraping research reports."""
        logger.info("Updating emerging markets data...")
        
        try:
            # Scrape research reports
            reports = []
            for source in self.research_sources:
                try:
                    reports.extend(await self._scrape_reports(source))
                except Exception as e:
                    logger.error(f"Error scraping {source['name']}: {str(e)}")
            
            # Process reports to find market opportunities
            self._analyze_reports(reports)
            self.last_updated = datetime.utcnow()
            logger.info(f"Updated {len(self.market_opportunities)} market opportunities")
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    async def _scrape_reports(self, source: Dict) -> List[Dict]:
        """Scrape research reports from a source."""
        reports = []
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.get(source['url'], timeout=10)
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find report links (this will need to be customized per source)
            links = []
            for term in source['search_terms']:
                # This is a simplified example - actual implementation would need to be tailored
                # to each website's structure
                for a in soup.find_all('a', href=True, string=re.compile(term, re.I)):
                    url = urljoin(source['base_url'], a['href'])
                    links.append({
                        'title': a.get_text(strip=True),
                        'url': url,
                        'source': source['name']
                    })
            
            # Process each report
            for link in links[:5]:  # Limit to 5 reports per source
                try:
                    article = Article(link['url'])
                    article.download()
                    article.parse()
                    
                    reports.append({
                        'title': link['title'],
                        'url': link['url'],
                        'source': link['source'],
                        'text': article.text,
                        'publish_date': article.publish_date or datetime.utcnow(),
                        'keywords': article.keywords
                    })
                except Exception as e:
                    logger.warning(f"Error processing {link['url']}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error scraping {source['name']}: {str(e)}")
        
        return reports
    
    def _analyze_reports(self, reports: List[Dict]) -> None:
        """Analyze reports to identify market opportunities."""
        # This is a simplified example - in practice, you'd use NLP to extract
        # market opportunities from the report text
        
        # Example opportunities (in a real implementation, these would be extracted from reports)
        example_opportunities = [
            {
                'market_name': 'African Fintech',
                'description': 'Digital payment solutions in Sub-Saharan Africa',
                'market_size': 15.0,  # $15B
                'growth_rate': 25.0,  # 25% annual growth
                'key_drivers': ['Mobile penetration', 'Unbanked population', 'Regulatory support'],
                'risks': ['Regulatory changes', 'Infrastructure challenges'],
                'source_urls': ['https://www.mckinsey.com/example1']
            },
            {
                'market_name': 'Indian EdTech',
                'description': 'Online education platforms in India',
                'market_size': 10.5,  # $10.5B
                'growth_rate': 30.0,  # 30% annual growth
                'key_drivers': ['Digital adoption', 'Young population', 'Government initiatives'],
                'risks': ['Competition', 'Regulatory environment'],
                'source_urls': ['https://www.bcg.com/example2']
            },
            {
                'market_name': 'Latin American E-commerce',
                'description': 'E-commerce platforms in Latin America',
                'market_size': 85.0,  # $85B
                'growth_rate': 20.0,  # 20% annual growth
                'key_drivers': ['Mobile adoption', 'Growing middle class', 'Improved logistics'],
                'risks': ['Infrastructure', 'Economic volatility'],
                'source_urls': ['https://www.bain.com/example3']
            },
            {
                'market_name': 'Southeast Asian Digital Health',
                'description': 'Digital healthcare solutions in Southeast Asia',
                'market_size': 35.0,  # $35B
                'growth_rate': 28.0,  # 28% annual growth
                'key_drivers': ['Aging population', 'Healthcare access', 'Government support'],
                'risks': ['Regulatory hurdles', 'Data privacy concerns'],
                'source_urls': ['https://www.mckinsey.com/example4']
            },
            {
                'market_name': 'Middle East Renewable Energy',
                'description': 'Renewable energy projects in the Middle East',
                'market_size': 42.0,  # $42B
                'growth_rate': 22.0,  # 22% annual growth
                'key_drivers': ['Government initiatives', 'Falling costs', 'Energy security'],
                'risks': ['Oil price volatility', 'Geopolitical factors'],
                'source_urls': ['https://www.bcg.com/example5']
            }
        ]
        
        # Convert to MarketOpportunity objects
        self.market_opportunities = [
            MarketOpportunity(
                market_name=opp['market_name'],
                description=opp['description'],
                market_size=opp['market_size'],
                growth_rate=opp['growth_rate'],
                key_drivers=opp['key_drivers'],
                risks=opp['risks'],
                source_urls=opp['source_urls'],
                last_updated=datetime.utcnow()
            ) for opp in example_opportunities
        ]
        
        # Sort by market score (highest first)
        self.market_opportunities.sort(key=lambda x: x.market_score, reverse=True)
    
    def get_signals(self) -> List[Dict]:
        """Generate investment signals based on market opportunities."""
        signals = []
        
        for opp in self.market_opportunities:
            # Only consider markets with size > $10B and growth > 15%
            if opp.market_size >= 10.0 and opp.growth_rate >= 15.0:
                signals.append({
                    'ticker': f"EM:{opp.market_name.replace(' ', '')}",
                    'signal': 'BUY' if opp.market_score >= 60 else 'HOLD',
                    'score': opp.market_score,
                    'market_size': opp.market_size,
                    'growth_rate': opp.growth_rate,
                    'reasoning': f"High potential market: {opp.market_name} (${opp.market_size}B, {opp.growth_rate}% yoy)"
                })
        
        return signals
    
    def get_market_opportunities(self) -> List[Dict]:
        """Get detailed market opportunities for the dashboard."""
        return [
            {
                'market_name': opp.market_name,
                'description': opp.description,
                'market_size': opp.market_size,
                'growth_rate': opp.growth_rate,
                'market_score': opp.market_score,
                'key_drivers': ', '.join(opp.key_drivers),
                'risks': ', '.join(opp.risks),
                'source_urls': opp.source_urls,
                'last_updated': opp.last_updated.isoformat()
            }
            for opp in self.market_opportunities
        ]
    
    def to_dict(self) -> Dict:
        """Convert strategy data to dictionary for serialization."""
        return {
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'market_opportunities': self.get_market_opportunities()
        }
