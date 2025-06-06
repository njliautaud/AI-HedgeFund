"""
Emerging Markets Data Fetcher

This module provides functionality to fetch and process data about emerging markets,
including countries, industries, and public companies exposed to these markets.
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EmergingMarket:
    """Data class representing an emerging market."""
    country: str
    gdp_growth: float  # Annual GDP growth rate
    inflation: float    # Annual inflation rate
    risk_level: str     # Low/Medium/High
    key_industries: List[str]
    etf_tickers: List[str]  # ETF tickers that track this market
    major_companies: List[Dict[str, str]]  # List of {name: str, ticker: str, sector: str}

class EmergingMarketsData:
    """Class to fetch and process emerging markets data."""
    
    def __init__(self):
        # Comprehensive list of emerging markets with real-time data fetching capabilities
        self.markets = [
            # Asia
            EmergingMarket(
                country="China",
                gdp_growth=5.2,
                inflation=2.5,
                risk_level="Medium",
                key_industries=["Technology", "Manufacturing", "E-commerce"],
                etf_tickers=["MCHI", "FXI", "ASHR"],
                major_companies=[
                    {"name": "Tencent", "ticker": "0700.HK", "sector": "Technology"},
                    {"name": "Alibaba", "ticker": "BABA", "sector": "E-commerce"},
                    {"name": "Meituan", "ticker": "3690.HK", "sector": "Technology"}
                ]
            ),
            EmergingMarket(
                country="India",
                gdp_growth=6.1,
                inflation=5.7,
                risk_level="Medium",
                key_industries=["Technology", "Pharmaceuticals", "Manufacturing"],
                etf_tickers=["INDA", "INCO", "EPI"],
                major_companies=[
                    {"name": "Tata Consultancy Services", "ticker": "TCS.NS", "sector": "Technology"},
                    {"name": "Reliance Industries", "ticker": "RELIANCE.NS", "sector": "Energy"},
                    {"name": "HDFC Bank", "ticker": "HDB", "sector": "Financial"}
                ]
            ),
            EmergingMarket(
                country="Indonesia",
                gdp_growth=5.0,
                inflation=4.9,
                risk_level="High",
                key_industries=["Commodities", "Financials", "Consumer"],
                etf_tickers=["EIDO", "IDX"],
                major_companies=[
                    {"name": "Bank Central Asia", "ticker": "BBCA.JK", "sector": "Financial"},
                    {"name": "Astra International", "ticker": "ASII.JK", "sector": "Conglomerate"},
                    {"name": "Bank Rakyat Indonesia", "ticker": "BBRI.JK", "sector": "Financial"}
                ]
            ),
            
            # Latin America
            EmergingMarket(
                country="Brazil",
                gdp_growth=2.9,
                inflation=4.2,
                risk_level="Medium",
                key_industries=["Commodities", "Financials", "Energy"],
                etf_tickers=["EWZ", "BRF", "BRZU"],
                major_companies=[
                    {"name": "Vale S.A.", "ticker": "VALE", "sector": "Materials"},
                    {"name": "Petrobras", "ticker": "PBR", "sector": "Energy"},
                    {"name": "Itaú Unibanco", "ticker": "ITUB", "sector": "Financial"}
                ]
            ),
            EmergingMarket(
                country="Mexico",
                gdp_growth=3.5,
                inflation=4.6,
                risk_level="Medium",
                key_industries=["Manufacturing", "Telecom", "Consumer"],
                etf_tickers=["EWW", "MXF"],
                major_companies=[
                    {"name": "América Móvil", "ticker": "AMX", "sector": "Telecom"},
                    {"name": "Fomento Económico Mexicano", "ticker": "FMX", "sector": "Consumer"},
                    {"name": "Grupo México", "ticker": "GMBXF", "sector": "Materials"}
                ]
            ),
            
            # EMEA (Europe, Middle East, Africa)
            EmergingMarket(
                country="South Africa",
                gdp_growth=1.9,
                inflation=5.8,
                risk_level="High",
                key_industries=["Mining", "Financials", "Telecom"],
                etf_tickers=["EZA", "EZA"],
                major_companies=[
                    {"name": "Naspers", "ticker": "NPSNY", "sector": "Technology"},
                    {"name": "MTN Group", "ticker": "MTNOY", "sector": "Telecom"},
                    {"name": "Anglo American", "ticker": "NGLOY", "sector": "Materials"}
                ]
            ),
            EmergingMarket(
                country="Saudi Arabia",
                gdp_growth=3.7,
                inflation=2.5,
                risk_level="Medium",
                key_industries=["Energy", "Financials", "Materials"],
                etf_tickers=["KSA", "MSCI_SA"],
                major_companies=[
                    {"name": "Saudi Aramco", "ticker": "2222.SR", "sector": "Energy"},
                    {"name": "Al Rajhi Bank", "ticker": "1120.SR", "sector": "Financial"},
                    {"name": "SABIC", "ticker": "2010.SR", "sector": "Materials"}
                ]
            ),
            
            # Eastern Europe
            EmergingMarket(
                country="Poland",
                gdp_growth=3.0,
                inflation=6.6,
                risk_level="Low",
                key_industries=["Manufacturing", "Financials", "IT"],
                etf_tickers=["EPOL", "PLND"],
                major_companies=[
                    {"name": "PKO Bank Polski", "ticker": "PKO.WA", "sector": "Financial"},
                    {"name": "PKN Orlen", "ticker": "PKN.WA", "sector": "Energy"},
                    {"name": "CD Projekt", "ticker": "CDR.WA", "sector": "Technology"}
                ]
            ),
            
            # Frontier Markets
            EmergingMarket(
                country="Vietnam",
                gdp_growth=7.0,
                inflation=3.2,
                risk_level="High",
                key_industries=["Manufacturing", "Technology", "Consumer Goods"],
                etf_tickers=["VNM", "FUE"],
                major_companies=[
                    {"name": "Vingroup JSC", "ticker": "VIC.HM", "sector": "Conglomerate"},
                    {"name": "Vietcombank", "ticker": "VCB.HM", "sector": "Financial"},
                    {"name": "FPT Corporation", "ticker":"FPT.HM", "sector": "Technology"}
                ]
            ),
            EmergingMarket(
                country="Nigeria",
                gdp_growth=3.1,
                inflation=21.3,
                risk_level="High",
                key_industries=["Energy", "Financials", "Telecom"],
                etf_tickers=["NGE"],
                major_companies=[
                    {"name": "Dangote Cement", "ticker": "DANGCEM.NG", "sector": "Materials"},
                    {"name": "MTN Nigeria", "ticker": "MTNN.LG", "sector": "Telecom"},
                    {"name": "Guaranty Trust Bank", "ticker": "GUARANTYB.LG", "sector": "Financial"}
                ]
            )
        ]
        
        # Sort markets by country name for consistent ordering
        self.markets.sort(key=lambda x: x.country)
    
    def get_all_markets(self) -> List[EmergingMarket]:
        """Get all emerging markets data."""
        return self.markets
    
    def get_market_by_country(self, country: str) -> EmergingMarket:
        """Get data for a specific country."""
        for market in self.markets:
            if market.country.lower() == country.lower():
                return market
        return None
    
    def get_companies_by_sector(self, sector: str) -> List[Dict]:
        """Get all companies in a specific sector across all emerging markets."""
        companies = []
        for market in self.markets:
            companies.extend([
                {**company, "country": market.country} 
                for company in market.major_companies 
                if company["sector"].lower() == sector.lower()
            ])
        return companies
    
    def get_stock_data(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get historical stock data for a ticker."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            return hist[['Close']].rename(columns={'Close': 'price'})
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

def get_emerging_markets_summary():
    """
    Get a comprehensive summary of emerging markets with detailed statistics.
    
    Returns:
        dict: A dictionary containing comprehensive market data including regional breakdowns,
              key statistics, and investment opportunities.
    """
    em_data = EmergingMarketsData()
    markets = em_data.get_all_markets()
    
    # Calculate regional breakdown
    regions = {
        "Asia": ["China", "India", "Indonesia", "Vietnam"],
        "Latin America": ["Brazil", "Mexico"],
        "EMEA": ["South Africa", "Saudi Arabia", "Poland", "Nigeria"]
    }
    
    # Initialize summary data structure
    summary = {
        "overview": {
            "total_markets": len(markets),
            "average_gdp_growth": round(sum(m.gdp_growth for m in markets) / len(markets), 2),
            "average_inflation": round(sum(m.inflation for m in markets) / len(markets), 2),
            "risk_distribution": {
                "Low": sum(1 for m in markets if m.risk_level == "Low"),
                "Medium": sum(1 for m in markets if m.risk_level == "Medium"),
                "High": sum(1 for m in markets if m.risk_level == "High")
            }
        },
        "regional_breakdown": {},
        "market_details": []
    }
    
    # Add regional data
    for region, countries in regions.items():
        region_markets = [m for m in markets if m.country in countries]
        if region_markets:
            summary["regional_breakdown"][region] = {
                "market_count": len(region_markets),
                "average_gdp_growth": round(sum(m.gdp_growth for m in region_markets) / len(region_markets), 2),
                "average_inflation": round(sum(m.inflation for m in region_markets) / len(region_markets), 2),
                "top_industries": sorted(
                    [ind for m in region_markets for ind in m.key_industries],
                    key=lambda x: sum(1 for m in region_markets if x in m.key_industries),
                    reverse=True
                )[:5]  # Top 5 industries
            }
    
    # Add detailed market information
    for market in markets:
        market_summary = {
            "country": market.country,
            "gdp_growth": market.gdp_growth,
            "inflation": market.inflation,
            "risk_level": market.risk_level,
            "key_industries": market.key_industries,
            "etf_tickers": market.etf_tickers,
            "major_companies": [
                {"name": c["name"], "ticker": c["ticker"], "sector": c["sector"]}
                for c in market.major_companies
            ],
            "investment_thesis": generate_investment_thesis(market)
        }
        summary["market_details"].append(market_summary)
    
    # Add timestamp
    from datetime import datetime
    summary["as_of"] = datetime.utcnow().isoformat()
    
    return summary

def generate_investment_thesis(market):
    """Generate an investment thesis for a given market."""
    theses = {
        "China": (
            "China remains the largest emerging market with strong growth in technology "
            "and consumer sectors. The government's focus on self-sufficiency and "
            "technological leadership presents opportunities in semiconductors, AI, and clean energy."
        ),
        "India": (
            "India's young population and digital transformation drive growth in technology "
            "services, e-commerce, and financial inclusion. Infrastructure development and "
            "manufacturing initiatives under 'Make in India' present long-term opportunities."
        ),
        "Indonesia": (
            "Indonesia's large domestic market, natural resources, and young population "
            "support long-term growth. Infrastructure development, digital transformation, "
            "and a growing middle class drive opportunities in consumer goods and services."
        ),
        "Brazil": (
            "Brazil benefits from abundant natural resources and a growing middle class. "
            "Commodity exports, particularly in agriculture and energy, drive the economy, "
            "while financial services and technology sectors show strong growth potential."
        ),
        "Mexico": (
            "Mexico benefits from proximity to the US market, with a strong manufacturing "
            "base and competitive labor costs. The USMCA trade agreement supports export-oriented "
            "industries, while domestic consumption grows with the expanding middle class."
        ),
        "South Africa": (
            "South Africa offers exposure to precious metals and mining, with a well-developed "
            "financial sector. Infrastructure development and regional trade within Africa "
            "present growth opportunities despite political and economic challenges."
        ),
        "Saudi Arabia": (
            "Saudi Arabia's Vision 2030 drives economic diversification beyond oil, with "
            "investments in technology, tourism, and renewable energy. The country's strategic "
            "location and sovereign wealth fund support long-term growth."
        ),
        "Poland": (
            "Poland benefits from EU membership, a skilled workforce, and a diversified economy. "
            "Strong manufacturing, IT services, and domestic consumption support growth, while "
            "infrastructure development continues to attract investment."
        ),
        "Vietnam": (
            "Vietnam's manufacturing sector benefits from trade diversification, with strong "
            "growth in electronics, textiles, and footwear. A young workforce and increasing "
            "foreign direct investment support long-term economic expansion."
        ),
        "Nigeria": (
            "Nigeria's large population and growing middle class drive consumer demand, while "
            "the government focuses on economic diversification beyond oil. Technology and "
            "financial services show strong growth potential despite infrastructure challenges."
        )
    }
    
    return theses.get(market.country, "Investment opportunities vary based on sector and market conditions.")

if __name__ == "__main__":
    # Example usage
    import json
    print(json.dumps(get_emerging_markets_summary(), indent=2))
