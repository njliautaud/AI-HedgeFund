from typing import Dict, Any, List, Optional
import yfinance as yf
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class MarketAgent(BaseAgent):
    """
    Rule-based agent for market and sector analysis.
    """
    
    def __init__(self):
        super().__init__(
            name="Market Analysis Agent",
            description="Analyzes market trends and sector performance"
        )
        self.required_data = []
        
        # Major market indices to track
        self.market_indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^RUT': 'Russell 2000',
            '^VIX': 'Volatility Index'
        }
        
        # Sector ETFs for analysis
        self.sector_etfs = {
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLY': 'Consumer Cyclical',
            'XLP': 'Consumer Defensive',
            'XLB': 'Materials',
            'XLI': 'Industrials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'XLC': 'Communication Services'
        }
    
    async def analyze_market(self) -> Dict[str, Any]:
        """
        Analyze overall market conditions.
        
        Returns:
            Dict containing market analysis results
        """
        market_data = {}
        
        # Get data for each market index
        for ticker, name in self.market_indices.items():
            try:
                data = yf.Ticker(ticker).history(period='1mo')
                if not data.empty:
                    latest = data.iloc[-1]
                    prev_close = data.iloc[-2] if len(data) > 1 else latest
                    
                    market_data[ticker] = {
                        'name': name,
                        'price': latest['Close'],
                        'change': latest['Close'] - prev_close['Close'],
                        'change_pct': ((latest['Close'] - prev_close['Close']) / prev_close['Close']) * 100,
                        'volume': latest['Volume']
                    }
            except Exception as e:
                market_data[ticker] = {'error': str(e)}
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'market_indicators': market_data,
            'market_summary': self._generate_market_summary(market_data)
        }
    
    async def analyze_sectors(self) -> Dict[str, Any]:
        """
        Analyze sector performance.
        
        Returns:
            Dict containing sector analysis results
        """
        sector_data = {}
        
        # Get data for each sector ETF
        for ticker, sector in self.sector_etfs.items():
            try:
                data = yf.Ticker(ticker).history(period='1mo')
                if not data.empty:
                    latest = data.iloc[-1]
                    week_ago = data.iloc[-5] if len(data) > 5 else data.iloc[0]
                    
                    sector_data[ticker] = {
                        'sector': sector,
                        'price': latest['Close'],
                        'week_change_pct': ((latest['Close'] - week_ago['Close']) / week_ago['Close']) * 100,
                        'volume': latest['Volume']
                    }
            except Exception as e:
                sector_data[ticker] = {'error': str(e)}
        
        # Sort sectors by performance
        sorted_sectors = sorted(
            [s for s in sector_data.values() if 'error' not in s],
            key=lambda x: x['week_change_pct'],
            reverse=True
        )
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'sectors': {s['sector']: s for s in sorted_sectors},
            'top_performers': sorted_sectors[:3],
            'bottom_performers': sorted_sectors[-3:],
            'sector_summary': self._generate_sector_summary(sorted_sectors)
        }
    
    async def analyze_stock_in_context(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a stock in the context of its sector and the overall market.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing contextual analysis results
        """
        # Get stock data
        stock_data = self.get_stock_data(ticker)
        if 'error' in stock_data:
            return {'error': stock_data['error']}
        
        # Get sector data
        sector_data = await self.analyze_sectors()
        
        # Get market data
        market_data = await self.analyze_market()
        
        # Find the stock's sector
        stock_sector = stock_data['info'].get('sector', 'Unknown')
        sector_performance = next(
            (s for s in sector_data['sectors'].values() 
             if s['sector'].lower() == stock_sector.lower()),
            None
        )
        
        # Calculate relative performance
        relative_performance = 0
        if sector_performance:
            # Compare stock's performance to its sector
            stock_perf = ((stock_data['info'].get('currentPrice', 0) - 
                          stock_data['info'].get('previousClose', 0)) / 
                         stock_data['info'].get('previousClose', 1)) * 100
            relative_performance = stock_perf - sector_performance['week_change_pct']
        
        return {
            'ticker': ticker,
            'company_name': stock_data['info'].get('longName', ticker),
            'sector': stock_sector,
            'current_price': stock_data['info'].get('currentPrice', 0),
            'relative_performance': relative_performance,
            'market_conditions': market_data['market_summary'],
            'sector_conditions': sector_data['sector_summary'],
            'summary': self._generate_stock_context_summary(
                ticker, 
                stock_sector, 
                relative_performance,
                market_data['market_summary'],
                sector_data['sector_summary']
            )
        }
    
    def _generate_market_summary(self, market_data: Dict[str, Any]) -> str:
        """Generate a summary of market conditions."""
        if not market_data:
            return "No market data available."
        
        sp500 = market_data.get('^GSPC', {})
        vix = market_data.get('^VIX', {})
        
        summary = []
        
        if 'price' in sp500 and 'change_pct' in sp500:
            trend = "up" if sp500['change_pct'] >= 0 else "down"
            summary.append(f"S&P 500 is {trend} {abs(sp500['change_pct']):.2f}% today.")
        
        if 'price' in vix:
            if vix['price'] > 30:
                summary.append("High market volatility (VIX > 30).")
            elif vix['price'] < 15:
                summary.append("Low market volatility (VIX < 15).")
        
        return " ".join(summary) if summary else "Market conditions are mixed."
    
    def _generate_sector_summary(self, sectors: List[Dict[str, Any]]) -> str:
        """Generate a summary of sector performance."""
        if not sectors:
            return "No sector data available."
        
        top_sector = sectors[0]
        bottom_sector = sectors[-1]
        
        return (
            f"Top performing sector: {top_sector['sector']} ({top_sector['week_change_pct']:.2f}%). "
            f"Weakest sector: {bottom_sector['sector']} ({bottom_sector['week_change_pct']:.2f}%)."
        )
    
    def _generate_stock_context_summary(self, ticker: str, sector: str, relative_perf: float, 
                                      market_summary: str, sector_summary: str) -> str:
        """Generate a summary of the stock's context."""
        summary = []
        
        if abs(relative_perf) > 5:  # Significant outperformance/underperformance
            if relative_perf > 0:
                summary.append(f"{ticker} is significantly outperforming its {sector} sector peers.")
            else:
                summary.append(f"{ticker} is significantly underperforming its {sector} sector peers.")
        
        summary.append(market_summary)
        summary.append(sector_summary)
        
        return " ".join(summary)
