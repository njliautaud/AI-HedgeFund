from abc import ABC, abstractmethod
from typing import Dict, Any
import yfinance as yf

class BaseAgent(ABC):
    """
    Base class for all rule-based agents in the AI Hedge Fund system.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.required_data = []
    
    @abstractmethod
    async def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Analyze the given ticker and return a dictionary of metrics and insights.
        
        Args:
            ticker: Stock ticker symbol to analyze
            **kwargs: Additional parameters specific to the agent
            
        Returns:
            Dict containing analysis results with metrics and insights
        """
        pass
    
    def get_stock_data(self, ticker: str, period: str = '1y') -> Dict[str, Any]:
        """
        Get basic stock data using yfinance.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data (default: '1y')
            
        Returns:
            Dictionary containing stock data
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period=period)
            
            return {
                'info': info,
                'history': hist.to_dict('records') if hist is not None else [],
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A')
            }
        except Exception as e:
            return {'error': str(e)}
