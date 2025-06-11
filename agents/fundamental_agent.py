from typing import Dict, Any
from .base_agent import BaseAgent

class FundamentalAgent(BaseAgent):
    """
    Rule-based agent for fundamental analysis of stocks.
    """
    
    def __init__(self):
        super().__init__(
            name="Fundamental Analysis Agent",
            description="Analyzes company fundamentals using rule-based metrics"
        )
        self.required_data = ['info', 'history']
    
    async def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform fundamental analysis on the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters
            
        Returns:
            Dict containing fundamental analysis results
        """
        data = self.get_stock_data(ticker)
        
        if 'error' in data:
            return {'error': data['error']}
        
        info = data['info']
        
        # Calculate fundamental metrics
        pe_ratio = info.get('trailingPE', 0)
        pb_ratio = info.get('priceToBook', 0)
        ps_ratio = info.get('priceToSalesTrailing12Months', 0)
        debt_equity = info.get('debtToEquity', 0)
        current_ratio = info.get('currentRatio', 0)
        roe = info.get('returnOnEquity', 0) * 100  # Convert to percentage
        
        # Rule-based scoring
        score = 50  # Neutral score
        
        # P/E ratio rules
        if pe_ratio < 15:
            score += 10
        elif pe_ratio > 25:
            score -= 10
            
        # P/B ratio rules
        if pb_ratio < 3:
            score += 10
        elif pb_ratio > 5:
            score -= 10
            
        # Debt/Equity rules
        if debt_equity < 0.5:
            score += 10
        elif debt_equity > 1.0:
            score -= 10
            
        # Current ratio rules
        if current_ratio > 1.5:
            score += 10
        elif current_ratio < 1.0:
            score -= 10
            
        # ROE rules
        if roe > 15:  # 15% ROE is considered good
            score += 10
        elif roe < 5:  # Below 5% is concerning
            score -= 10
            
        # Ensure score is within 0-100 range
        score = max(0, min(100, score))
        
        return {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'metrics': {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'debt_to_equity': debt_equity,
                'current_ratio': current_ratio,
                'return_on_equity': roe,
                'market_cap': info.get('marketCap', 0),
                'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
            },
            'score': score,
            'summary': self._generate_summary(score, info)
        }
    
    def _generate_summary(self, score: float, info: Dict[str, Any]) -> str:
        """Generate a summary based on the analysis."""
        company = info.get('longName', 'This company')
        
        if score >= 70:
            return f"{company} shows strong fundamentals with a score of {score:.1f}/100. " \
                   "The company appears to be undervalued with healthy financial ratios."
        elif score >= 50:
            return f"{company} has moderate fundamentals with a score of {score:.1f}/100. " \
                   "The company shows average valuation metrics and financial health."
        else:
            return f"{company} has weak fundamentals with a score of {score:.1f}/100. " \
                   "Caution is advised as the company shows concerning financial metrics."
