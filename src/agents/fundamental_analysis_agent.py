"""
Fundamental Analysis Policy Agent

A rule-based agent that values companies using standard fundamental metrics.
"""
from typing import Dict, Any, Optional, List
import yfinance as yf
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import logging
from ..agents.agent_base import Agent, AgentAction

logger = logging.getLogger(__name__)

@dataclass
class FundamentalMetrics:
    """Container for fundamental financial metrics."""
    ticker: str
    current_price: float
    market_cap: float
    revenue: float  # TTM
    net_income: float  # TTM
    pe_ratio: float
    peg_ratio: float
    ps_ratio: float  # Price to Sales
    pb_ratio: float  # Price to Book
    debt_to_equity: float
    current_ratio: float
    roe: float  # Return on Equity
    roa: float  # Return on Assets
    free_cash_flow: float
    dividend_yield: float
    sector: str
    industry: str
    country: str

class FundamentalAnalysisAgent(Agent):
    """
    A rule-based agent that values companies using fundamental analysis metrics.
    """
    
    def __init__(self, agent_id: str, config: Dict = None):
        """Initialize the Fundamental Analysis Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary (optional)
        """
        config = config or {}
        config.update({
            'name': config.get('name', 'Fundamental Analyst'),
            'description': 'Values companies based on fundamental financial metrics',
            'version': '1.0'
        })
        
        super().__init__(agent_id, config)
        
        # Agent-specific attributes
        self.name = config['name']
        self.description = config['description']
        self.version = config['version']
        self.required_metrics = [
            'revenue', 'netIncome', 'trailingPE', 'pegRatio', 'priceToSalesTrailing12Months',
            'priceToBook', 'debtToEquity', 'currentRatio', 'returnOnEquity', 'returnOnAssets',
            'freeCashflow', 'dividendYield', 'sector', 'industry', 'country', 'marketCap'
        ]
    
    async def process(self, inputs: Dict[str, Any]) -> AgentAction:
        """
        Process inputs and generate an analysis action.
        
        Args:
            inputs: Dictionary containing 'ticker' key with stock symbol
            
        Returns:
            AgentAction containing the analysis results
        """
        await self.update_state('ANALYZING')
        
        try:
            ticker = inputs.get('ticker')
            if not ticker:
                raise ValueError("No ticker provided in inputs")
                
            analysis = self._analyze(ticker)
            
            # Create an action with the analysis results
            action = AgentAction(
                action_type='ANALYSIS',
                content={
                    'ticker': ticker,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                },
                metadata={
                    'agent_id': self.agent_id,
                    'agent_name': self.name,
                    'version': self.version
                }
            )
            
            await self.update_state('IDLE', action)
            return action
            
        except Exception as e:
            logger.error(f"Error in process: {str(e)}")
            await self.update_state('ERROR')
            raise
    
    def _analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a company using fundamental metrics and return a valuation.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing analysis results and predicted market cap
        """
        try:
            # Get company info from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we have all required metrics
            missing_metrics = [m for m in self.required_metrics if m not in info or info[m] is None]
            if missing_metrics:
                raise ValueError(f"Missing required metrics: {', '.join(missing_metrics)}")
            
            # Extract fundamental metrics
            metrics = FundamentalMetrics(
                ticker=ticker,
                current_price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
                market_cap=info['marketCap'],
                revenue=info['revenue'],
                net_income=info['netIncome'],
                pe_ratio=info['trailingPE'],
                peg_ratio=info['pegRatio'],
                ps_ratio=info['priceToSalesTrailing12Months'],
                pb_ratio=info['priceToBook'],
                debt_to_equity=info['debtToEquity'],
                current_ratio=info['currentRatio'],
                roe=info['returnOnEquity'],
                roa=info['returnOnAssets'],
                free_cash_flow=info['freeCashflow'],
                dividend_yield=info['dividendYield'] * 100 if info['dividendYield'] else 0,
                sector=info['sector'],
                industry=info['industry'],
                country=info.get('country', 'N/A')
            )
            
            # Calculate intrinsic value using multiple methods
            pe_valuation = self._calculate_pe_valuation(metrics)
            ps_valuation = self._calculate_ps_valuation(metrics)
            dcf_valuation = self._calculate_dcf_valuation(metrics)
            
            # Weighted average of valuation methods
            predicted_market_cap = (pe_valuation * 0.4 + 
                                  ps_valuation * 0.3 + 
                                  dcf_valuation * 0.3)
            
            # Calculate margin of safety (20% below intrinsic value)
            margin_of_safety = predicted_market_cap * 0.8
            
            # Prepare analysis result
            result = {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'current_price': metrics.current_price,
                'current_market_cap': metrics.market_cap,
                'predicted_market_cap': predicted_market_cap,
                'margin_of_safety_price': margin_of_safety / (info.get('sharesOutstanding', 1) * 1e6),
                'valuation_methods': {
                    'pe_valuation': pe_valuation,
                    'ps_valuation': ps_valuation,
                    'dcf_valuation': dcf_valuation
                },
                'fundamentals': {
                    'revenue': metrics.revenue,
                    'net_income': metrics.net_income,
                    'pe_ratio': metrics.pe_ratio,
                    'peg_ratio': metrics.peg_ratio,
                    'ps_ratio': metrics.ps_ratio,
                    'pb_ratio': metrics.pb_ratio,
                    'debt_to_equity': metrics.debt_to_equity,
                    'current_ratio': metrics.current_ratio,
                    'roe': metrics.roe,
                    'roa': metrics.roa,
                    'free_cash_flow': metrics.free_cash_flow,
                    'dividend_yield': metrics.dividend_yield,
                    'sector': metrics.sector,
                    'industry': metrics.industry,
                    'country': metrics.country
                },
                'analysis_date': datetime.now().isoformat(),
                'agent': self.name,
                'agent_version': self.version
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'analysis_date': datetime.now().isoformat(),
                'agent': self.name
            }
    
    def _calculate_pe_valuation(self, metrics: FundamentalMetrics) -> float:
        """Calculate valuation using P/E ratio approach."""
        # Use sector average P/E as a benchmark
        sector_avg_pe = self._get_sector_avg_pe(metrics.sector)
        
        # If company has negative earnings, use P/S ratio as fallback
        if metrics.pe_ratio <= 0 or metrics.net_income <= 0:
            return self._calculate_ps_valuation(metrics)
            
        # Adjust P/E based on growth and quality
        adjusted_pe = metrics.pe_ratio
        
        # Adjust for growth (PEG ratio)
        if metrics.peg_ratio and metrics.peg_ratio > 0:
            if metrics.peg_ratio < 1:  # Undervalued based on growth
                adjusted_pe *= 1.1
            elif metrics.peg_ratio > 1.5:  # Overvalued based on growth
                adjusted_pe *= 0.9
        
        # Adjust for financial health
        if metrics.debt_to_equity > 1.0:  # High debt
            adjusted_pe *= 0.9
        elif metrics.debt_to_equity < 0.3:  # Low debt
            adjusted_pe *= 1.05
            
        # Cap the P/E ratio at 30x to be conservative
        adjusted_pe = min(adjusted_pe, 30)
        
        # Calculate valuation
        return metrics.net_income * adjusted_pe
    
    def _calculate_ps_valuation(self, metrics: FundamentalMetrics) -> float:
        """Calculate valuation using P/S ratio approach."""
        # Use sector average P/S as a benchmark
        sector_avg_ps = self._get_sector_avg_ps(metrics.sector)
        
        # Adjust P/S based on profitability and growth
        adjusted_ps = metrics.ps_ratio
        
        # Adjust for profit margin
        if metrics.revenue > 0:
            profit_margin = metrics.net_income / metrics.revenue
            if profit_margin > 0.2:  # High margin
                adjusted_ps *= 1.1
            elif profit_margin < 0.05:  # Low margin
                adjusted_ps *= 0.9
        
        # Adjust for growth
        if metrics.peg_ratio and metrics.peg_ratio > 0:
            if metrics.peg_ratio < 1:  # High growth
                adjusted_ps *= 1.1
            elif metrics.peg_ratio > 1.5:  # Low growth
                adjusted_ps *= 0.9
        
        # Calculate valuation
        return metrics.revenue * adjusted_ps
    
    def _calculate_dcf_valuation(self, metrics: FundamentalMetrics, 
                               growth_rate: float = 0.05,
                               discount_rate: float = 0.1,
                               terminal_growth: float = 0.02) -> float:
        """
        Calculate valuation using Discounted Cash Flow (DCF) model.
        
        This is a simplified DCF model that projects free cash flows for 5 years.
        """
        if metrics.free_cash_flow <= 0:
            # If no positive FCF, fall back to P/E or P/S
            return (self._calculate_pe_valuation(metrics) * 0.5 + 
                   self._calculate_ps_valuation(metrics) * 0.5)
        
        # Project FCF for 5 years
        fcf = metrics.free_cash_flow
        total_pv = 0
        
        for year in range(1, 6):
            fcf *= (1 + growth_rate)
            pv = fcf / ((1 + discount_rate) ** year)
            total_pv += pv
        
        # Terminal value (perpetual growth method)
        terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value / ((1 + discount_rate) ** 5)
        
        # Total enterprise value
        enterprise_value = total_pv + pv_terminal
        
        # Convert to equity value (simplified)
        equity_value = enterprise_value - (metrics.debt_to_equity * metrics.market_cap)
        
        return max(equity_value, 0)  # Don't return negative values
    
    def _get_sector_avg_pe(self, sector: str) -> float:
        """Get average P/E ratio for a sector."""
        # These are simplified averages - in a real system, you'd fetch current data
        sector_pe = {
            'Technology': 25.0,
            'Healthcare': 22.0,
            'Financial Services': 15.0,
            'Consumer Cyclical': 20.0,
            'Industrials': 18.0,
            'Energy': 12.0,
            'Utilities': 16.0,
            'Real Estate': 18.0,
            'Consumer Defensive': 20.0,
            'Communication Services': 18.0,
            'Basic Materials': 14.0
        }
        return sector_pe.get(sector, 15.0)  # Default to 15 if sector not found
    
    def _get_sector_avg_ps(self, sector: str) -> float:
        """Get average P/S ratio for a sector."""
        # These are simplified averages - in a real system, you'd fetch current data
        sector_ps = {
            'Technology': 5.0,
            'Healthcare': 4.0,
            'Financial Services': 3.0,
            'Consumer Cyclical': 1.5,
            'Industrials': 1.8,
            'Energy': 1.2,
            'Utilities': 2.0,
            'Real Estate': 4.0,
            'Consumer Defensive': 1.5,
            'Communication Services': 2.5,
            'Basic Materials': 1.3
        }
        return sector_ps.get(sector, 2.0)  # Default to 2.0 if sector not found

    async def analyze(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a company (public method for direct usage).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing analysis results
        """
        action = await self.process({'ticker': ticker})
        return action.content['analysis']

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = FundamentalAnalysisAgent(
            agent_id="fundamental_analyst_1",
            config={
                'name': 'Fundamental Analyst',
                'description': 'Analyzes stocks using fundamental metrics',
                'version': '1.0'
            }
        )
        
        # Example analysis of a few stocks
        tickers = ["AAPL", "MSFT", "JPM", "TSLA"]
        
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            try:
                result = await agent.analyze(ticker)
                print(f"Company: {result['company_name']}")
                print(f"Current Market Cap: ${result['current_market_cap']/1e9:.2f}B")
                print(f"Predicted Market Cap: ${result['predicted_market_cap']/1e9:.2f}B")
                print(f"Margin of Safety Price: ${result['margin_of_safety_price']:.2f}")
                print(f"P/E: {result['fundamentals']['pe_ratio']:.2f}")
                print(f"PEG: {result['fundamentals']['peg_ratio']:.2f}" if result['fundamentals']['peg_ratio'] else "PEG: N/A")
                print(f"ROE: {result['fundamentals']['roe']:.2%}")
                print(f"Sector: {result['fundamentals']['sector']}")
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
    
    asyncio.run(main())
