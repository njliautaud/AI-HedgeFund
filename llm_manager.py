import os
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from agents import FundamentalAgent, TechnicalAgent, SentimentAgent, MarketAgent

class LLMInvestmentManager:
    """
    LLM-based investment manager that coordinates between specialized agents
    and makes final investment decisions.
    """
    
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize the LLM investment manager.
        
        Args:
            gemini_api_key: Gemini API key. If not provided, will look for GEMINI_API_KEY environment variable.
        """
        # Initialize specialized agents
        self.fundamental_agent = FundamentalAgent()
        self.technical_agent = TechnicalAgent()
        self.sentiment_agent = SentimentAgent()
        self.market_agent = MarketAgent()
        
        # Initialize Gemini
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it to the constructor.")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Define investment rules and guidelines
        self.investment_rules = """
        Investment Decision Rules:
        1. Buy when:
           - Fundamental score > 70 AND
           - Technical score > 60 AND
           - Sentiment is positive
           - Market conditions are favorable
           
        2. Strong Buy when:
           - Fundamental score > 80 AND
           - Technical score > 70 AND
           - Strong positive sentiment
           - Market conditions are very favorable
           
        3. Hold when:
           - Fundamental score between 40-70 AND
           - Technical score between 40-70
           
        4. Sell when:
           - Fundamental score < 40 OR
           - Technical score < 30
           
        5. Strong Sell when:
           - Fundamental score < 30 AND
           - Technical score < 30
           - Negative sentiment
           - Poor market conditions
        """
    
    async def analyze_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Analyze a ticker using all available agents and make an investment decision.
        
        Args:
            ticker: Stock ticker symbol to analyze
            
        Returns:
            Dict containing analysis results and investment decision
        """
        # Get analysis from all agents in parallel
        fundamental = self.fundamental_agent.analyze(ticker)
        technical = self.technical_agent.analyze(ticker)
        sentiment = self.sentiment_agent.analyze(ticker)
        market = self.market_agent.analyze_market()
        
        # Wait for all analyses to complete
        fundamental_result = await fundamental
        technical_result = await technical
        sentiment_result = await sentiment
        market_result = await market
        
        # Get stock context (sector, industry, etc.)
        stock_context = await self.market_agent.analyze_stock_in_context(ticker)
        
        # Generate a prompt for the LLM
        prompt = self._generate_decision_prompt(
            ticker,
            fundamental_result,
            technical_result,
            sentiment_result,
            market_result,
            stock_context
        )
        
        # Get decision from LLM
        try:
            response = self.model.generate_content(prompt)
            decision_summary = response.text
        except Exception as e:
            decision_summary = f"Error generating decision: {str(e)}"
        
        # Compile all results
        return {
            'ticker': ticker,
            'timestamp': stock_context.get('timestamp'),
            'company_name': stock_context.get('company_name', ticker),
            'sector': stock_context.get('sector', 'Unknown'),
            'current_price': stock_context.get('current_price', 0),
            'analyses': {
                'fundamental': fundamental_result,
                'technical': technical_result,
                'sentiment': sentiment_result,
                'market': market_result,
                'context': stock_context
            },
            'decision_summary': decision_summary
        }
    
    def _generate_decision_prompt(
        self,
        ticker: str,
        fundamental: Dict[str, Any],
        technical: Dict[str, Any],
        sentiment: Dict[str, Any],
        market: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a prompt for the LLM to make an investment decision."""
        return f"""
        You are an AI investment manager. Analyze the following information and make an investment decision for {ticker}.
        
        {self.investment_rules}
        
        Stock: {ticker} ({context.get('company_name', 'N/A')})
        Sector: {context.get('sector', 'N/A')}
        Current Price: ${context.get('current_price', 'N/A'):.2f}
        
        Fundamental Analysis:
        - Score: {fundamental.get('score', 0):.1f}/100
        - Summary: {fundamental.get('summary', 'No summary available')}
        
        Technical Analysis:
        - Score: {technical.get('score', 0):.1f}/100
        - Summary: {technical.get('summary', 'No summary available')}
        
        Sentiment Analysis:
        - Score: {sentiment.get('sentiment', {}).get('score', 50):.1f}/100
        - Summary: {sentiment.get('summary', 'No summary available')}
        
        Market Context:
        - Market Summary: {market.get('market_summary', 'No market data')}
        - Sector Performance: {context.get('sector_conditions', 'No sector data')}
        - Relative Performance: {context.get('relative_performance', 0):.2f}% vs sector
        
        Based on the above information and the investment rules, provide:
        1. A clear investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
        2. A brief explanation of the reasoning
        3. Key factors influencing the decision
        4. Any risks or considerations
        5. Suggested price targets if applicable
        """
    
    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get an overview of the current market conditions.
        
        Returns:
            Dict containing market overview and analysis
        """
        market_data = await self.market_agent.analyze_market()
        sector_data = await self.market_agent.analyze_sectors()
        
        prompt = f"""
        You are an AI market analyst. Provide a concise overview of the current market conditions based on the following data:
        
        Market Indices:
        {market_data}
        
        Sector Performance:
        {sector_data}
        
        Please provide:
        1. Overall market assessment (bullish/bearish/neutral)
        2. Best and worst performing sectors
        3. Notable market trends or anomalies
        4. Any important economic events or news that might be impacting the market
        5. General investment advice for the current market conditions
        """
        
        try:
            response = self.model.generate_content(prompt)
            analysis = response.text
        except Exception as e:
            analysis = f"Error generating market analysis: {str(e)}"
        
        return {
            'timestamp': market_data.get('timestamp'),
            'market_data': market_data,
            'sector_data': sector_data,
            'analysis': analysis
        }
