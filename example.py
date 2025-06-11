"""
Example script demonstrating how to use the AI Hedge Fund system.
"""
import asyncio
import os
from llm_manager import LLMInvestmentManager

def print_analysis(analysis: dict):
    """Print the analysis results in a readable format."""
    print("\n" + "="*80)
    print(f"Analysis for {analysis['ticker']} - {analysis['company_name']}")
    print(f"Sector: {analysis['sector']}")
    print(f"Current Price: ${analysis['current_price']:.2f}")
    
    print("\nFundamental Analysis:")
    print(f"- Score: {analysis['analyses']['fundamental'].get('score', 0):.1f}/100")
    print(f"- Summary: {analysis['analyses']['fundamental'].get('summary', 'No summary')}")
    
    print("\nTechnical Analysis:")
    print(f"- Score: {analysis['analyses']['technical'].get('score', 0):.1f}/100")
    print(f"- Summary: {analysis['analyses']['technical'].get('summary', 'No summary')}")
    
    print("\nSentiment Analysis:")
    print(f"- Score: {analysis['analyses']['sentiment'].get('sentiment', {}).get('score', 50):.1f}/100")
    print(f"- Summary: {analysis['analyses']['sentiment'].get('summary', 'No summary')}")
    
    print("\nMarket Context:")
    print(f"- {analysis['analyses']['context'].get('market_conditions', 'No market data')}")
    print(f"- {analysis['analyses']['context'].get('sector_conditions', 'No sector data')}")
    
    print("\nInvestment Decision:")
    print(analysis['decision_summary'])
    print("="*80 + "\n")

async def main():
    # Initialize the LLM Investment Manager
    # Make sure to set GEMINI_API_KEY environment variable or pass it here
    try:
        manager = LLMInvestmentManager()
        
        # Example tickers to analyze
        tickers = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL"]
        
        # Analyze each ticker
        for ticker in tickers:
            try:
                print(f"\nAnalyzing {ticker}...")
                analysis = await manager.analyze_ticker(ticker)
                print_analysis(analysis)
                
                # Be nice to the API
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
        
        # Get market overview
        print("\nGetting market overview...")
        market_overview = await manager.get_market_overview()
        print("\n" + "="*80)
        print("MARKET OVERVIEW")
        print("="*80)
        print(market_overview['analysis'])
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"Error initializing LLM Investment Manager: {str(e)}")
        print("Make sure you have set the GEMINI_API_KEY environment variable.")

if __name__ == "__main__":
    asyncio.run(main())
