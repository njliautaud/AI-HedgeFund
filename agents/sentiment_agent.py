from typing import Dict, Any, List, Optional
import requests
from datetime import datetime, timedelta
from .base_agent import BaseAgent

class SentimentAgent(BaseAgent):
    """
    Rule-based agent for sentiment analysis of stocks.
    Uses news and social media sentiment as indicators.
    """
    
    def __init__(self, news_api_key: str = None):
        super().__init__(
            name="Sentiment Analysis Agent",
            description="Analyzes market sentiment using news and social media data"
        )
        self.news_api_key = news_api_key
        self.required_data = ['info']
    
    async def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform sentiment analysis on the given ticker.
        
        Args:
            ticker: Stock ticker symbol
            **kwargs: Additional parameters
            
        Returns:
            Dict containing sentiment analysis results
        """
        data = self.get_stock_data(ticker)
        
        if 'error' in data:
            return {'error': data['error']}
        
        # Get company name for news search
        company_name = data['info'].get('shortName', ticker)
        
        # Get news sentiment (simplified for this example)
        news_sentiment = await self._get_news_sentiment(ticker, company_name)
        
        # Calculate overall sentiment score (0-100)
        sentiment_score = 50  # Neutral
        
        if news_sentiment['article_count'] > 0:
            sentiment_score = 50 + (news_sentiment['sentiment_score'] * 25)  # Scale to 25-75 range
            sentiment_score = max(0, min(100, sentiment_score))  # Clamp to 0-100
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'sentiment': {
                'score': sentiment_score,
                'news': news_sentiment,
                'social_media': {
                    'twitter': self._get_twitter_sentiment(ticker),
                    'reddit': self._get_reddit_sentiment(ticker)
                }
            },
            'summary': self._generate_summary(sentiment_score, news_sentiment)
        }
    
    async def _get_news_sentiment(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Get news sentiment for the given ticker."""
        if not self.news_api_key:
            return {
                'article_count': 0,
                'sentiment_score': 0,
                'articles': [],
                'error': 'News API key not configured'
            }
        
        try:
            # This is a placeholder for actual news API integration
            # In a real implementation, you would call a news API here
            url = f"https://newsapi.org/v2/everything?q={company_name} {ticker}&apiKey={self.news_api_key}"
            response = requests.get(url)
            data = response.json()
            
            if response.status_code != 200 or data.get('status') != 'ok':
                return {
                    'article_count': 0,
                    'sentiment_score': 0,
                    'articles': [],
                    'error': 'Failed to fetch news'
                }
            
            # Simple sentiment analysis (simplified for this example)
            articles = data.get('articles', [])[:5]  # Limit to 5 most recent articles
            sentiment_scores = []
            processed_articles = []
            
            for article in articles:
                # In a real implementation, you would use a sentiment analysis library here
                # For this example, we'll use a simple keyword-based approach
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                # Simple keyword-based sentiment analysis
                positive_words = ['up', 'rise', 'gain', 'surge', 'boost', 'profit', 'growth']
                negative_words = ['down', 'fall', 'drop', 'plunge', 'loss', 'decline', 'slump']
                
                score = 0
                text = f"{title} {description}"
                
                for word in positive_words:
                    if word in text:
                        score += 1
                
                for word in negative_words:
                    if word in text:
                        score -= 1
                
                # Normalize score to -1 to 1 range
                score = max(-1, min(1, score / 3))  # Divide by 3 to prevent extreme scores
                sentiment_scores.append(score)
                
                processed_articles.append({
                    'title': article.get('title', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                    'sentiment_score': score
                })
            
            # Calculate average sentiment score
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'article_count': len(articles),
                'sentiment_score': avg_sentiment,
                'articles': processed_articles
            }
            
        except Exception as e:
            return {
                'article_count': 0,
                'sentiment_score': 0,
                'articles': [],
                'error': str(e)
            }
    
    def _get_twitter_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get Twitter sentiment for the given ticker."""
        # In a real implementation, you would use the Twitter API here
        # For this example, we'll return mock data
        return {
            'tweet_count': 0,
            'sentiment_score': 0,
            'top_hashtags': [],
            'error': 'Twitter API not implemented'
        }
    
    def _get_reddit_sentiment(self, ticker: str) -> Dict[str, Any]:
        """Get Reddit sentiment for the given ticker."""
        # In a real implementation, you would use the Reddit API here
        # For this example, we'll return mock data
        return {
            'post_count': 0,
            'sentiment_score': 0,
            'top_subreddits': [],
            'error': 'Reddit API not implemented'
        }
    
    def _generate_summary(self, score: float, news_sentiment: Dict[str, Any]) -> str:
        """Generate a summary based on the sentiment analysis."""
        if news_sentiment.get('article_count', 0) == 0:
            return "No recent news sentiment data available."
        
        sentiment = "neutral"
        if score >= 60:
            sentiment = "positive"
        elif score <= 40:
            sentiment = "negative"
        
        article_count = news_sentiment.get('article_count', 0)
        
        return f"Sentiment analysis shows {sentiment} sentiment based on {article_count} recent articles. " \
               f"Sentiment score: {score:.1f}/100"
