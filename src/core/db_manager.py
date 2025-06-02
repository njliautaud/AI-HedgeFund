import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger
import pandas as pd
from pathlib import Path

class DatabaseManager:
    """
    Manages the SQLite database for the AI Hedge Fund system.
    
    Handles:
    - Database initialization and schema management
    - Data persistence
    - Query optimization
    - Data integrity
    """
    
    def __init__(self, db_path: str = "hedgefund.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection = None
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize the database and create necessary tables."""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self._create_tables()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
            
    def _create_tables(self):
        """Create all necessary database tables."""
        try:
            cursor = self.connection.cursor()
            
            # Create market data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # Create economic indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS economic_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    value REAL,
                    metadata TEXT,
                    UNIQUE(indicator, timestamp)
                )
            ''')
            
            # Create sentiment data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT,
                    sentiment REAL,
                    raw_text TEXT,
                    metadata TEXT,
                    UNIQUE(source, timestamp, symbol)
                )
            ''')
            
            # Create portfolio positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity REAL,
                    price REAL,
                    type TEXT,
                    metadata TEXT,
                    UNIQUE(timestamp, symbol)
                )
            ''')
            
            self.connection.commit()
            logger.info("Database tables created")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
            
    def save_market_data(self, data: pd.DataFrame, symbol: str):
        """
        Save market data to the database.
        
        Args:
            data: DataFrame containing market data
            symbol: Stock symbol
        """
        try:
            cursor = self.connection.cursor()
            
            for _, row in data.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    row.name.to_pydatetime(),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume'],
                    row['Adj Close'] if 'Adj Close' in row else row['Close']
                ))
            
            self.connection.commit()
            logger.info(f"Saved {len(data)} market data points for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving market data: {e}")
            raise
            
    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get market data from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame of market data
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT timestamp, open, high, low, close, volume, adj_close
                FROM market_data
                WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (symbol, start_date, end_date))
            
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'adj_close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
            
    def save_economic_indicator(self, indicator: str, value: float, timestamp: datetime, metadata: Dict = None):
        """
        Save economic indicator data.
        
        Args:
            indicator: Indicator name
            value: Indicator value
            timestamp: Timestamp
            metadata: Additional metadata
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO economic_indicators 
                (indicator, timestamp, value, metadata)
                VALUES (?, ?, ?, ?)
            ''', (
                indicator,
                timestamp,
                value,
                json.dumps(metadata) if metadata else None
            ))
            
            self.connection.commit()
            logger.info(f"Saved economic indicator {indicator} for {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving economic indicator: {e}")
            raise
            
    def get_economic_indicators(self, indicator: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get economic indicator data.
        
        Args:
            indicator: Indicator name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame of indicator data
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT timestamp, value, metadata
                FROM economic_indicators
                WHERE indicator = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (indicator, start_date, end_date))
            
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(rows, columns=['timestamp', 'value', 'metadata'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting economic indicators: {e}")
            return pd.DataFrame()
            
    def save_sentiment_data(self, source: str, symbol: str, sentiment: float, raw_text: str, metadata: Dict = None):
        """
        Save sentiment data.
        
        Args:
            source: Source of sentiment data (e.g., 'twitter', 'news')
            symbol: Stock symbol
            sentiment: Sentiment score
            raw_text: Raw text content
            metadata: Additional metadata
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO sentiment_data 
                (source, timestamp, symbol, sentiment, raw_text, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                source,
                datetime.now(),
                symbol,
                sentiment,
                raw_text,
                json.dumps(metadata) if metadata else None
            ))
            
            self.connection.commit()
            logger.info(f"Saved sentiment data from {source} for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving sentiment data: {e}")
            raise
            
    def get_sentiment_data(self, source: str, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get sentiment data.
        
        Args:
            source: Source of sentiment data
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame of sentiment data
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT timestamp, sentiment, raw_text, metadata
                FROM sentiment_data
                WHERE source = ? AND symbol = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (source, symbol, start_date, end_date))
            
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(rows, columns=['timestamp', 'sentiment', 'raw_text', 'metadata'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment data: {e}")
            return pd.DataFrame()
            
    def save_portfolio_position(self, symbol: str, quantity: float, price: float, position_type: str, metadata: Dict = None):
        """
        Save portfolio position.
        
        Args:
            symbol: Stock symbol
            quantity: Position quantity
            price: Price at which position was taken
            position_type: Type of position ('long', 'short', etc.)
            metadata: Additional metadata
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_positions 
                (timestamp, symbol, quantity, price, type, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                symbol,
                quantity,
                price,
                position_type,
                json.dumps(metadata) if metadata else None
            ))
            
            self.connection.commit()
            logger.info(f"Saved portfolio position for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving portfolio position: {e}")
            raise
            
    def get_portfolio_positions(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Get portfolio positions.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame of portfolio positions
        """
        try:
            cursor = self.connection.cursor()
            
            cursor.execute('''
                SELECT timestamp, symbol, quantity, price, type, metadata
                FROM portfolio_positions
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (start_date, end_date))
            
            rows = cursor.fetchall()
            if not rows:
                return pd.DataFrame()
                
            df = pd.DataFrame(rows, columns=['timestamp', 'symbol', 'quantity', 'price', 'type', 'metadata'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting portfolio positions: {e}")
            return pd.DataFrame()
            
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
