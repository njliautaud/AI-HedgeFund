#!/usr/bin/env python3
"""
Data Cache - Provides caching functionality for financial data

Implements a disk-based caching system for financial data to:
- Reduce API calls to external data providers
- Improve system performance by storing frequently accessed data
- Handle data staleness with configurable expiration policies
- Support both historical and real-time data caching needs
"""

import os
import json
import pickle
import hashlib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger


class DataCache:
    """
    Implements a disk-based caching system for financial data.
    
    Features:
    - Configurable cache location
    - Automatic cache expiration based on data type
    - Support for different data formats (JSON, Pickle, CSV)
    - Cache statistics and monitoring
    """
    
    def __init__(self, cache_dir: Optional[str] = None, max_age_days: Dict[str, int] = None):
        """
        Initialize the data cache.
        
        Args:
            cache_dir: Directory to store cache files (default: project_root/data/cache)
            max_age_days: Dictionary mapping data types to maximum age in days
                          e.g., {'stock_daily': 1, 'economic': 7, 'fundamentals': 30}
        """
        # Set default cache directory if not provided
        if cache_dir is None:
            # Get project root directory (assuming this file is in src/utils)
            project_root = Path(__file__).resolve().parents[2]
            cache_dir = os.path.join(project_root, 'data', 'cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        self.cache_dir = cache_dir
        self.stock_dir = os.path.join(cache_dir, 'stocks')
        self.economic_dir = os.path.join(cache_dir, 'economic')
        self.fundamental_dir = os.path.join(cache_dir, 'fundamentals')
        self.metadata_dir = os.path.join(cache_dir, 'metadata')
        
        # Create subdirectories
        for directory in [self.stock_dir, self.economic_dir, self.fundamental_dir, self.metadata_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set default max age for different data types if not provided
        self.max_age_days = max_age_days or {
            'stock_daily': 1,      # Daily stock data expires after 1 day
            'stock_intraday': 0,   # Intraday data always refreshed (0 = always refresh)
            'economic': 7,         # Economic data expires after 7 days
            'fundamentals': 30,    # Fundamental data expires after 30 days
            'metadata': 90         # Metadata expires after 90 days
        }
        
        # Initialize cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'expired': 0,
            'saved': 0
        }
        
        logger.info(f"Data cache initialized at {cache_dir}")
    
    def _get_cache_path(self, key: str, data_type: str) -> str:
        """
        Get the file path for a cache item.
        
        Args:
            key: Cache key (e.g., ticker symbol, economic indicator)
            data_type: Type of data ('stock_daily', 'economic', etc.)
            
        Returns:
            Path to the cache file
        """
        # Create a hash of the key to use as filename
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        
        # Determine the appropriate directory based on data type
        if data_type.startswith('stock'):
            directory = self.stock_dir
        elif data_type.startswith('economic'):
            directory = self.economic_dir
        elif data_type.startswith('fundamental'):
            directory = self.fundamental_dir
        elif data_type.startswith('metadata'):
            directory = self.metadata_dir
        else:
            # Default to main cache directory
            directory = self.cache_dir
        
        # Create a subdirectory for the specific data type if needed
        subdir = os.path.join(directory, data_type)
        os.makedirs(subdir, exist_ok=True)
        
        # Return the full path
        return os.path.join(subdir, f"{hashed_key}.pkl")
    
    def _is_expired(self, filepath: str, data_type: str) -> bool:
        """
        Check if a cache item is expired.
        
        Args:
            filepath: Path to the cache file
            data_type: Type of data
            
        Returns:
            True if the cache item is expired, False otherwise
        """
        # If the file doesn't exist, it's considered expired
        if not os.path.exists(filepath):
            return True
        
        # Get the maximum age for this data type
        max_age = self.max_age_days.get(data_type, 1)  # Default to 1 day if not specified
        
        # If max_age is 0, always consider it expired (always refresh)
        if max_age == 0:
            return True
        
        # Get the file's last modification time
        mtime = os.path.getmtime(filepath)
        mtime_dt = datetime.fromtimestamp(mtime)
        
        # Check if the file is older than the maximum age
        age_limit = datetime.now() - timedelta(days=max_age)
        
        return mtime_dt < age_limit
    
    def get(self, key: str, data_type: str) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        
        Args:
            key: Cache key
            data_type: Type of data
            
        Returns:
            The cached data if found and not expired, None otherwise
        """
        filepath = self._get_cache_path(key, data_type)
        
        # Check if the cache item is expired
        if self._is_expired(filepath, data_type):
            self.stats['expired'] += 1
            return None
        
        # Try to load the cache item
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for {key} ({data_type})")
            return data
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            self.stats['misses'] += 1
            logger.debug(f"Cache miss for {key} ({data_type}): {str(e)}")
            return None
    
    def set(self, key: str, data: Any, data_type: str) -> bool:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            data_type: Type of data
            
        Returns:
            True if successful, False otherwise
        """
        filepath = self._get_cache_path(key, data_type)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.stats['saved'] += 1
            logger.debug(f"Cached {key} ({data_type})")
            return True
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Failed to cache {key} ({data_type}): {str(e)}")
            return False
    
    def invalidate(self, key: str, data_type: str) -> bool:
        """
        Invalidate (delete) a cache item.
        
        Args:
            key: Cache key
            data_type: Type of data
            
        Returns:
            True if successful, False otherwise
        """
        filepath = self._get_cache_path(key, data_type)
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Invalidated cache for {key} ({data_type})")
            return True
        except IOError as e:
            logger.error(f"Failed to invalidate cache for {key} ({data_type}): {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear each data type directory
            for directory in [self.stock_dir, self.economic_dir, self.fundamental_dir, self.metadata_dir]:
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.pkl'):
                            os.remove(os.path.join(root, file))
            
            logger.info("Cleared all cache data")
            return True
        except IOError as e:
            logger.error(f"Failed to clear cache: {str(e)}")
            return False
    
    def clear_type(self, data_type: str) -> bool:
        """
        Clear all cached data of a specific type.
        
        Args:
            data_type: Type of data to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine the appropriate directory based on data type
            if data_type.startswith('stock'):
                directory = self.stock_dir
            elif data_type.startswith('economic'):
                directory = self.economic_dir
            elif data_type.startswith('fundamental'):
                directory = self.fundamental_dir
            elif data_type.startswith('metadata'):
                directory = self.metadata_dir
            else:
                # Default to main cache directory
                directory = self.cache_dir
            
            # Create a subdirectory path for the specific data type
            subdir = os.path.join(directory, data_type)
            
            # Check if the directory exists
            if os.path.exists(subdir):
                for file in os.listdir(subdir):
                    if file.endswith('.pkl'):
                        os.remove(os.path.join(subdir, file))
                
                logger.info(f"Cleared cache for data type: {data_type}")
            return True
        except IOError as e:
            logger.error(f"Failed to clear cache for data type {data_type}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.stats
    
    def get_size(self) -> Dict[str, float]:
        """
        Get the size of the cache in megabytes.
        
        Returns:
            Dictionary with cache size information
        """
        sizes = {}
        
        # Calculate size for each data type directory
        for data_type, directory in [
            ('stocks', self.stock_dir),
            ('economic', self.economic_dir),
            ('fundamentals', self.fundamental_dir),
            ('metadata', self.metadata_dir)
        ]:
            total_size = 0
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.pkl'):
                            total_size += os.path.getsize(os.path.join(root, file))
            
            # Convert to megabytes
            sizes[data_type] = total_size / (1024 * 1024)
        
        # Add total size
        sizes['total'] = sum(sizes.values())
        
        return sizes
    
    def get_dataframe(self, key: str, data_type: str) -> Optional[pd.DataFrame]:
        """
        Retrieve a DataFrame from the cache.
        
        This is a convenience method for retrieving pandas DataFrames,
        which are commonly used for financial data.
        
        Args:
            key: Cache key
            data_type: Type of data
            
        Returns:
            The cached DataFrame if found and not expired, None otherwise
        """
        data = self.get(key, data_type)
        
        if data is not None and isinstance(data, pd.DataFrame):
            return data
        return None
    
    def set_dataframe(self, key: str, df: pd.DataFrame, data_type: str) -> bool:
        """
        Store a DataFrame in the cache.
        
        Args:
            key: Cache key
            df: DataFrame to cache
            data_type: Type of data
            
        Returns:
            True if successful, False otherwise
        """
        return self.set(key, df, data_type)
    
    def get_json(self, key: str, data_type: str) -> Optional[Dict]:
        """
        Retrieve JSON data from the cache.
        
        Args:
            key: Cache key
            data_type: Type of data
            
        Returns:
            The cached JSON data if found and not expired, None otherwise
        """
        data = self.get(key, data_type)
        
        if data is not None and isinstance(data, dict):
            return data
        return None
    
    def set_json(self, key: str, data: Dict, data_type: str) -> bool:
        """
        Store JSON data in the cache.
        
        Args:
            key: Cache key
            data: JSON data to cache
            data_type: Type of data
            
        Returns:
            True if successful, False otherwise
        """
        return self.set(key, data, data_type)