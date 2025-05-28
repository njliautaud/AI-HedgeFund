#!/usr/bin/env python3
"""
Cache Manager - Handles data caching for the AI Hedge Fund system

Provides efficient caching mechanisms for:
- Market data
- Economic indicators
- Fundamental data
- Technical indicators
- Sentiment data

Implements:
- Time-based cache expiration
- Disk persistence
- Memory caching for frequently accessed data
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import pandas as pd
from loguru import logger


class CacheManager:
    """
    Manages caching of financial data to reduce API calls and improve performance.
    
    Features:
    - In-memory caching for fast access
    - Disk persistence for data between sessions
    - Configurable expiration policies
    - Automatic staleness detection
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the cache manager.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # Cache settings
        cache_config = config.get('cache', {})
        self.cache_dir = os.path.join('data', 'cache')
        self.memory_cache_enabled = cache_config.get('memory_cache_enabled', True)
        self.disk_cache_enabled = cache_config.get('disk_cache_enabled', True)
        self.default_expiry = cache_config.get('default_expiry_seconds', 3600)  # 1 hour
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize memory cache
        self.memory_cache = {}
        
        logger.info("Cache manager initialized")
    
    async def get(self, key: str, category: str = 'market_data') -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            category: Data category (market_data, economic_data, etc.)
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        # Check memory cache first
        if self.memory_cache_enabled:
            cache_entry = self.memory_cache.get(f"{category}:{key}")
            if cache_entry and not self._is_expired(cache_entry.get('timestamp'), category):
                logger.debug(f"Memory cache hit for {category}:{key}")
                return cache_entry.get('data')
        
        # Check disk cache if memory cache missed
        if self.disk_cache_enabled:
            disk_data = await self._load_from_disk(key, category)
            if disk_data and not self._is_expired(disk_data.get('timestamp'), category):
                # Update memory cache
                if self.memory_cache_enabled:
                    self.memory_cache[f"{category}:{key}"] = disk_data
                logger.debug(f"Disk cache hit for {category}:{key}")
                return disk_data.get('data')
        
        logger.debug(f"Cache miss for {category}:{key}")
        return None
    
    async def set(self, key: str, data: Any, category: str = 'market_data', expiry: Optional[int] = None) -> None:
        """
        Store an item in the cache.
        
        Args:
            key: Cache key
            data: Data to cache
            category: Data category
            expiry: Custom expiry time in seconds (overrides default)
        """
        cache_entry = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'expiry': expiry or self.default_expiry
        }
        
        # Update memory cache
        if self.memory_cache_enabled:
            self.memory_cache[f"{category}:{key}"] = cache_entry
        
        # Update disk cache
        if self.disk_cache_enabled:
            await self._save_to_disk(key, cache_entry, category)
        
        logger.debug(f"Cached {category}:{key}")
    
    async def invalidate(self, key: str, category: str = 'market_data') -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key
            category: Data category
        """
        # Remove from memory cache
        if self.memory_cache_enabled:
            self.memory_cache.pop(f"{category}:{key}", None)
        
        # Remove from disk cache
        if self.disk_cache_enabled:
            cache_path = os.path.join(self.cache_dir, category, f"{key}.json")
            if os.path.exists(cache_path):
                os.remove(cache_path)
        
        logger.debug(f"Invalidated cache for {category}:{key}")
    
    async def clear_category(self, category: str) -> None:
        """
        Clear all cache entries for a category.
        
        Args:
            category: Data category to clear
        """
        # Clear from memory cache
        if self.memory_cache_enabled:
            keys_to_remove = [k for k in self.memory_cache.keys() if k.startswith(f"{category}:")]
            for k in keys_to_remove:
                self.memory_cache.pop(k, None)
        
        # Clear from disk cache
        if self.disk_cache_enabled:
            category_dir = os.path.join(self.cache_dir, category)
            if os.path.exists(category_dir):
                for filename in os.listdir(category_dir):
                    os.remove(os.path.join(category_dir, filename))
        
        logger.info(f"Cleared cache for category {category}")
    
    async def clear_all(self) -> None:
        """
        Clear all cache entries.
        """
        # Clear memory cache
        if self.memory_cache_enabled:
            self.memory_cache.clear()
        
        # Clear disk cache
        if self.disk_cache_enabled:
            for category in os.listdir(self.cache_dir):
                category_path = os.path.join(self.cache_dir, category)
                if os.path.isdir(category_path):
                    for filename in os.listdir(category_path):
                        os.remove(os.path.join(category_path, filename))
        
        logger.info("Cleared all cache entries")
    
    def _is_expired(self, timestamp_str: str, category: str) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            timestamp_str: ISO format timestamp string
            category: Data category
            
        Returns:
            True if expired, False otherwise
        """
        if not timestamp_str:
            return True
        
        timestamp = datetime.fromisoformat(timestamp_str)
        category_expiry = self.config.get('cache', {}).get(f'{category}_expiry_seconds', self.default_expiry)
        
        return (datetime.now() - timestamp).total_seconds() > category_expiry
    
    async def _save_to_disk(self, key: str, cache_entry: Dict, category: str) -> None:
        """
        Save cache entry to disk.
        
        Args:
            key: Cache key
            cache_entry: Cache entry dictionary
            category: Data category
        """
        category_dir = os.path.join(self.cache_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        cache_path = os.path.join(category_dir, f"{key}.json")
        
        # Handle pandas DataFrames specially
        data = cache_entry['data']
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to dict for JSON serialization
            serialized_entry = cache_entry.copy()
            serialized_entry['data'] = data.to_dict()
            serialized_entry['_is_dataframe'] = True
            
            with open(cache_path, 'w') as f:
                json.dump(serialized_entry, f)
        else:
            try:
                # Try standard JSON serialization
                with open(cache_path, 'w') as f:
                    json.dump(cache_entry, f)
            except (TypeError, OverflowError):
                # Fall back to pickle for complex objects
                pickle_path = os.path.join(category_dir, f"{key}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(cache_entry, f)
    
    async def _load_from_disk(self, key: str, category: str) -> Optional[Dict]:
        """
        Load cache entry from disk.
        
        Args:
            key: Cache key
            category: Data category
            
        Returns:
            Cache entry dictionary if found, None otherwise
        """
        category_dir = os.path.join(self.cache_dir, category)
        if not os.path.exists(category_dir):
            return None
        
        # Try JSON first
        json_path = os.path.join(category_dir, f"{key}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Reconstruct DataFrame if needed
                if cache_entry.get('_is_dataframe', False):
                    cache_entry['data'] = pd.DataFrame.from_dict(cache_entry['data'])
                    del cache_entry['_is_dataframe']
                
                return cache_entry
            except json.JSONDecodeError:
                logger.warning(f"Corrupted JSON cache file: {json_path}")
                return None
        
        # Try pickle as fallback
        pickle_path = os.path.join(category_dir, f"{key}.pkl")
        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                logger.warning(f"Corrupted pickle cache file: {pickle_path}")
                return None
        
        return None