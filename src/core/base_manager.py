"""
Base Manager - Abstract base class for all manager classes in the system.

Provides common functionality and interface that all managers should implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger


class BaseManager(ABC):
    """Abstract base class for all manager classes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base manager with optional configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
        logger.debug(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def initialize(self):
        """Initialize the manager's resources.
        
        This method should be implemented by subclasses to perform any necessary
        setup or resource allocation.
        """
        pass
    
    @abstractmethod
    async def stop(self):
        """Clean up resources used by the manager.
        
        This method should be implemented by subclasses to perform any necessary
        cleanup when the manager is no longer needed.
        """
        pass
    
    def is_healthy(self) -> bool:
        """Check if the manager is in a healthy state.
        
        Returns:
            bool: True if the manager is healthy, False otherwise
        """
        return self.is_initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the manager.
        
        Returns:
            Dict containing status information
        """
        return {
            "initialized": self.is_initialized,
            "healthy": self.is_healthy(),
            "class": self.__class__.__name__
        }
