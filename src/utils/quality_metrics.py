"""
Quality Metrics for Fundamental Analysis

Contains implementations of various quality metrics to assess company strength:
- Return on Equity (ROE)
- Debt to Equity Ratio
- Earnings Stability
- Profit Margin
- Operating Margin
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

class QualityScorer:
    """
    Calculates quality scores for companies based on fundamental metrics
    """
    
    def __init__(self):
        self.metrics = {
            'roe': self.calculate_roe,
            'debt_ratio': self.calculate_debt_ratio,
            'earnings_stability': self.calculate_earnings_stability,
            'profit_margin': self.calculate_profit_margin,
            'operating_margin': self.calculate_operating_margin
        }
        
    def calculate_roe(self, net_income: float, shareholders_equity: float) -> float:
        """
        Calculate Return on Equity (ROE)
        
        Args:
            net_income: Company's net income
            shareholders_equity: Shareholders' equity
            
        Returns:
            ROE percentage
        """
        try:
            return (net_income / shareholders_equity) * 100
        except ZeroDivisionError:
            return 0.0

    def calculate_debt_ratio(self, total_debt: float, total_assets: float) -> float:
        """
        Calculate Debt to Asset Ratio
        
        Args:
            total_debt: Total company debt
            total_assets: Total company assets
            
        Returns:
            Debt ratio percentage
        """
        try:
            return (total_debt / total_assets) * 100
        except ZeroDivisionError:
            return 0.0

    def calculate_earnings_stability(self, earnings_history: Dict[str, float]) -> float:
        """
        Calculate earnings stability score
        
        Args:
            earnings_history: Dictionary of earnings by year
            
        Returns:
            Stability score (0-100)
        """
        if not earnings_history:
            return 0.0
            
        earnings = list(earnings_history.values())
        if len(earnings) < 2:
            return 0.0
            
        std_dev = np.std(earnings)
        avg = np.mean(earnings)
        
        try:
            return (1 - (std_dev / avg)) * 100
        except ZeroDivisionError:
            return 0.0

    def calculate_profit_margin(self, net_income: float, revenue: float) -> float:
        """
        Calculate Net Profit Margin
        
        Args:
            net_income: Company's net income
            revenue: Company's revenue
            
        Returns:
            Profit margin percentage
        """
        try:
            return (net_income / revenue) * 100
        except ZeroDivisionError:
            return 0.0

    def calculate_operating_margin(self, operating_income: float, revenue: float) -> float:
        """
        Calculate Operating Margin
        
        Args:
            operating_income: Operating income
            revenue: Revenue
            
        Returns:
            Operating margin percentage
        """
        try:
            return (operating_income / revenue) * 100
        except ZeroDivisionError:
            return 0.0

    def score_company(self, financial_data: Dict[str, float]) -> float:
        """
        Calculate overall quality score for a company
        
        Args:
            financial_data: Dictionary containing financial metrics
            
        Returns:
            Overall quality score (0-100)
        """
        scores = []
        for metric, func in self.metrics.items():
            try:
                score = func(**{k: v for k, v in financial_data.items() if k in func.__code__.co_varnames})
                scores.append(score)
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                scores.append(0.0)
                
        if not scores:
            return 0.0
            
        return np.mean(scores)
