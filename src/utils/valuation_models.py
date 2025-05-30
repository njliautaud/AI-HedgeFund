"""
Valuation Models for Financial Analysis

Contains implementations of various valuation models including:
- Discounted Cash Flow (DCF) Model
- Graham Number
"""

import math
from typing import Dict, Optional, Tuple
from datetime import datetime
import numpy as np


class DCFModel:
    """
    Discounted Cash Flow (DCF) Model for valuing companies
    """
    def __init__(self, 
                 discount_rate: float = 0.10,
                 terminal_growth_rate: float = 0.02,
                 projection_years: int = 10):
        """
        Initialize DCF Model parameters
        
        Args:
            discount_rate: Required rate of return
            terminal_growth_rate: Long-term growth rate
            projection_years: Number of years to project cash flows
        """
        self.discount_rate = discount_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.projection_years = projection_years

    def calculate_dcf(self, 
                     cash_flows: Dict[int, float],
                     terminal_value: Optional[float] = None) -> float:
        """
        Calculate the present value using DCF
        
        Args:
            cash_flows: Dictionary of projected cash flows {year: amount}
            terminal_value: Terminal value of the company
        
        Returns:
            Present value of the company
        """
        present_value = 0
        for year, cash_flow in cash_flows.items():
            present_value += cash_flow / ((1 + self.discount_rate) ** (year - min(cash_flows.keys())))
        
        if terminal_value:
            final_year = max(cash_flows.keys())
            present_value += terminal_value / ((1 + self.discount_rate) ** (final_year - min(cash_flows.keys())))
        
        return present_value

class GrahamNumber:
    """
    Graham Number calculation for intrinsic value
    """
    def __init__(self):
        pass

    def calculate(self, eps: float, book_value: float) -> float:
        """
        Calculate Graham Number
        
        Args:
            eps: Earnings per share
            book_value: Book value per share
            
        Returns:
            Graham Number (intrinsic value estimate)
        """
        try:
            return round(math.sqrt(22.5 * eps * book_value), 2)
        except (ValueError, TypeError):
            return 0.0

    def is_undervalued(self, current_price: float, eps: float, book_value: float) -> Tuple[bool, float]:
        """
        Check if stock is undervalued based on Graham Number
        
        Args:
            current_price: Current stock price
            eps: Earnings per share
            book_value: Book value per share
            
        Returns:
            Tuple of (is_undervalued, margin_of_safety)
        """
        graham_number = self.calculate(eps, book_value)
        margin_of_safety = (graham_number - current_price) / graham_number if graham_number > 0 else 0
        return current_price < graham_number, margin_of_safety
