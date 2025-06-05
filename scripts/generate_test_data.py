#!/usr/bin/env python3
"""
Test Data Generator for AI Hedge Fund Dashboard

Generates sample data for testing the dashboard.
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Output directories
DATA_DIR = project_root / "data"
MODELS_DIR = project_root / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Sample data configuration
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "GS", "BRK-A"]
STRATEGIES = ["Warren Buffett", "Peter Lynch", "Ray Dalio", "George Soros", "Joel Greenblatt"]
MARKET_INDICES = ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"]

def generate_time_series(start_date, end_date, freq="D"):
    """Generate a time series between start_date and end_date."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def generate_stock_data(symbol, start_date, end_date):
    """Generate sample stock data for a symbol."""
    dates = generate_time_series(start_date, end_date)
    
    # Generate random walk for prices
    np.random.seed(hash(symbol) % 2**32)
    n = len(dates)
    returns = np.random.normal(0.0005, 0.02, n)
    prices = 100 * np.cumprod(1 + returns)
    
    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "open": prices * (1 + np.random.normal(0, 0.005, n)),
        "high": prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        "low": prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        "close": prices,
        "volume": np.random.lognormal(10, 1, n).astype(int),
        "symbol": symbol
    })
    
    return df

def generate_portfolio_data():
    """Generate sample portfolio data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Generate positions
    positions = []
    for symbol in SYMBOLS[:5]:  # Only use first 5 symbols for portfolio
        df = generate_stock_data(symbol, start_date, end_date)
        last_row = df.iloc[-1]
        
        position = {
            "symbol": symbol,
            "shares": random.randint(10, 1000),
            "avg_price": last_row["close"] * random.uniform(0.8, 1.2),
            "current_price": last_row["close"],
            "value": last_row["close"] * random.randint(10, 1000),
            "pnl": random.uniform(-1000, 5000),
            "pnl_pct": random.uniform(-10, 25),
            "sector": random.choice(["Technology", "Finance", "Healthcare", "Consumer", "Industrial"])
        }
        positions.append(position)
    
    # Calculate portfolio summary
    total_value = sum(p["value"] for p in positions)
    total_pnl = sum(p["pnl"] for p in positions)
    
    summary = {
        "total_value": total_value,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / (total_value - total_pnl)) * 100 if (total_value - total_pnl) > 0 else 0,
        "cash": random.uniform(10000, 50000),
        "last_updated": datetime.now().isoformat()
    }
    
    return {"summary": summary, "positions": positions}

def generate_strategy_data():
    """Generate sample strategy performance data."""
    strategies = []
    
    for i, name in enumerate(STRATEGIES):
        returns = np.random.normal(0.0005, 0.02, 100)
        cum_returns = np.cumprod(1 + returns) - 1
        
        strategy = {
            "id": f"strategy_{i+1}",
            "name": name,
            "description": f"{name}'s investment strategy",
            "sharpe_ratio": random.uniform(0.5, 2.5),
            "max_drawdown": -abs(random.uniform(0.05, 0.3)),
            "annual_return": random.uniform(0.05, 0.25),
            "volatility": random.uniform(0.1, 0.3),
            "returns": cum_returns.tolist(),
            "positions": [{"symbol": s, "weight": random.random()} for s in random.sample(SYMBOLS, 3)]
        }
        strategies.append(strategy)
    
    return strategies

def generate_risk_metrics():
    """Generate sample risk metrics."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Generate time series for VaR and other metrics
    dates = generate_time_series(start_date, end_date)
    
    metrics = {
        "var_95": random.uniform(0.01, 0.05),
        "cvar_95": random.uniform(0.015, 0.07),
        "sharpe_ratio": random.uniform(0.5, 2.5),
        "beta": random.uniform(0.7, 1.3),
        "volatility": random.uniform(0.1, 0.3),
        "max_drawdown": -abs(random.uniform(0.05, 0.2)),
        "metrics_over_time": [
            {
                "date": (end_date - timedelta(days=i)).strftime("%Y-%m-%d"),
                "var_95": random.uniform(0.01, 0.05),
                "cvar_95": random.uniform(0.015, 0.07),
                "volatility": random.uniform(0.1, 0.3)
            }
            for i in range(30, 0, -1)
        ]
    }
    
    return metrics

def save_data(data, filename):
    """Save data to JSON file."""
    filepath = DATA_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {filepath}")

def main():
    """Generate all test data."""
    print("Generating test data...")
    
    # Generate and save portfolio data
    portfolio_data = generate_portfolio_data()
    save_data(portfolio_data, "portfolio.json")
    
    # Generate and save strategy data
    strategy_data = generate_strategy_data()
    save_data(strategy_data, "strategies.json")
    
    # Generate and save risk metrics
    risk_metrics = generate_risk_metrics()
    save_data(risk_metrics, "risk_metrics.json")
    
    print("\nTest data generation complete!")

if __name__ == "__main__":
    main()
