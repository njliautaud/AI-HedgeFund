"""
Backtesting Engine for AI Hedge Fund System

This module provides the BacktestEngine class for running backtests on trading strategies
using historical market data.
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from loguru import logger

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies on historical data.
    
    This class provides functionality to run backtests, calculate performance metrics,
    and generate reports on strategy performance.
    """
    
    def __init__(self, config, strategy_manager, data_manager):
        """
        Initialize the backtesting engine.
        
        Args:
            config: System configuration
            strategy_manager: Strategy manager instance
            data_manager: Data manager instance for accessing historical data
        """
        self.config = config
        self.strategy_manager = strategy_manager
        self.data_manager = data_manager
        self.results = {}
        
        # Configure backtest parameters from config
        self.backtest_config = self.config.get('backtesting', {})
        self.initial_capital = self.backtest_config.get('initial_capital', 100000.0)
        self.commission = self.backtest_config.get('commission', 0.001)  # 0.1% commission
        self.slippage = self.backtest_config.get('slippage', 0.0005)  # 0.05% slippage
        
        logger.info(f"BacktestEngine initialized with initial capital: ${self.initial_capital:,.2f}")
    
    async def run_backtest(
        self, 
        start_date: str, 
        end_date: str, 
        initial_capital: Optional[float] = None,
        symbols: Optional[List[str]] = None,
        strategy_filter: Optional[str] = None
    ) -> Dict:
        """
        Run a backtest for the specified date range and symbols.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            initial_capital: Initial capital for the backtest (overrides config)
            symbols: List of symbols to include in the backtest (None for all available)
            
        Returns:
            Dict containing backtest results and performance metrics
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
            
        logger.info(f"Starting backtest from {start_date} to {end_date} with ${self.initial_capital:,.2f}")
        
        # Convert string dates to datetime objects
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        
        # Get historical data for the specified date range and symbols
        if symbols is None:
            # Use all available symbols if none specified
            symbols = self.data_manager.get_available_symbols()
            
        logger.info(f"Fetching historical data for {len(symbols)} symbols")
        
        try:
            # Get historical price data
            historical_data = {}
            for symbol in symbols:
                try:
                    data = await self.data_manager.get_historical_data(
                        symbol, 
                        start_date=start_date,
                        end_date=end_date,
                        interval='1d'
                    )
                    if not data.empty:
                        historical_data[symbol] = data
                        logger.debug(f"Fetched {len(data)} rows for {symbol}")
                    else:
                        logger.warning(f"No data available for {symbol} in the specified date range")
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
            
            if not historical_data:
                raise ValueError("No historical data available for the specified symbols and date range")
            
            # Combine data into a single DataFrame with multi-index (symbol, date)
            combined_data = pd.concat(historical_data, names=['symbol', 'date'])
            
            # Sort the index for proper time-based operations
            combined_data = combined_data.sort_index()
            
            # Initialize portfolio state
            portfolio = {
                'cash': self.initial_capital,
                'positions': {symbol: 0 for symbol in historical_data.keys()},
                'value': self.initial_capital,
                'trades': [],
                'equity_curve': []
            }
            
            # Run the backtest by iterating through each day
            logger.info("Running backtest simulation...")
            
            # Get unique dates from the combined data
            dates = combined_data.index.get_level_values('date').unique().sort_values()
            
            for date in dates:
                # Get data for the current date
                date_str = date.strftime('%Y-%m-%d')
                daily_data = combined_data.xs(date, level='date')
                
                # Update portfolio value based on current prices
                self._update_portfolio_value(portfolio, daily_data, date_str)
                
                # Generate trading signals for the next day
                signals = await self.strategy_manager.generate_signals(daily_data)
                
                # Execute trades based on signals
                if signals:
                    self._execute_trades(portfolio, signals, daily_data, date_str)
                
                # Record daily portfolio value
                portfolio['equity_curve'].append({
                    'date': date_str,
                    'value': portfolio['value'],
                    'cash': portfolio['cash'],
                    'positions': portfolio['positions'].copy()
                })
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(portfolio, start_date, end_date)
            
            # Store results
            self.results = {
                'portfolio': portfolio,
                'performance': performance,
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': self.initial_capital,
                'final_value': portfolio['value']
            }
            
            logger.info(f"Backtest completed. Final portfolio value: ${portfolio['value']:,.2f}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error during backtest: {e}")
            raise
    
    def _update_portfolio_value(self, portfolio, daily_data, date_str):
        """
        Update the portfolio value based on current market prices.
        
        Args:
            portfolio: Dictionary containing portfolio state
            daily_data: DataFrame with price data for the current date
            date_str: Current date as a string
        ""
        # Calculate the value of all positions
        positions_value = 0
        for symbol, position in portfolio['positions'].items():
            if symbol in daily_data.index and position != 0:
                # Use the closing price to value the position
                price = daily_data.loc[symbol, 'close']
                positions_value += position * price
        
        # Update total portfolio value
        portfolio['value'] = portfolio['cash'] + positions_value
    
    def _execute_trades(self, portfolio, signals, daily_data, date_str):
        """
        Execute trades based on signals and update portfolio.
        
        Args:
            portfolio: Dictionary containing portfolio state
            signals: Dictionary of trading signals from strategies
            daily_data: DataFrame with price data for the current date
            date_str: Current date as a string
        """
        # Sort signals by priority if needed
        # For now, just process all signals
        
        for symbol, signal in signals.items():
            if symbol not in daily_data.index:
                logger.warning(f"No price data for {symbol} on {date_str}, skipping trade")
                continue
                
            current_price = daily_data.loc[symbol, 'close']
            current_position = portfolio['positions'].get(symbol, 0)
            
            # Apply slippage to the execution price
            execution_price = current_price * (1 + self.slippage if signal['action'] == 'buy' else 1 - self.slippage)
            
            # Calculate position size based on the signal
            # For simplicity, we'll use a fixed position size for now
            position_size = signal.get('size', 1)
            
            if signal['action'] == 'buy' and portfolio['cash'] > 0:
                # Calculate the maximum number of shares we can buy with available cash
                max_shares = portfolio['cash'] // (execution_price * (1 + self.commission))
                shares_to_buy = min(position_size, max_shares)
                
                if shares_to_buy > 0:
                    cost = shares_to_buy * execution_price * (1 + self.commission)
                    portfolio['cash'] -= cost
                    portfolio['positions'][symbol] = portfolio['positions'].get(symbol, 0) + shares_to_buy
                    
                    # Record the trade
                    portfolio['trades'].append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'buy',
                        'shares': shares_to_buy,
                        'price': execution_price,
                        'cost': cost,
                        'commission': cost * self.commission
                    })
                    
                    logger.debug(f"Bought {shares_to_buy} shares of {symbol} at ${execution_price:.2f}")
            
            elif signal['action'] == 'sell' and current_position > 0:
                shares_to_sell = min(position_size, current_position)
                
                if shares_to_sell > 0:
                    # Calculate proceeds after commission
                    proceeds = shares_to_sell * execution_price * (1 - self.commission)
                    portfolio['cash'] += proceeds
                    portfolio['positions'][symbol] -= shares_to_sell
                    
                    # Record the trade
                    portfolio['trades'].append({
                        'date': date_str,
                        'symbol': symbol,
                        'action': 'sell',
                        'shares': shares_to_sell,
                        'price': execution_price,
                        'proceeds': proceeds,
                        'commission': shares_to_sell * execution_price * self.commission
                    })
                    
                    logger.debug(f"Sold {shares_to_sell} shares of {symbol} at ${execution_price:.2f}")
    
    def _calculate_performance_metrics(self, portfolio, start_date, end_date):
        """
        Calculate performance metrics for the backtest.
        
        Args:
            portfolio: Dictionary containing portfolio state
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary containing performance metrics
        """
        if not portfolio['equity_curve']:
            return {}
        
        # Extract equity curve data
        equity_curve = pd.DataFrame(portfolio['equity_curve'])
        equity_curve['date'] = pd.to_datetime(equity_curve['date'])
        equity_curve.set_index('date', inplace=True)
        
        # Calculate returns
        equity_curve['daily_return'] = equity_curve['value'].pct_change().fillna(0)
        
        # Calculate metrics
        total_return = (portfolio['value'] / self.initial_capital - 1) * 100  # in %
        
        # Annualized return (assuming 252 trading days per year)
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        years = max(days / 252, 1)  # Avoid division by zero
        annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Calculate volatility (annualized)
        volatility = equity_curve['daily_return'].std() * np.sqrt(252) * 100  # in %
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0% for simplicity)
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = (1 + equity_curve['daily_return']).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100  # in %
        
        # Count trades
        num_trades = len(portfolio['trades'])
        winning_trades = len([t for t in portfolio['trades'] if 'proceeds' in t and t['proceeds'] > t.get('cost', 0)])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'num_trades': num_trades,
            'win_rate_pct': win_rate,
            'final_value': portfolio['value'],
            'initial_capital': self.initial_capital
        }
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report of the backtest results.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No backtest results available. Run a backtest first."
        
        perf = self.results['performance']
        
        report = [
            "\n" + "=" * 50,
            "BACKTEST REPORT",
            "=" * 50,
            f"Period: {self.results['start_date']} to {self.results['end_date']}",
            f"Initial Capital: ${self.initial_capital:,.2f}",
            f"Final Value: ${self.results['final_value']:,.2f}",
            "-" * 50,
            f"Total Return: {perf['total_return_pct']:.2f}%",
            f"Annualized Return: {perf['annualized_return_pct']:.2f}%",
            f"Volatility: {perf['volatility_pct']:.2f}%",
            f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}",
            f"Max Drawdown: {perf['max_drawdown_pct']:.2f}%",
            f"Number of Trades: {perf['num_trades']}",
            f"Win Rate: {perf['win_rate_pct']:.1f}%",
            "=" * 50 + "\n"
        ]
        
        return "\n".join(report)
    
    def plot_equity_curve(self, save_path: str = None):
        """
        Plot the equity curve from the backtest.
        
        Args:
            save_path: Path to save the plot (optional)
            
        Returns:
            Matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            if not self.results or 'portfolio' not in self.results or not self.results['portfolio']['equity_curve']:
                logger.error("No equity curve data available to plot")
                return None
                
            # Extract equity curve data
            equity_curve = pd.DataFrame(self.results['portfolio']['equity_curve'])
            equity_curve['date'] = pd.to_datetime(equity_curve['date'])
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve['date'], equity_curve['value'], label='Portfolio Value')
            
            # Add benchmark if available
            # For simplicity, we'll skip this for now
            
            # Format the plot
            plt.title('Portfolio Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # Format x-axis to show dates nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate()
            
            # Save the plot if a path is provided
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                logger.info(f"Equity curve plot saved to {save_path}")
            
            return plt.gcf()
            
        except ImportError:
            logger.warning("Matplotlib not available. Install with 'pip install matplotlib' to enable plotting.")
            return None
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            return None
