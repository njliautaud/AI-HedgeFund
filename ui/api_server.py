#!/usr/bin/env python3
"""
AI Hedge Fund - API Server

A RESTful API server for the AI Hedge Fund system that allows external applications to:
- Query portfolio status and performance
- Retrieve strategy signals and performance metrics
- Control the trading system (start/stop)
- Access risk metrics and market data

Built with Flask for a lightweight and efficient API interface.
"""

import asyncio
import threading
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class APIServer:
    """
    API server for the AI Hedge Fund system.
    Provides a RESTful interface for external integrations.
    """
    
    def __init__(self, config, portfolio_manager, strategy_manager, risk_manager, data_manager):
        """
        Initialize the API server with system components.
        
        Args:
            config: System configuration
            portfolio_manager: Portfolio management component
            strategy_manager: Strategy management component
            risk_manager: Risk management component
            data_manager: Data management component
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self.data_manager = data_manager
        
        self.api_port = self.config.get('ui', {}).get('api_port', 8081)
        self.api_thread = None
        self.running = False
        self.app = None
        self.server = None
        
    async def start(self):
        """
        Start the API server in a separate thread.
        """
        if self.running:
            return
            
        self.running = True
        self.api_thread = threading.Thread(target=self._run_api_server)
        self.api_thread.daemon = True
        self.api_thread.start()
        
    async def stop(self):
        """
        Stop the API server.
        """
        self.running = False
        # The thread will terminate when the main process exits
        
    def _run_api_server(self):
        """
        Run the Flask API server.
        """
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Run the Flask app
        self.app.run(host='0.0.0.0', port=self.api_port, debug=False, use_reloader=False)
        
    def _setup_routes(self):
        """
        Set up the API routes.
        """
        app = self.app
        
        # API documentation
        @app.route('/')
        def api_docs():
            return jsonify({
                'name': 'AI Hedge Fund API',
                'version': '1.0.0',
                'description': 'RESTful API for the AI Hedge Fund system',
                'endpoints': [
                    {'path': '/system/status', 'method': 'GET', 'description': 'Get system status'},
                    {'path': '/system/start', 'method': 'POST', 'description': 'Start the system'},
                    {'path': '/system/stop', 'method': 'POST', 'description': 'Stop the system'},
                    {'path': '/portfolio/summary', 'method': 'GET', 'description': 'Get portfolio summary'},
                    {'path': '/portfolio/holdings', 'method': 'GET', 'description': 'Get portfolio holdings'},
                    {'path': '/portfolio/performance', 'method': 'GET', 'description': 'Get portfolio performance'},
                    {'path': '/strategies', 'method': 'GET', 'description': 'Get all strategies'},
                    {'path': '/strategies/<strategy_id>', 'method': 'GET', 'description': 'Get strategy details'},
                    {'path': '/strategies/<strategy_id>/performance', 'method': 'GET', 'description': 'Get strategy performance'},
                    {'path': '/strategies/<strategy_id>/signals', 'method': 'GET', 'description': 'Get strategy signals'},
                    {'path': '/market/data/<symbol>', 'method': 'GET', 'description': 'Get market data for a symbol'},
                    {'path': '/market/economic/<indicator>', 'method': 'GET', 'description': 'Get economic indicator data'},
                    {'path': '/risk/metrics', 'method': 'GET', 'description': 'Get risk metrics'},
                    {'path': '/risk/alerts', 'method': 'GET', 'description': 'Get risk alerts'}
                ]
            })
        
        # System endpoints
        @app.route('/system/status')
        def system_status():
            return jsonify({
                'status': 'running' if self.running else 'stopped',
                'uptime': '3h 24m',  # Mock data
                'last_update': datetime.now().isoformat(),
                'mode': 'paper_trading',  # Mock data
                'components': {
                    'data_manager': 'running',
                    'strategy_manager': 'running',
                    'portfolio_manager': 'running',
                    'risk_manager': 'running'
                }
            })
        
        @app.route('/system/start', methods=['POST'])
        def system_start():
            # In a real implementation, this would trigger the system to start
            return jsonify({
                'status': 'success',
                'message': 'System started successfully',
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/system/stop', methods=['POST'])
        def system_stop():
            # In a real implementation, this would trigger the system to stop
            return jsonify({
                'status': 'success',
                'message': 'System stopped successfully',
                'timestamp': datetime.now().isoformat()
            })
        
        # Portfolio endpoints
        @app.route('/portfolio/summary')
        def portfolio_summary():
            # In a real implementation, this would get data from the portfolio manager
            return jsonify({
                'total_value': 1250430,
                'cash': 125043,
                'invested': 1125387,
                'daily_pnl': 12504,
                'daily_pnl_percent': 1.01,
                'monthly_return': 4.2,
                'ytd_return': 12.5,
                'sharpe_ratio': 1.85,
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/portfolio/holdings')
        def portfolio_holdings():
            # In a real implementation, this would get data from the portfolio manager
            holdings = [
                {'symbol': 'AAPL', 'name': 'Apple Inc.', 'quantity': 150, 'avg_price': 150.25, 'current_price': 155.75, 'market_value': 23362.50, 'gain_loss_percent': 3.66},
                {'symbol': 'MSFT', 'name': 'Microsoft Corp.', 'quantity': 100, 'avg_price': 280.10, 'current_price': 290.30, 'market_value': 29030.00, 'gain_loss_percent': 3.64},
                {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'quantity': 50, 'avg_price': 2750.80, 'current_price': 2800.10, 'market_value': 140005.00, 'gain_loss_percent': 1.79},
                {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'quantity': 30, 'avg_price': 3200.50, 'current_price': 3250.20, 'market_value': 97506.00, 'gain_loss_percent': 1.55},
                {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'quantity': 75, 'avg_price': 190.75, 'current_price': 185.60, 'market_value': 13920.00, 'gain_loss_percent': -2.70}
            ]
            return jsonify({
                'holdings': holdings,
                'count': len(holdings),
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/portfolio/performance')
        def portfolio_performance():
            # In a real implementation, this would get data from the portfolio manager
            # Generate sample performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
            performance = []
            for i, date in enumerate(dates):
                performance.append({
                    'date': date.isoformat(),
                    'portfolio_value': 1000000 * (1 + 0.0003 * i) + 50000 * np.sin(0.05 * i),
                    'benchmark': 1000000 * (1 + 0.0002 * i)
                })
            
            return jsonify({
                'performance': performance,
                'period': '6m',
                'last_update': datetime.now().isoformat()
            })
        
        # Strategy endpoints
        @app.route('/strategies')
        def get_strategies():
            # In a real implementation, this would get data from the strategy manager
            strategies = [
                {'id': 'warren_buffett', 'name': 'Warren Buffett', 'type': 'value', 'enabled': True, 'weight': 25},
                {'id': 'peter_lynch', 'name': 'Peter Lynch', 'type': 'growth', 'enabled': True, 'weight': 20},
                {'id': 'benjamin_graham', 'name': 'Benjamin Graham', 'type': 'value', 'enabled': True, 'weight': 15},
                {'id': 'ray_dalio', 'name': 'Ray Dalio', 'type': 'macro', 'enabled': True, 'weight': 15},
                {'id': 'george_soros', 'name': 'George Soros', 'type': 'global_macro', 'enabled': False, 'weight': 0},
                {'id': 'joel_greenblatt', 'name': 'Joel Greenblatt', 'type': 'value', 'enabled': True, 'weight': 10},
                {'id': 'fed_policy', 'name': 'Fed Policy Tracker', 'type': 'economic', 'enabled': True, 'weight': 10},
                {'id': 'inflation', 'name': 'Inflation Protection', 'type': 'economic', 'enabled': True, 'weight': 5}
            ]
            return jsonify({
                'strategies': strategies,
                'count': len(strategies),
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/strategies/<strategy_id>')
        def get_strategy(strategy_id):
            # In a real implementation, this would get data from the strategy manager
            strategy_details = {
                'id': strategy_id,
                'name': 'Warren Buffett' if strategy_id == 'warren_buffett' else strategy_id.replace('_', ' ').title(),
                'description': 'Value investing strategy based on Warren Buffett\'s principles',
                'type': 'value',
                'enabled': True,
                'weight': 25,
                'metrics': {
                    'win_rate': 68,
                    'avg_return': 1.8,
                    'sharpe_ratio': 1.92,
                    'active_signals': 7
                },
                'parameters': {
                    'pe_ratio_max': 15,
                    'debt_to_equity_max': 0.5,
                    'roe_min': 15,
                    'moat_score_min': 7
                },
                'last_update': datetime.now().isoformat()
            }
            return jsonify(strategy_details)
        
        @app.route('/strategies/<strategy_id>/performance')
        def strategy_performance(strategy_id):
            # In a real implementation, this would get data from the strategy manager
            # Generate sample performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
            performance = []
            for i, date in enumerate(dates):
                performance.append({
                    'date': date.isoformat(),
                    'strategy_value': 1000000 * (1 + 0.0004 * i) + 30000 * np.sin(0.06 * i),
                    'benchmark': 1000000 * (1 + 0.0002 * i)
                })
            
            return jsonify({
                'strategy_id': strategy_id,
                'performance': performance,
                'period': '6m',
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/strategies/<strategy_id>/signals')
        def strategy_signals(strategy_id):
            # In a real implementation, this would get data from the strategy manager
            signals = [
                {'date': (datetime.now() - timedelta(days=1)).isoformat(), 'symbol': 'AAPL', 'signal': 'BUY', 'confidence': 0.85, 'status': 'Executed'},
                {'date': (datetime.now() - timedelta(days=3)).isoformat(), 'symbol': 'MSFT', 'signal': 'BUY', 'confidence': 0.78, 'status': 'Executed'},
                {'date': (datetime.now() - timedelta(days=5)).isoformat(), 'symbol': 'GOOGL', 'signal': 'BUY', 'confidence': 0.81, 'status': 'Executed'},
                {'date': (datetime.now() - timedelta(days=7)).isoformat(), 'symbol': 'AMZN', 'signal': 'SELL', 'confidence': 0.73, 'status': 'Executed'},
                {'date': (datetime.now() - timedelta(days=9)).isoformat(), 'symbol': 'V', 'signal': 'BUY', 'confidence': 0.88, 'status': 'Pending'}
            ]
            return jsonify({
                'strategy_id': strategy_id,
                'signals': signals,
                'count': len(signals),
                'last_update': datetime.now().isoformat()
            })
        
        # Market data endpoints
        @app.route('/market/data/<symbol>')
        def market_data(symbol):
            # In a real implementation, this would get data from the data manager
            # Generate sample price data
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            prices = []
            base_price = 150.0 if symbol == 'AAPL' else 100.0
            for i, date in enumerate(dates):
                prices.append({
                    'date': date.isoformat(),
                    'open': base_price + 20 * np.sin(0.1 * i) + np.random.normal(0, 1),
                    'high': base_price + 20 * np.sin(0.1 * i) + 2 + np.random.normal(0, 1),
                    'low': base_price + 20 * np.sin(0.1 * i) - 2 + np.random.normal(0, 1),
                    'close': base_price + 20 * np.sin(0.1 * i) + 0.5 + np.random.normal(0, 1),
                    'volume': int(50000000 + 10000000 * np.random.random())
                })
            
            return jsonify({
                'symbol': symbol,
                'name': 'Apple Inc.' if symbol == 'AAPL' else f'{symbol} Inc.',
                'prices': prices,
                'period': '1m',
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/market/economic/<indicator>')
        def economic_data(indicator):
            # In a real implementation, this would get data from the data manager
            # Generate sample economic data
            dates = pd.date_range(start=datetime.now() - timedelta(days=365*5), end=datetime.now(), freq='MS')
            values = []
            
            if indicator == 'inflation':
                base = 2.0
                amplitude = 1.5
                noise = 0.2
                ylabel = 'Inflation Rate (%)'
            elif indicator == 'interest_rates':
                base = 3.0
                amplitude = 2.0
                noise = 0.1
                ylabel = 'Interest Rate (%)'
            elif indicator == 'unemployment':
                base = 5.0
                amplitude = 2.0
                noise = 0.3
                ylabel = 'Unemployment Rate (%)'
            elif indicator == 'gdp':
                base = 2.5
                amplitude = 2.0
                noise = 0.5
                ylabel = 'GDP Growth Rate (%)'
            else:  # consumer_sentiment
                base = 80.0
                amplitude = 15.0
                noise = 3.0
                ylabel = 'Consumer Sentiment Index'
            
            for i, date in enumerate(dates):
                values.append({
                    'date': date.isoformat(),
                    'value': base + amplitude * np.sin(0.1 * i) + np.random.normal(0, noise)
                })
            
            return jsonify({
                'indicator': indicator,
                'name': indicator.replace('_', ' ').title(),
                'unit': ylabel,
                'values': values,
                'period': '5y',
                'last_update': datetime.now().isoformat()
            })
        
        # Risk endpoints
        @app.route('/risk/metrics')
        def risk_metrics():
            # In a real implementation, this would get data from the risk manager
            return jsonify({
                'var_95': 25430,
                'var_99': 42150,
                'expected_shortfall': 32150,
                'beta': 0.85,
                'volatility': 12.4,
                'sharpe_ratio': 1.85,
                'sortino_ratio': 2.1,
                'max_drawdown': 8.5,
                'sector_allocations': [
                    {'sector': 'Technology', 'allocation': 35},
                    {'sector': 'Healthcare', 'allocation': 25},
                    {'sector': 'Financials', 'allocation': 15},
                    {'sector': 'Consumer', 'allocation': 15},
                    {'sector': 'Energy', 'allocation': 10}
                ],
                'factor_exposures': [
                    {'factor': 'Momentum', 'exposure': 40},
                    {'factor': 'Value', 'exposure': 25},
                    {'factor': 'Growth', 'exposure': 20},
                    {'factor': 'Quality', 'exposure': 15}
                ],
                'last_update': datetime.now().isoformat()
            })
        
        @app.route('/risk/alerts')
        def risk_alerts():
            # In a real implementation, this would get data from the risk manager
            alerts = [
                {
                    'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
                    'type': 'Sector Exposure',
                    'description': 'Technology sector exposure exceeds 35% threshold',
                    'severity': 'High',
                    'status': 'Active'
                },
                {
                    'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
                    'type': 'Volatility',
                    'description': 'Portfolio volatility increased by 20% in the last week',
                    'severity': 'Medium',
                    'status': 'Active'
                },
                {
                    'timestamp': (datetime.now() - timedelta(days=3)).isoformat(),
                    'type': 'Correlation',
                    'description': 'Increased correlation between assets reducing diversification',
                    'severity': 'Medium',
                    'status': 'Resolved'
                },
                {
                    'timestamp': (datetime.now() - timedelta(days=4)).isoformat(),
                    'type': 'Drawdown',
                    'description': 'Strategy drawdown exceeded 10% threshold',
                    'severity': 'High',
                    'status': 'Active'
                },
                {
                    'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
                    'type': 'Liquidity',
                    'description': 'Reduced liquidity in small-cap holdings',
                    'severity': 'Low',
                    'status': 'Monitoring'
                }
            ]
            return jsonify({
                'alerts': alerts,
                'count': len(alerts),
                'active_count': sum(1 for alert in alerts if alert['status'] == 'Active'),
                'last_update': datetime.now().isoformat()
            })


# For testing the API server directly
if __name__ == "__main__":
    import numpy as np
    
    # Mock configuration and managers for testing
    class MockManager:
        async def initialize(self):
            pass
    
    config = {"ui": {"api_port": 8081}}
    portfolio_manager = MockManager()
    strategy_manager = MockManager()
    risk_manager = MockManager()
    data_manager = MockManager()
    
    # Create and run API server
    api_server = APIServer(config, portfolio_manager, strategy_manager, risk_manager, data_manager)
    api_server._run_api_server()