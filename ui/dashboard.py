#!/usr/bin/env python3
"""
AI Hedge Fund - Web Dashboard

A sleek, minimalist web dashboard for the AI Hedge Fund system that allows users to:
- Monitor portfolio performance and holdings
- View strategy signals and performance
- Start/stop the trading system
- Configure system parameters
- View risk metrics and alerts

Built with Streamlit for a responsive and visually pleasing interface.
"""

import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import threading
import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class Dashboard:
    """
    Web dashboard for the AI Hedge Fund system.
    Provides a user-friendly interface to monitor and control the system.
    """
    
    def __init__(self, config, portfolio_manager, strategy_manager, risk_manager, data_manager):
        """
        Initialize the dashboard with system components.
        
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
        
        self.dashboard_port = self.config.get('ui', {}).get('dashboard_port', 8080)
        self.dashboard_thread = None
        self.running = False
        
    async def start(self):
        """
        Start the dashboard in a separate thread.
        """
        if self.running:
            return
            
        self.running = True
        self.dashboard_thread = threading.Thread(target=self._run_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
        
    async def stop(self):
        """
        Stop the dashboard.
        """
        self.running = False
        # Streamlit doesn't have a clean shutdown method, the thread will terminate when the main process exits
        
    def _run_dashboard(self):
        """
        Run the Streamlit dashboard.
        """
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = str(self.dashboard_port)
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        
        # This is a hack to run Streamlit programmatically
        # In a real implementation, you would create a separate Streamlit script
        # and run it using the Streamlit CLI
        import streamlit.web.bootstrap as bootstrap
        bootstrap.run(self._streamlit_app, '', [], flag_options={})
        
    def _streamlit_app(self):
        """
        Define the Streamlit application.
        """
        st.set_page_config(
            page_title="AI Hedge Fund Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Apply custom CSS for a sleek, minimalist design
        st.markdown("""
        <style>
        .main {background-color: #f5f7fa}
        .stApp {max-width: 1200px; margin: 0 auto;}
        h1, h2, h3 {font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;}
        .stTabs [data-baseweb="tab-list"] {gap: 24px;}
        .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap;}
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar for system control
        with st.sidebar:
            st.title("AI Hedge Fund")
            st.write("System Control Panel")
            
            # System status
            system_status = "Running" if self.running else "Stopped"
            st.write(f"Status: {system_status}")
            
            # Start/Stop buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start System", use_container_width=True):
                    st.session_state["system_command"] = "start"
            with col2:
                if st.button("Stop System", use_container_width=True):
                    st.session_state["system_command"] = "stop"
            
            # System mode selection
            st.write("System Mode")
            mode = st.radio(
                "Select operation mode:",
                ["Live Trading", "Paper Trading", "Backtesting", "Training"]
            )
            
            # Date range for backtesting
            if mode == "Backtesting":
                st.write("Backtesting Period")
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", datetime.now())
                initial_capital = st.number_input("Initial Capital", min_value=10000, value=100000, step=10000)
            
            # Training settings
            if mode == "Training":
                st.write("Training Settings")
                strategy_to_train = st.selectbox(
                    "Strategy to Train",
                    ["Warren Buffett", "Peter Lynch", "Benjamin Graham", "Ray Dalio", "All Strategies"]
                )
                training_epochs = st.slider("Training Epochs", min_value=10, max_value=1000, value=100)
            
            # System information
            st.write("System Information")
            st.info(
                f"Dashboard: http://localhost:{self.dashboard_port}\n"
                f"API Server: http://localhost:{self.config.get('ui', {}).get('api_port', 8081)}"
            )
        
        # Main content area with tabs
        st.title("AI Hedge Fund Dashboard")
        
        tabs = st.tabs(["Portfolio", "Strategies", "Market Data", "Risk Metrics", "System Settings"])
        
        # Portfolio Tab
        with tabs[0]:
            st.header("Portfolio Overview")
            
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", "$1,250,430", "+2.3%")
            with col2:
                st.metric("Daily P&L", "+$12,504", "+1.01%")
            with col3:
                st.metric("Monthly Return", "+4.2%", "+0.5%")
            with col4:
                st.metric("Sharpe Ratio", "1.85", "+0.03")
            
            # Portfolio performance chart
            st.subheader("Performance History")
            
            # Sample data for demonstration
            dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
            performance = pd.DataFrame({
                'Date': dates,
                'Portfolio Value': 1000000 * (1 + 0.0003 * (dates - dates[0]).days.values) + \
                                  50000 * np.sin(0.05 * (dates - dates[0]).days.values),
                'Benchmark': 1000000 * (1 + 0.0002 * (dates - dates[0]).days.values)
            })
            
            fig = px.line(performance, x='Date', y=['Portfolio Value', 'Benchmark'],
                         title='Portfolio Performance vs Benchmark')
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Holdings table
            st.subheader("Current Holdings")
            
            # Sample holdings data
            holdings = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'Name': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Tesla Inc.'],
                'Quantity': [150, 100, 50, 30, 75],
                'Avg Price': [150.25, 280.10, 2750.80, 3200.50, 190.75],
                'Current Price': [155.75, 290.30, 2800.10, 3250.20, 185.60],
                'Market Value': [23362.50, 29030.00, 140005.00, 97506.00, 13920.00],
                'Gain/Loss %': [3.66, 3.64, 1.79, 1.55, -2.70]
            })
            
            st.dataframe(holdings, use_container_width=True)
        
        # Strategies Tab
        with tabs[1]:
            st.header("Strategy Performance")
            
            # Strategy selector
            strategy = st.selectbox(
                "Select Strategy",
                ["All Strategies", "Warren Buffett", "Peter Lynch", "Benjamin Graham", "Ray Dalio", 
                 "George Soros", "Joel Greenblatt", "Fed Policy Tracker", "Inflation Protection"]
            )
            
            # Strategy performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Win Rate", "68%", "+2%")
            with col2:
                st.metric("Avg Return", "+1.8%", "+0.3%")
            with col3:
                st.metric("Sharpe Ratio", "1.92", "+0.05")
            with col4:
                st.metric("Active Signals", "7", "+2")
            
            # Strategy performance chart
            st.subheader("Strategy Performance")
            
            # Sample strategy performance data
            dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
            strategy_perf = pd.DataFrame({
                'Date': dates,
                'Warren Buffett': 1000000 * (1 + 0.0004 * (dates - dates[0]).days.values) + \
                                 30000 * np.sin(0.06 * (dates - dates[0]).days.values),
                'Peter Lynch': 1000000 * (1 + 0.00035 * (dates - dates[0]).days.values) + \
                              40000 * np.sin(0.05 * (dates - dates[0]).days.values + 1),
                'Benjamin Graham': 1000000 * (1 + 0.00025 * (dates - dates[0]).days.values) + \
                                  45000 * np.sin(0.04 * (dates - dates[0]).days.values + 2),
                'Benchmark': 1000000 * (1 + 0.0002 * (dates - dates[0]).days.values)
            })
            
            fig = px.line(strategy_perf, x='Date', y=['Warren Buffett', 'Peter Lynch', 'Benjamin Graham', 'Benchmark'],
                         title='Strategy Performance Comparison')
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent signals
            st.subheader("Recent Strategy Signals")
            
            # Sample signals data
            signals = pd.DataFrame({
                'Date': pd.date_range(start=datetime.now() - timedelta(days=10), end=datetime.now(), freq='D'),
                'Strategy': ['Warren Buffett', 'Peter Lynch', 'Benjamin Graham', 'Ray Dalio', 'Warren Buffett',
                            'George Soros', 'Joel Greenblatt', 'Fed Policy Tracker', 'Inflation Protection', 'Peter Lynch'],
                'Symbol': ['AAPL', 'MSFT', 'JNJ', 'PG', 'GOOGL', 'AMZN', 'V', 'TLT', 'GLD', 'TSLA'],
                'Signal': ['BUY', 'BUY', 'BUY', 'HOLD', 'BUY', 'SELL', 'BUY', 'SELL', 'BUY', 'SELL'],
                'Confidence': [0.85, 0.78, 0.92, 0.65, 0.81, 0.73, 0.88, 0.77, 0.82, 0.79],
                'Status': ['Executed', 'Executed', 'Executed', 'N/A', 'Executed', 'Executed', 'Pending', 'Executed', 'Pending', 'Executed']
            })
            
            st.dataframe(signals, use_container_width=True)
        
        # Market Data Tab
        with tabs[2]:
            st.header("Market Data")
            
            # Market data selector
            data_type = st.selectbox(
                "Select Data Type",
                ["Stocks", "ETFs", "Indices", "Economic Indicators", "Sentiment Analysis"]
            )
            
            if data_type in ["Stocks", "ETFs", "Indices"]:
                # Symbol search
                symbol = st.text_input("Enter Symbol", "AAPL")
                
                # Market data metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Price", "$155.75", "+1.2%")
                with col2:
                    st.metric("Volume", "52.3M", "-5.1%")
                with col3:
                    st.metric("P/E Ratio", "28.5", "+0.3")
                with col4:
                    st.metric("52W Range", "$120.45 - $182.94", "")
                
                # Price chart
                st.subheader("Price History")
                
                # Sample price data
                dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
                price_data = pd.DataFrame({
                    'Date': dates,
                    'Open': 150 + 20 * np.sin(0.1 * np.arange(len(dates))) + np.random.normal(0, 1, len(dates)),
                    'High': 150 + 20 * np.sin(0.1 * np.arange(len(dates))) + 2 + np.random.normal(0, 1, len(dates)),
                    'Low': 150 + 20 * np.sin(0.1 * np.arange(len(dates))) - 2 + np.random.normal(0, 1, len(dates)),
                    'Close': 150 + 20 * np.sin(0.1 * np.arange(len(dates))) + 0.5 + np.random.normal(0, 1, len(dates)),
                    'Volume': 50000000 + 10000000 * np.random.random(len(dates))
                })
                
                fig = go.Figure(data=[go.Candlestick(
                    x=price_data['Date'],
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close']
                )])
                fig.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
            elif data_type == "Economic Indicators":
                # Economic indicator selector
                indicator = st.selectbox(
                    "Select Economic Indicator",
                    ["GDP Growth", "Unemployment Rate", "Inflation Rate", "Interest Rates", "Consumer Sentiment"]
                )
                
                # Economic data chart
                st.subheader(f"{indicator} History")
                
                # Sample economic data
                dates = pd.date_range(start=datetime.now() - timedelta(days=365*5), end=datetime.now(), freq='MS')
                if indicator == "Inflation Rate":
                    values = 2 + 1.5 * np.sin(0.1 * np.arange(len(dates))) + np.random.normal(0, 0.2, len(dates))
                    ylabel = "Inflation Rate (%)"
                elif indicator == "Interest Rates":
                    values = 3 + 2 * np.sin(0.05 * np.arange(len(dates))) + np.random.normal(0, 0.1, len(dates))
                    ylabel = "Interest Rate (%)"
                elif indicator == "Unemployment Rate":
                    values = 5 + 2 * np.sin(0.08 * np.arange(len(dates))) + np.random.normal(0, 0.3, len(dates))
                    ylabel = "Unemployment Rate (%)"
                elif indicator == "GDP Growth":
                    values = 2.5 + 2 * np.sin(0.2 * np.arange(len(dates))) + np.random.normal(0, 0.5, len(dates))
                    ylabel = "GDP Growth Rate (%)"
                else:  # Consumer Sentiment
                    values = 80 + 15 * np.sin(0.1 * np.arange(len(dates))) + np.random.normal(0, 3, len(dates))
                    ylabel = "Consumer Sentiment Index"
                
                econ_data = pd.DataFrame({
                    'Date': dates,
                    'Value': values
                })
                
                fig = px.line(econ_data, x='Date', y='Value', title=f"{indicator} Over Time")
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis_title=ylabel)
                st.plotly_chart(fig, use_container_width=True)
            
            elif data_type == "Sentiment Analysis":
                # Sentiment analysis selector
                sentiment_type = st.selectbox(
                    "Select Sentiment Type",
                    ["Market Sentiment", "Sector Sentiment", "Stock Sentiment"]
                )
                
                if sentiment_type == "Stock Sentiment":
                    symbol = st.text_input("Enter Symbol", "AAPL")
                
                # Sentiment chart
                st.subheader(f"{sentiment_type} Analysis")
                
                # Sample sentiment data
                dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
                sentiment_data = pd.DataFrame({
                    'Date': dates,
                    'Positive': 0.6 + 0.2 * np.sin(0.3 * np.arange(len(dates))) + np.random.normal(0, 0.05, len(dates)),
                    'Neutral': 0.3 + 0.1 * np.sin(0.2 * np.arange(len(dates)) + 1) + np.random.normal(0, 0.03, len(dates)),
                    'Negative': 0.1 + 0.1 * np.sin(0.25 * np.arange(len(dates)) + 2) + np.random.normal(0, 0.02, len(dates))
                })
                
                # Normalize to ensure sum is 1
                for i in range(len(sentiment_data)):
                    total = sentiment_data.loc[i, ['Positive', 'Neutral', 'Negative']].sum()
                    sentiment_data.loc[i, ['Positive', 'Neutral', 'Negative']] /= total
                
                fig = px.area(sentiment_data, x='Date', y=['Positive', 'Neutral', 'Negative'],
                              title=f"{sentiment_type} Over Time", color_discrete_map={
                                  'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'
                              })
                fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk Metrics Tab
        with tabs[3]:
            st.header("Risk Management")
            
            # Risk metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Value at Risk (95%)", "$25,430", "-1.5%")
            with col2:
                st.metric("Expected Shortfall", "$32,150", "-2.1%")
            with col3:
                st.metric("Beta", "0.85", "-0.03")
            with col4:
                st.metric("Volatility", "12.4%", "-0.5%")
            
            # Risk allocation chart
            st.subheader("Risk Allocation")
            
            # Sample risk allocation data
            risk_allocation = pd.DataFrame({
                'Category': ['Sector', 'Sector', 'Sector', 'Sector', 'Sector', 'Factor', 'Factor', 'Factor', 'Factor'],
                'Name': ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy', 
                         'Momentum', 'Value', 'Growth', 'Quality'],
                'Allocation': [35, 25, 15, 15, 10, 40, 25, 20, 15]
            })
            
            # Create two separate dataframes for sectors and factors
            sectors = risk_allocation[risk_allocation['Category'] == 'Sector']
            factors = risk_allocation[risk_allocation['Category'] == 'Factor']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(sectors, values='Allocation', names='Name', title='Sector Risk Allocation')
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(factors, values='Allocation', names='Name', title='Factor Risk Allocation')
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk alerts
            st.subheader("Risk Alerts")
            
            # Sample risk alerts
            risk_alerts = pd.DataFrame({
                'Timestamp': pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='D'),
                'Type': ['Sector Exposure', 'Volatility', 'Correlation', 'Drawdown', 'Liquidity'],
                'Description': [
                    'Technology sector exposure exceeds 35% threshold',
                    'Portfolio volatility increased by 20% in the last week',
                    'Increased correlation between assets reducing diversification',
                    'Strategy drawdown exceeded 10% threshold',
                    'Reduced liquidity in small-cap holdings'
                ],
                'Severity': ['High', 'Medium', 'Medium', 'High', 'Low'],
                'Status': ['Active', 'Active', 'Resolved', 'Active', 'Monitoring']
            })
            
            # Apply color to severity
            def color_severity(val):
                if val == 'High':
                    return 'background-color: #ffcccc'
                elif val == 'Medium':
                    return 'background-color: #ffffcc'
                else:
                    return 'background-color: #ccffcc'
            
            st.dataframe(risk_alerts.style.applymap(color_severity, subset=['Severity']), use_container_width=True)
        
        # System Settings Tab
        with tabs[4]:
            st.header("System Settings")
            
            # Settings sections
            st.subheader("Trading Settings")
            col1, col2 = st.columns(2)
            with col1:
                trading_mode = st.selectbox("Trading Mode", ["Live", "Paper"], index=1)
                update_interval = st.number_input("Update Interval (seconds)", min_value=60, max_value=3600, value=300, step=60)
            with col2:
                max_position_size = st.slider("Max Position Size (%)", min_value=1, max_value=20, value=5)
                max_sector_allocation = st.slider("Max Sector Allocation (%)", min_value=5, max_value=50, value=25)
            
            st.subheader("Strategy Settings")
            
            # Sample strategy settings
            strategies = [
                {"name": "Warren Buffett", "enabled": True, "weight": 25},
                {"name": "Peter Lynch", "enabled": True, "weight": 20},
                {"name": "Benjamin Graham", "enabled": True, "weight": 15},
                {"name": "Ray Dalio", "enabled": True, "weight": 15},
                {"name": "George Soros", "enabled": False, "weight": 0},
                {"name": "Joel Greenblatt", "enabled": True, "weight": 10},
                {"name": "Fed Policy Tracker", "enabled": True, "weight": 10},
                {"name": "Inflation Protection", "enabled": True, "weight": 5}
            ]
            
            for i, strategy in enumerate(strategies):
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.write(strategy["name"])
                with col2:
                    strategies[i]["enabled"] = st.checkbox("Enabled", value=strategy["enabled"], key=f"enabled_{i}")
                with col3:
                    if strategies[i]["enabled"]:
                        strategies[i]["weight"] = st.slider("Weight (%)", min_value=0, max_value=100, 
                                                           value=strategy["weight"], key=f"weight_{i}")
                    else:
                        st.write("Disabled")
            
            st.subheader("Data Settings")
            col1, col2 = st.columns(2)
            with col1:
                data_refresh = st.number_input("Data Refresh Interval (seconds)", min_value=60, max_value=3600, value=300, step=60)
                cache_expiry = st.number_input("Cache Expiry (hours)", min_value=1, max_value=72, value=24, step=1)
            with col2:
                st.write("Data Sources")
                st.checkbox("Alpha Vantage", value=True)
                st.checkbox("FRED", value=True)
                st.checkbox("Finnhub", value=True)
                st.checkbox("Quandl", value=True)
            
            # Save settings button
            if st.button("Save Settings", use_container_width=True):
                st.success("Settings saved successfully!")
        
        # Process system commands
        if "system_command" in st.session_state:
            if st.session_state["system_command"] == "start":
                st.success("System started successfully!")
                del st.session_state["system_command"]
            elif st.session_state["system_command"] == "stop":
                st.warning("System stopped!")
                del st.session_state["system_command"]


# For testing the dashboard directly
if __name__ == "__main__":
    import numpy as np
    
    # Mock configuration and managers for testing
    class MockManager:
        async def initialize(self):
            pass
    
    config = {"ui": {"dashboard_port": 8080}}
    portfolio_manager = MockManager()
    strategy_manager = MockManager()
    risk_manager = MockManager()
    data_manager = MockManager()
    
    # Create and run dashboard
    dashboard = Dashboard(config, portfolio_manager, strategy_manager, risk_manager, data_manager)
    dashboard._run_dashboard()