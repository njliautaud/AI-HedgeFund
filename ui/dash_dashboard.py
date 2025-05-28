#!/usr/bin/env python3
"""
AI Hedge Fund - Dash Web Dashboard

A sleek, minimalist web dashboard for the AI Hedge Fund system using Dash.
Provides a modern, responsive interface for monitoring and controlling the system.
"""

import os
import sys
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class DashDashboard:
    """
    Web dashboard for the AI Hedge Fund system using Dash.
    Provides a user-friendly interface to monitor and control the system.
    """
    
    def __init__(self, config, portfolio_manager=None, strategy_manager=None, risk_manager=None, data_manager=None, agent_manager=None):
        """
        Initialize the dashboard with system components.
        
        Args:
            config: System configuration
            portfolio_manager: Portfolio management component
            strategy_manager: Strategy management component
            risk_manager: Risk management component (can be None)
            data_manager: Data management component
            agent_manager: Agent management component
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager # May be None
        self.data_manager = data_manager
        self.agent_manager = agent_manager # Changed from ai_model_manager
        # self.ai_model_manager can be accessed via agent_manager if needed, or passed separately
        self.ai_model_manager = agent_manager.ai_model_manager if agent_manager else None
        
        self.dashboard_port = self.config.get('ui', {}).get('dashboard_port', 8080)
        self.dashboard_thread = None
        self.running = False
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="AI Hedge Fund Dashboard",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        # Set up the layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        self._setup_agent_callbacks() # Add new callbacks for agents
    
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
        # The thread will terminate when the main process exits
    
    def _run_dashboard(self):
        """
        Run the Dash dashboard.
        """
        self.app.run(debug=False, port=self.dashboard_port, host='0.0.0.0')
    
    def _create_model_training_chart(self, model_id, metrics):
        """
        Create a chart showing model training metrics over epochs.
        
        Args:
            model_id: Identifier for the model
            metrics: Dictionary containing training metrics
            
        Returns:
            Plotly figure object
        """
        # Convert metrics to DataFrame for plotting
        epochs = list(range(1, len(metrics['loss']) + 1))
        training_data = pd.DataFrame({
            'Epoch': epochs,
            'Training Loss': metrics['loss'],
            'Validation Loss': metrics['val_loss'],
            'Training Accuracy': metrics['accuracy'],
            'Validation Accuracy': metrics['val_accuracy']
        })
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add traces for loss (left y-axis)
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Training Loss'],
            name='Training Loss',
            line=dict(color='#1f77b4')
        ))
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Validation Loss'],
            name='Validation Loss',
            line=dict(color='#1f77b4', dash='dash')
        ))
        
        # Add traces for accuracy (right y-axis)
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Training Accuracy'],
            name='Training Accuracy',
            line=dict(color='#ff7f0e'),
            yaxis='y2'
        ))
        fig.add_trace(go.Scatter(
            x=training_data['Epoch'],
            y=training_data['Validation Accuracy'],
            name='Validation Accuracy',
            line=dict(color='#ff7f0e', dash='dash'),
            yaxis='y2'
        ))
        
        # Set up layout with two y-axes
        fig.update_layout(
            title=f'Training Metrics for {model_id}',
            xaxis=dict(title='Epoch'),
            yaxis=dict(
                title='Loss',
                titlefont=dict(color='#1f77b4'),
                tickfont=dict(color='#1f77b4')
            ),
            yaxis2=dict(
                title='Accuracy',
                titlefont=dict(color='#ff7f0e'),
                tickfont=dict(color='#ff7f0e'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        return fig
    
    def _setup_layout(self):
        """
        Set up the Dash app layout.
        """
        # Create a modern, sleek layout with sidebar and main content area
        self.app.layout = dbc.Container(
            [
                dbc.Row(
                    [
                        # Sidebar
                        dbc.Col(
                            [
                                html.H2("AI Hedge Fund", className="display-6 mt-3 mb-4"),
                                html.Hr(),
                                html.H5("System Control", className="mb-3"),
                                
                                # System status
                                html.Div(
                                    [
                                        html.P("Status: ", className="d-inline"),
                                        html.P("Running", id="system-status", className="d-inline text-success")
                                    ],
                                    className="mb-3"
                                ),
                                
                                # Start/Stop buttons
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.Button("Start System", id="start-button", color="success", className="w-100")),
                                        dbc.Col(dbc.Button("Stop System", id="stop-button", color="danger", className="w-100"))
                                    ],
                                    className="mb-4"
                                ),
                                
                                # System mode selection
                                html.H5("System Mode", className="mb-2"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Live Trading", "value": "live"},
                                        {"label": "Paper Trading", "value": "paper"},
                                        {"label": "Backtesting", "value": "backtest"},
                                        {"label": "Training", "value": "training"}
                                    ],
                                    value="paper",
                                    id="mode-selection",
                                    className="mb-3"
                                ),
                                
                                # Backtesting settings (conditionally displayed)
                                html.Div(
                                    [
                                        html.H6("Backtesting Period", className="mt-3 mb-2"),
                                        dbc.Label("Start Date"),
                                        dcc.DatePickerSingle(
                                            id="backtest-start-date",
                                            date=datetime.now() - timedelta(days=365),
                                            className="mb-2"
                                        ),
                                        dbc.Label("End Date"),
                                        dcc.DatePickerSingle(
                                            id="backtest-end-date",
                                            date=datetime.now(),
                                            className="mb-2"
                                        ),
                                        dbc.Label("Initial Capital"),
                                        dbc.Input(id="initial-capital", type="number", value=100000, min=10000, step=10000)
                                    ],
                                    id="backtest-settings",
                                    style={"display": "none"}
                                ),
                                
                                # System information
                                html.Div(
                                    [
                                        html.H5("System Information", className="mt-4 mb-2"),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.P(f"Dashboard: http://localhost:{self.dashboard_port}"),
                                                    html.P(f"API Server: http://localhost:{self.config.get('ui', {}).get('api_port', 8081)}")
                                                ]
                                            ),
                                            className="mb-3"
                                        )
                                    ]
                                )
                            ],
                            width=3,
                            className="bg-light p-4 h-100"
                        ),
                        
                        # Main content area with tabs
                        dbc.Col(
                            [
                                html.H1("AI Hedge Fund Dashboard", className="mt-3 mb-4"),
                                
                                # Tabs for different sections
                                dbc.Tabs(
                                    [
                                        # Portfolio Tab
                                        dbc.Tab(
                                            [
                                                html.H3("Portfolio Overview", className="mt-3 mb-4"),
                                                
                                                # Portfolio metrics cards
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Total Value"), html.H3(id="total-value", children="$1,250,430"), html.P(id="total-value-change", children="+2.3%", className="text-success")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Daily P&L"), html.H3(id="daily-pnl", children="+$12,504"), html.P(id="daily-pnl-percent", children="+1.01%", className="text-success")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Monthly Return"), html.H3(id="monthly-return", children="+4.2%"), html.P(id="monthly-return-change", children="+0.5%", className="text-success")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Sharpe Ratio"), html.H3(id="sharpe-ratio", children="1.85"), html.P(id="sharpe-ratio-change", children="+0.03", className="text-success")]), className="text-center"))
                                                    ],
                                                    className="mb-4"
                                                ),
                                                
                                                # Portfolio performance chart
                                                html.H4("Performance History", className="mt-4 mb-3"),
                                                dcc.Graph(id="portfolio-chart", figure=self._create_portfolio_chart()),
                                                
                                                # Holdings table
                                                html.H4("Current Holdings", className="mt-4 mb-3"),
                                                self._create_holdings_table()
                                            ],
                                            label="Portfolio"
                                        ),
                                        
                                        # Strategies Tab
                                        dbc.Tab(
                                            [
                                                html.H3("Strategy Performance", className="mt-3 mb-4"),
                                                
                                                # Strategy allocation chart
                                                html.H4("Strategy Allocation", className="mt-3 mb-3"),
                                                dcc.Graph(id="strategy-allocation-chart", figure=self._create_strategy_allocation_chart()),
                                                
                                                # Strategy performance table
                                                html.H4("Strategy Performance", className="mt-4 mb-3"),
                                                self._create_strategy_performance_table()
                                            ],
                                            label="Strategies"
                                        ),
                                        
                                        # Market Data Tab
                                        dbc.Tab(
                                            [
                                                html.H3("Market Data", className="mt-3 mb-4"),
                                                
                                                # Market indices chart
                                                html.H4("Market Indices", className="mt-3 mb-3"),
                                                dcc.Graph(id="market-indices-chart", figure=self._create_market_indices_chart()),
                                                
                                                # Economic indicators
                                                html.H4("Economic Indicators", className="mt-4 mb-3"),
                                                self._create_economic_indicators()
                                            ],
                                            label="Market Data"
                                        ),
                                        
                                        # Risk Metrics Tab
                                        dbc.Tab(
                                            [
                                                html.H3("Risk Metrics", className="mt-3 mb-4"),
                                                
                                                # Risk metrics cards
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Portfolio Beta"), html.H3(id="portfolio-beta", children="0.85")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Value at Risk (95%)"), html.H3(id="var-value", children="$12,504")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Max Drawdown"), html.H3(id="max-drawdown", children="8.2%")]), className="text-center")),
                                                        dbc.Col(dbc.Card(dbc.CardBody([html.H5("Volatility"), html.H3(id="volatility", children="12.5%")]), className="text-center"))
                                                    ],
                                                    className="mb-4"
                                                ),
                                                
                                                # Risk heatmap
                                                html.H4("Correlation Heatmap", className="mt-4 mb-3"),
                                                dcc.Graph(id="correlation-heatmap", figure=self._create_correlation_heatmap())
                                            ],
                                            label="Risk Metrics"
                                        ),
                                        
                                        # Model Training Tab (New)
                                        dbc.Tab(
                                            [
                                                html.H3("AI Model Training & Analysis", className="mt-3 mb-4"),
                                                
                                                # Model selection and training controls
                                                dbc.Row(
                                                    [
                                                        dbc.Col(
                                                            [
                                                                html.H4("Model Training", className="mb-3"),
                                                                dbc.Label("Select Model"),
                                                                dcc.Dropdown(
                                                                    id="model-selection",
                                                                    options=[
                                                                        {"label": "Warren Buffett Value Investor", "value": "legendary_investors_warren_buffett"},
                                                                        {"label": "Peter Lynch Growth Investor", "value": "legendary_investors_peter_lynch"},
                                                                        {"label": "Benjamin Graham Deep Value", "value": "legendary_investors_benjamin_graham"},
                                                                        {"label": "Ray Dalio All Weather", "value": "legendary_investors_ray_dalio"},
                                                                        {"label": "Momentum Strategy", "value": "technical_momentum"},
                                                                        {"label": "Mean Reversion Strategy", "value": "technical_mean_reversion"}
                                                                    ],
                                                                    value="legendary_investors_warren_buffett"
                                                                ),
                                                                
                                                                # Training parameters
                                                                dbc.Row(
                                                                    [
                                                                        dbc.Col(
                                                                            [
                                                                                dbc.Label("Epochs", className="mt-3"),
                                                                                dbc.Input(id="training-epochs", type="number", value=50, min=10, max=1000, step=10)
                                                                            ],
                                                                            width=6
                                                                        ),
                                                                        dbc.Col(
                                                                            [
                                                                                dbc.Label("Batch Size", className="mt-3"),
                                                                                dbc.Input(id="training-batch-size", type="number", value=32, min=8, max=256, step=8)
                                                                            ],
                                                                            width=6
                                                                        )
                                                                    ],
                                                                    className="mb-3"
                                                                ),
                                                                
                                                                # Training button
                                                                dbc.Button("Train Model", id="train-model-button", color="primary", className="mt-2")
                                                            ],
                                                            width=6
                                                        ),
                                                        
                                                        dbc.Col(
                                                            [
                                                                dbc.Label("Analysis Type"),
                                                                dbc.RadioItems(
                                                                    id="analysis-type",
                                                                    options=[
                                                                        {"label": "Fundamental", "value": "fundamental"},
                                                                        {"label": "Technical", "value": "technical"},
                                                                        {"label": "Sentiment", "value": "sentiment"},
                                                                        {"label": "All", "value": "all"}
                                                                    ],
                                                                    value="all",
                                                                    inline=True
                                                                )
                                                            ],
                                                            width=6
                                                        )
                                                    ],
                                                    className="mb-3"
                                                ),
                                                
                                                # Model training results container
                                                html.Div(id="model-training-results"),
                                                html.Div(id="model-training-history"),
                                                html.Div(id="model-metrics", className="mt-3")
                                            ],
                                            label="Model Training" # Changed from "Strategies"
                                        ),
                                        
                                        # Agent control section
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H5("Agent Control", className="card-title"),
                                            dcc.Dropdown(
                                                id='agent-selector-dropdown',
                                                options=[], # To be populated by callback
                                                placeholder="Select an Agent",
                                                className="mb-2"
                                            ),
                                            html.Div(id='agent-details-output'),
                                            # Add more agent-specific controls here if needed
                                        ]
                                    ),
                                    className="mb-3"
                                ),

                                # Ticker Analysis Section
                                dbc.Card(
                                    dbc.CardBody([
                                        html.H5("Ticker Analysis", className="card-title"),
                                        dcc.Input(id="ticker-input", type="text", placeholder="Enter Ticker (e.g., AAPL)", className="mb-2"),
                                        dcc.Dropdown(
                                            id='agent-for-ticker-analysis-dropdown',
                                            options=[], # To be populated by callback
                                            placeholder="Select Agent for Analysis",
                                            className="mb-2"
                                        ),
                                        dbc.Button("Analyze Ticker", id="analyze-ticker-button", color="primary", className="mb-2"),
                                        dcc.Loading(id="loading-ticker-analysis", children=[html.Div(id="ticker-analysis-results")])
                                    ]),
                                    className="mb-3"
                                ),
                                    ]
                                    # label="Model Training" # Removed incorrect label from dbc.Tabs
                                )
                            ],
                            width=9