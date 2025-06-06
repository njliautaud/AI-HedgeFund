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
import requests # Added for API calls
from loguru import logger

# Import our new components
from ui.components.emerging_markets import EmergingMarketsComponent

# Configure logger
logger.add(
    "logs/dashboard.log",
    rotation="10 MB",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
)

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback, callback_context # Added State
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
        self.agent_manager = agent_manager 
        self.ai_model_manager = agent_manager.ai_model_manager if agent_manager else None
        
        self.dashboard_port = self.config.get('ui', {}).get('dashboard_port', 8088)  # Using new port from config
        # Get API port from config
        api_port = self.config.get('ui', {}).get('api_port', 8089)
        self.api_base_url = f"http://localhost:{api_port}/api/v1"
        self.dashboard_thread = None
        self.running = False
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME], # Changed theme for a more modern look
            title="AI Hedge Fund Dashboard",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        self.server = self.app.server # Expose server for Gunicorn/Waitress
        
        # Set up the layout using URL routing
        self.app.layout = self._create_main_layout() # Changed to main layout for routing
        
        # Set up callbacks
        self._setup_callbacks()
        self._setup_emerging_markets_callbacks()
        
        # Set up callbacks using decorators
        @self.app.callback(
            Output("page-content", "children"),
            Input("url", "pathname")
        )
        def display_page(pathname):
            return self._render_page_content(pathname)
            
        @self.app.callback(
            Output("agent-details", "children"),
            Input("agent-select", "value"),
            State("url", "pathname")
        )
        def display_agent_details(agent_id, pathname):
            return self._create_agent_details(agent_id, pathname)
            
        @callback(
            Output("market-data", "children"),
            Input("interval-component", "n_intervals")
        )
        def update_emerging_markets(n_intervals):
            return self._update_emerging_markets(n_intervals)
            
        # Add other specific callback setups here (e.g., portfolio, strategy)
    
    def _run_dashboard(self):
        """
        Run the Dash dashboard.
        This method is run in a separate thread.
        """
        try:
            logger.info(f"Starting dashboard server on port {self.dashboard_port}")
            self.app.run_server(debug=False, port=self.dashboard_port, host='0.0.0.0')
        except Exception as e:
            logger.error(f"Error in dashboard server: {e}")
            self.running = False
    
    async def start(self):
        """
        Start the dashboard in a separate thread.
        """
        if self.running:
            logger.warning("Dashboard is already running")
            return
            
        try:
            self.running = True
            # Create a thread to run the dashboard
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True,
                name="DashboardThread"
            )
            self.dashboard_thread.start()
            logger.info("Dashboard started")
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self.running = False
            raise
        
    async def stop(self):
        """
        Stop the dashboard.
        """
        self.running = False
        # The thread will terminate when the main process exits
    
    def _create_main_layout(self):
        """
        Create the main layout of the dashboard with navigation and content area.
        
        Returns:
            dash.html.Div: The main layout component
        """
        # Define the navigation bar
        navbar = dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink("Overview", href="/", active="exact")),
                dbc.NavItem(dbc.NavLink("Portfolio", href="/portfolio", active="exact")),
                dbc.NavItem(dbc.NavLink("Strategies", href="/strategies", active="exact")),
                dbc.NavItem(dbc.NavLink("Emerging Markets", href="/emerging-markets", active="exact")),
                dbc.NavItem(dbc.NavLink("Risk", href="/risk", active="exact")),
                dbc.NavItem(dbc.NavLink("Models", href="/models", active="exact")),
                dbc.NavItem(dbc.NavLink("Agents", href="/agents", active="exact")),
                dbc.NavItem(dbc.NavLink("Settings", href="/settings", active="exact")),
            ],
            brand="AI Hedge Fund",
            brand_href="/",
            color="primary",
            dark=True,
            className="mb-4"
        )
        
        # Define the content area that will be updated based on URL
        content = html.Div(id="page-content", className="container")
        
        # Combine navbar and content with URL routing
        layout = html.Div([
            dcc.Location(id="url"),
            navbar,
            content,
            # Add hidden div for storing intermediate values
            html.Div(id="intermediate-value", style={"display": "none"}),
            # Add interval for auto-updating data
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # in milliseconds, update every minute
                n_intervals=0
            )
        ])
        
        return layout
        
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
                                html.Div(
                                    [
                                        html.I(className="fas fa-brain fa-2x me-2"), # Changed icon
                                        html.H2("AI Hedge Fund", className="display-6"),
                                    ],
                                    className="d-flex align-items-center mt-3 mb-4"
                                ),
                                html.Hr(),
                                dbc.Nav(
                                    [
                                        dbc.NavLink([html.I(className="fas fa-tachometer-alt me-2"), "Overview"], href="/", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-users-cog me-2"), "Agent Control"], href="/agents", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-cogs me-2"), "Model Training"], href="/models", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-chart-pie me-2"), "Portfolio"], href="/portfolio", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-lightbulb me-2"), "Strategies"], href="/strategies", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-database me-2"), "Market Data"], href="/market-data", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-shield-alt me-2"), "Risk Metrics"], href="/risk", active="exact"),
                                        dbc.NavLink([html.I(className="fas fa-globe me-2"), "Emerging Markets"], href="/emerging-markets", active="exact"),  # Note: Typo in URL path (should be 'markets')
                                    ],
                                    vertical=True,
                                    pills=True,
                                    className="mb-4"
                                ),
                                html.Hr(),
                                html.Div(
                                    [
                                        html.H5("System Information", className="mt-4 mb-2"),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.P(f"Dashboard: http://localhost:{self.dashboard_port}"),
                                                    html.P(f"API Server: {self.api_base_url.replace('/api/v1', '')}")
                                                ]
                                            ),
                                            className="mb-3"
                                        )
                                    ]
                                )
                            ],
                            width=3,
                            className="bg-light p-4 h-100 shadow-sm"
                        ),
                        
                        # Main content area - content will be rendered here by callbacks
                        dbc.Col(
                            [
                                dcc.Location(id='url', refresh=False),
                                html.Div(id='page-content', className="p-4") 
                            ],
                            width=9,
                            className="h-100 overflow-auto" # Ensure content area scrolls if needed
                        )
                    ],
                    className="vh-100 m-0" # Ensure full height and no margin
                )
            ],
            fluid=True,
            className="p-0"
        )

    def _render_page_content(self, pathname):
        """Renders the content for the selected page based on the URL pathname."""
        if not pathname or pathname == "/":
            return self._create_overview_page()
        elif pathname == "/agents":
            return self._create_agent_control_page()
        elif pathname == "/models":
            return self._create_model_training_page()
        elif pathname == "/portfolio":
            return self._create_portfolio_page() # Placeholder
        elif pathname == "/strategies":
            return self._create_strategies_page() # Placeholder
        elif pathname == "/market-data":
            return self._create_market_data_page() # Placeholder
        elif pathname == "/risk":
            return self._create_risk_metrics_page() # Placeholder
        elif pathname == "/emerging-markets":
            return self._create_emerging_markets_page() # New page
        # Add more pages here
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname if pathname else 'None'} was not recognised...")
            ],
            className="p-3"
        )
    
    def _create_overview_page(self):
        """Creates the layout for the Overview page."""
        return html.Div([
            html.H3("AI Hedge Fund Dashboard", className="mb-4"),
            dbc.Row([
                # System Status Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.Div([
                                html.P("Status: ", className="d-inline fw-bold"),
                                html.Span(
                                    "Running",
                                    id="system-status",
                                    className="badge bg-success ms-2"
                                )
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col(
                                    dbc.Button(
                                        [html.I(className="fas fa-play me-2"), "Start System"],
                                        id="start-button",
                                        color="success",
                                        className="w-100 mb-2"
                                    )
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        [html.I(className="fas fa-stop me-2"), "Stop System"],
                                        id="stop-button",
                                        color="danger",
                                        className="w-100 mb-2"
                                    )
                                )
                            ], className="mb-4"),
                            html.H5("System Information", className="mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Version: ", className="fw-bold"),
                                    html.Span("1.0.0", id="system-version")
                                ], className="mb-2")
                            ])
                        ])
                    ])
                ], md=6, className="mb-4"),
                # Portfolio Summary Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Summary"),
                        dbc.CardBody([
                            html.H4("$0.00", className="card-title"),
                            html.Div([
                                html.Span("24h Change: ", className="fw-bold"),
                                html.Span("0.00%", className="text-muted")
                            ]),
                            html.Div([
                                html.Span("Positions: ", className="fw-bold"),
                                html.Span("0")
                            ], className="mt-2")
                        ])
                    ])
                ], md=6, className="mb-4"),
            ]),
            # Market Overview
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Overview"),
                        dbc.CardBody([
                            html.Div("Market data will be displayed here", id="market-overview-content")
                        ])
                    ])
                ])
            ])
        ])

    def _get_trading_options(self):
        return [
            {"label": "Live Trading", "value": "live"},
            {"label": "Paper Trading", "value": "paper"},
            {"label": "Backtesting", "value": "backtest"},
            {"label": "Training", "value": "training"}
        ]

    def _create_overview_page(self):
        """Creates the layout for the Overview page."""
        return html.Div([
            html.H3("AI Hedge Fund Dashboard", className="mb-4"),
            dbc.Row([
                # Left column - System controls
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Controls"),
                        dbc.CardBody([
                            html.H5("Trading Mode", className="mb-3"),
                            dcc.Dropdown(
                                id="mode-selection",
                                options=self._get_trading_options(),
                                value="paper",  # Default mode
                                className="mb-3"
                            ),
                            # Backtesting settings (initially hidden)
                            html.Div(
                                [
                                    html.H6("Backtesting Period", className="mt-3 mb-2"),
                                    dbc.Label("Start Date"),
                                    dcc.DatePickerSingle(
                                        id="backtest-start-date",
                                        date=datetime.now() - timedelta(days=365),
                                        className="mb-2 w-100"
                                    ),
                                    dbc.Label("End Date"),
                                    dcc.DatePickerSingle(
                                        id="backtest-end-date",
                                        date=datetime.now(),
                                        className="mb-2 w-100"
                                    ),
                                    dbc.Label("Initial Capital"),
                                    dbc.Input(
                                        id="initial-capital",
                                        type="number",
                                        value=100000,
                                        min=10000,
                                        step=10000,
                                        className="w-100"
                                    )
                                ],
                                id="backtest-settings",
                                style={"display": "none"}
                            ),
                            dbc.Button(
                                "Start System",
                                id="start-system-btn",
                                color="success",
                                className="me-2"
                            ),
                            dbc.Button(
                                "Stop System",
                                id="stop-system-btn",
                                color="danger",
                                className="me-2",
                                disabled=True
                            )
                        ])
                    ])
                ], md=6, className="mb-4"),
                # Right column - System status and metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("System Status"),
                        dbc.CardBody([
                            html.Div(id="system-status-container", children=[
                                html.Div([
                                    html.Strong("Status: "),
                                    html.Span(
                                        "Stopped",
                                        id="system-status-text",
                                        className="badge bg-secondary ms-2"
                                    )
                                ]),
                                html.Div([
                                    html.Strong("Uptime: "),
                                    html.Span("00:00:00", id="system-uptime")
                                ], className="mt-2")
                            ])
                        ])
                    ], className="mb-4"),
                    dbc.Card([
                        dbc.CardHeader("Recent Activity"),
                        dbc.CardBody([
                            dcc.Textarea(
                                id='system-logs-overview',
                                value='System initialized.\nMonitoring market data...\nAgent X performed action Y.',
                                style={'width': '100%', 'height': 200},
                                readOnly=True,
                                className="mb-3"
                            )
                        ])
                    ])
                ], md=6)
            ]),
            # Bottom row - Key metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Key Metrics"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Active Agents", className="card-title"),
                                            html.H4("0", id="active-agents-metric", className="text-center")
                                        ])
                                    ])
                                ], md=4, className="mb-3"),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Trades Today", className="card-title"),
                                            html.H4("0", id="trades-today-metric", className="text-center")
                                        ])
                                    ])
                                ], md=4, className="mb-3"),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H6("Current P&L", className="card-title"),
                                            html.H4("$0.00", id="current-pnl-metric", className="text-center")
                                        ])
                                    ])
                                ], md=4, className="mb-3")
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def _create_agent_control_page(self):
        """Creates the layout for the Agent Control page."""
        return html.Div([
            html.H3("Agent Control & Analysis", className="mb-4"),
            dbc.Tabs([
                # Tab for general agent control and analysis
                dbc.Tab(label="Agent Management", children=[
                    html.Div(className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Select Agent"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='agent-selector-dropdown',
                                        options=[], # To be populated by callback
                                        placeholder="Select an Agent",
                                        className="mb-3"
                                    ),
                                    dcc.Loading(id="loading-agent-details", children=[html.Div(id='agent-details-output')])
                                ])
                            ], className="mb-3")
                        ], md=5),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Ticker Analysis"),
                                dbc.CardBody([
                                    dcc.Input(
                                        id="ticker-input", 
                                        type="text", 
                                        placeholder="Enter Ticker (e.g., AAPL)", 
                                        className="mb-2 w-100"
                                    ),
                                    dcc.Dropdown(
                                        id='agent-for-ticker-analysis-dropdown',
                                        options=[], # To be populated by callback
                                        placeholder="Select Agent for Analysis",
                                        className="mb-3"
                                    ),
                                    dbc.Button(
                                        "Analyze Ticker", 
                                        id="analyze-ticker-button", 
                                        color="primary", 
                                        className="w-100"
                                    ),
                                    dcc.Loading(
                                        id="loading-ticker-analysis", 
                                        children=[
                                            html.Div(id="ticker-analysis-results", className="mt-3")
                                        ]
                                    )
                                ])
                            ])
                        ], md=7)
                    ])
                ]),
                # Tab for fundamental analysis
                dbc.Tab(label="Fundamental Analysis", children=[
                    html.Div(className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Fundamental Analysis"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.InputGroup([
                                                dbc.InputGroupText("Ticker"),
                                                dbc.Input(
                                                    id="fundamental-ticker-input",
                                                    placeholder="e.g. AAPL",
                                                    type="text",
                                                    className="w-100"
                                                ),
                                                dbc.Button(
                                                    "Analyze",
                                                    id="fundamental-analyze-button",
                                                    color="primary",
                                                    className="ms-2"
                                                )
                                            ], className="mb-3"),
                                        ], width=8),
                                    ]),
                                    dcc.Loading(
                                        id="loading-fundamental-analysis",
                                        type="default",
                                        children=[
                                            html.Div(id="fundamental-analysis-results", className="mt-3")
                                        ]
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def _create_model_training_page(self):
        """Creates the layout for the Model Training page."""
        return html.Div([
            html.H3("AI Model Training & Analysis", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Train New Model"),
                        dbc.CardBody([
                            dbc.Label("Select Model Type/Agent Base"),
                            dcc.Dropdown(
                                id="model-selection", 
                                options=[
                                    # These should ideally be fetched from an API endpoint listing available model blueprints/agent types
                                    {"label": "Warren Buffett Value Investor", "value": "legendary_investors_warren_buffett"},
                                    {"label": "Peter Lynch Growth Investor", "value": "legendary_investors_peter_lynch"},
                                    {"label": "Benjamin Graham Deep Value", "value": "legendary_investors_benjamin_graham"},
                                    {"label": "Ray Dalio All Weather", "value": "legendary_investors_ray_dalio"},
                                    {"label": "Momentum Strategy", "value": "technical_momentum"},
                                    {"label": "Mean Reversion Strategy", "value": "technical_mean_reversion"}
                                ], 
                                value="legendary_investors_warren_buffett", 
                                className="mb-3"
                            ),
                            dbc.Row([
                                dbc.Col([dbc.Label("Epochs"), dbc.Input(id="training-epochs", type="number", value=50, min=10, max=1000, step=10)], width=6, className="mb-2"),
                                dbc.Col([dbc.Label("Batch Size"), dbc.Input(id="training-batch-size", type="number", value=32, min=8, max=256, step=8)], width=6, className="mb-2")
                            ]),
                            dbc.Label("Analysis Type for Training Data"),
                            dbc.RadioItems(
                                id="analysis-type",
                                options=[
                                    {"label": "Fundamental", "value": "fundamental"},
                                    {"label": "Technical", "value": "technical"},
                                    {"label": "Sentiment", "value": "sentiment"},
                                    {"label": "All", "value": "all"}
                                ],
                                value="all",
                                inline=True,
                                className="mb-3"
                            ),
                            dbc.Button("Train Model", id="train-model-button", color="primary", className="w-100")
                        ])
                    ])
                ], md=5),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Training Results & Metrics"),
                        dbc.CardBody([
                            dcc.Loading(id="loading-model-training", children=[
                                html.Div(id="model-training-status"), 
                                html.Div(id="model-training-history-graph"), 
                                html.Div(id="model-metrics-display", className="mt-3")
                            ])
                        ])
                    ])
                ], md=7)
            ]),
            html.Hr(className="my-4"),
            html.H4("Existing Models/Agents", className="mb-3"),
            # Placeholder for a table or list of existing trained models/agents, with options to view details, deploy, or delete
            html.Div(id="existing-models-list", children="[Agent/Model listing will appear here - API call needed]")
        ])

    def _create_emerging_markets_page(self):
        """Creates the layout for the Emerging Markets page."""
        return html.Div([
            html.H3("Emerging Markets Analysis", className="mb-4"),
            dbc.Row([
                dbc.Col([
                    html.H5("Emerging Market Opportunities", className="mb-3"),
                    html.P("High-potential emerging markets identified by our AI:"),
                    dcc.Loading(
                        id="loading-emerging-markets",
                        type="circle",
                        children=html.Div(id="emerging-markets-table")
                    ),
                    html.Div(id="market-details", className="mt-3")
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="market-size-chart")
                ], width=6),
                dbc.Col([
                    dcc.Graph(id="market-growth-chart")
                ], width=6)
            ])
        ])

    def _create_portfolio_content(self):
        """Create the portfolio management content with real-time data and simulation features."""
        return html.Div([
            dbc.Card([
                dbc.CardHeader(html.H4("Portfolio Simulation")),
                dbc.CardBody([
                    # Add portfolio simulation controls
                    dbc.Row([
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Initial Capital"),
                                dbc.Input(
                                    id="initial-capital",
                                    value="100000",
                                    type="number",
                                    min="0"
                                ),
                                dbc.InputGroupText("$")
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.InputGroup([
                                dbc.InputGroupText("Risk Level"),
                                dbc.Select(
                                    id="risk-level",
                                    options=[
                                        {'label': 'Conservative', 'value': 'conservative'},
                                        {'label': 'Moderate', 'value': 'moderate'},
                                        {'label': 'Aggressive', 'value': 'aggressive'}
                                    ],
                                    value='moderate'
                                )
                            ])
                        ], width=3),
                        dbc.Col([
                            dbc.Button(
                                "Start Simulation",
                                id="start-simulation",
                                color="primary",
                                className="mt-2"
                            )
                        ], width=3)
                    ], className="mb-4"),
                    
                    # Add portfolio performance metrics
                    dbc.Row([
                        dbc.Col([
                            html.H6("Backtest Performance"),
                            html.H3("+12.5%"),
                            html.P("Since Inception", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            html.H6("Sharpe Ratio"),
                            html.H3("1.85"),
                            html.P("Risk-adjusted return", className="text-muted")
                        ], width=3),
                        dbc.Col([
                            html.H6("Max Drawdown"),
                            html.H3("-5.2%"),
                            html.P("Worst period", className="text-danger")
                        ], width=3),
                        dbc.Col([
                            html.H6("Win Rate"),
                            html.H3("78%"),
                            html.P("Positive trades", className="text-success")
                        ], width=3)
                    ], className="mb-4"),
                    
                    # Add portfolio holdings table
                    dbc.Row([
                        dbc.Col([
                            html.H6("Current Holdings"),
                            dash_table.DataTable(
                                id='portfolio-table',
                                columns=[
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Name', 'id': 'name'},
                                    {'name': 'Weight', 'id': 'weight'},
                                    {'name': 'Current Price', 'id': 'current_price'},
                                    {'name': '1D %', 'id': 'daily_change'},
                                    {'name': '1W %', 'id': 'weekly_change'},
                                    {'name': '1M %', 'id': 'monthly_change'}
                                ],
                                data=[],
                                style_table={'overflowX': 'auto'},
                                style_cell={'textAlign': 'center'},
                                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                                style_data_conditional=[
                                    {
                                        'if': {
                                            'filter_query': '{daily_change} < 0',
                                            'column_id': 'daily_change'
                                        },
                                        'color': 'red'
                                    },
                                    {
                                        'if': {
                                            'filter_query': '{daily_change} > 0',
                                            'column_id': 'daily_change'
                                        },
                                        'color': 'green'
                                    }
                                ]
                            )
                        ], width=12)
                    ])
                ])
            ])
        ])

        # Add callback for portfolio table updates
        @self.app.callback(
            Output("portfolio-table", "data"),
            Input("interval-component", "n_intervals"),
            prevent_initial_call=True
        )
        def update_portfolio_table(n_intervals):
            """Update portfolio holdings with real-time data."""
            try:
                # Get real-time market data
                portfolio_data = []
                if self.data_manager and hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    # Get current holdings
                    holdings = self.portfolio_manager.get_holdings()
                    for symbol in holdings:
                        # Get latest market data
                        data = self.data_manager.get_current_market_data(symbol)
                        if data:
                            # Calculate percentage changes
                            daily_change = ((data['close'] / data['close'].shift(1) - 1) * 100).iloc[-1]
                            weekly_change = ((data['close'] / data['close'].shift(5) - 1) * 100).iloc[-1]
                            monthly_change = ((data['close'] / data['close'].shift(21) - 1) * 100).iloc[-1]
                            
                            # Add to portfolio data
                            portfolio_data.append({
                                'symbol': symbol,
                                'name': symbol,  # Could be enhanced with a security name lookup
                                'weight': '10%',  # Placeholder - should come from portfolio manager
                                'current_price': f"${data['close'].iloc[-1]:.2f}",
                                'daily_change': f"{daily_change:.2f}%",
                                'weekly_change': f"{weekly_change:.2f}%",
                                'monthly_change': f"{monthly_change:.2f}%"
                            })
                return portfolio_data
            except Exception as e:
                logger.error(f"Error updating portfolio table: {e}")
                return []

        
    def _create_portfolio_page(self):
        """Creates the layout for the Portfolio page."""
        return html.Div([
            html.H1("Portfolio Management", className="mb-4"),
            html.P("Portfolio details and charts will go here.")
        ])
        
    def _create_strategies_page(self):
        """Creates the layout for the Strategy Management page."""
        return html.Div([
            html.H1("Strategy Management", className="mb-4"),
            html.P("Strategy allocation and performance will go here.")
        ])
        
    def _create_market_data_page(self):
        """Creates the layout for the Market Data page."""
        return html.Div([
            html.H1("Market Data", className="mb-4"),
            html.P("Market indices and economic indicators will go here.")
        ])
        
    def _create_risk_metrics_page(self):
        """Creates the layout for the Risk Metrics page."""
        return html.Div([
            html.H1("Risk Metrics", className="mb-4"),
            html.P("Risk metrics and heatmaps will go here.")
        ])
        
    def _create_emerging_markets_page(self):
        """Create the Emerging Markets page with market opportunities and analysis."""
        # Initialize the emerging markets component
        em_component = EmergingMarketsComponent()
        
        # Get the layout from the component
        layout = em_component.get_layout()
        
        return layout

    def _create_404_page(self):
        """Creates a 404 error page with navigation options."""
        return dbc.Container(className="py-5", children=[
            dbc.Row(className="justify-content-center", children=[
                dbc.Col(md=8, lg=6, children=[
                    dbc.Card(className="border-0 shadow-sm", children=[
                        dbc.CardBody(className="text-center p-5", children=[
                            html.H1("404", className="display-1 text-danger mb-4"),
                            html.H2("Page Not Found", className="mb-4"),
                            html.P(
                                "The page you are looking for might have been removed, "
                                "had its name changed, or is temporarily unavailable.",
                                className="text-muted mb-4"
                            ),
                            dbc.Button("Go to Homepage", href="/", color="primary", className="px-4"),
                            html.Div(className="mt-4", children=[
                                html.P("Or navigate to:", className="text-muted mb-2"),
                                html.Div(className="d-flex justify-content-center gap-3", children=[
                                    dbc.Button("Portfolio", href="/portfolio", color="outline-primary"),
                                    dbc.Button("Strategies", href="/strategies", color="outline-primary"),
                                    dbc.Button("Market Data", href="/market-data", color="outline-primary")
                                ])
                            ])
                        ])
                    ])
                ])
            ])
        ])

    def _setup_navigation_callbacks(self):
        """
        Set up callbacks for page navigation based on URL.
        """
        @self.app.callback(
            Output("page-content", "children"),
            [Input("url", "pathname")]
        )
        def display_page(pathname):
            try:
                if not pathname or pathname == "/" or pathname == "/overview":
                    return self._create_overview_page()
                elif pathname == "/portfolio":
                    return self._create_portfolio_page()
                elif pathname == "/strategies":
                    return self._create_strategies_page()
                elif pathname == "/risk":
                    return self._create_risk_metrics_page()
                elif pathname in ["/models", "/model-training"]:
                    return self._create_model_training_page()
                elif pathname in ["/agents", "/agent-control"]:
                    return self._create_agent_control_page()
                elif pathname in ["/emerging-markets", "/emerging-markets/", "/emerging-markets"]:  # Note: There's a typo in the URL path (should be 'markets')
                    return self._create_emerging_markets_page()
                elif pathname == "/market-data":
                    return self._create_market_data_page()
                else:
                    return self._create_404_page()
            except Exception as e:
                logger.error(f"Error rendering page {pathname}: {str(e)}")
                import traceback
                traceback.print_exc()
                return html.Div([
                    html.H1("Error loading page", className="text-danger"),
                    html.P(f"An error occurred while loading the page: {str(e)}"),
                    html.P("Please check the logs for more details.")
                ])
                
    def _setup_emerging_markets_callbacks(self):
        """Set up callbacks for the Emerging Markets page."""
        # Import our callbacks
        from ui.callbacks.emerging_markets_callbacks import register_callbacks
        
        # Register the callbacks with the app
        register_callbacks(self.app)
    def _setup_agent_callbacks(self):
        """Sets up callbacks related to agent control and ticker analysis."""
        
        # Callback to populate agent selector dropdowns
        @self.app.callback(
            [Output('agent-selector-dropdown', 'options'),
             Output('agent-for-ticker-analysis-dropdown', 'options'),
             Output('agent-selector-dropdown', 'value'), # To pre-select if only one agent or first agent
             Output('agent-for-ticker-analysis-dropdown', 'value')],
            [Input('url', 'pathname')]  # Trigger when page loads or URL changes to agent control page
        )
        def populate_agent_selectors(pathname):
            if pathname == "/agents": # Only fetch if on the relevant page
                try:
                    response = requests.get(f"{self.api_base_url}/agents") # API call to list agents
                    response.raise_for_status()
                    agents = response.json()
                    if not agents:
                        return [], [], None, None
                    options = [{'label': agent.get('name', agent.get('id', 'Unnamed Agent')), 'value': agent['id']} for agent in agents]
                    # Pre-select the first agent if available
                    first_agent_id = options[0]['value'] if options else None
                    return options, options, first_agent_id, first_agent_id
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching agents: {e}")
                    # Optionally, return an error message to be displayed in the UI
                    return [], [], None, None
                except Exception as e:
                    print(f"An unexpected error occurred while fetching agents: {e}")
                    return [], [], None, None
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update # Don't update if not on agent control page

        # Callback to display agent details
        @self.app.callback(
            Output('agent-details-output', 'children'),
            [Input('agent-selector-dropdown', 'value')],
            [State('url', 'pathname')] 
        )
        def display_agent_details(agent_id, pathname):
            if not agent_id or pathname != "/agents":
                if pathname == "/agents" and not agent_id:
                     return dbc.Alert("Select an agent to see details.", color="info", className="mt-2")
                return dash.no_update
            try:
                response = requests.get(f"{self.api_base_url}/agents/{agent_id}") # API call for agent details
                response.raise_for_status()
                agent_details = response.json()
                
                # Nicely formatted agent details
                card_content = [
                    html.H5(agent_details.get('name', 'N/A'), className="card-title"),
                    html.P(f"ID: {agent_details.get('id')}"),
                    html.P(f"Status: {agent_details.get('status', 'N/A')}"),
                    html.P(f"Type/Class: {agent_details.get('agent_class', 'N/A')}"),
                    html.H6("Configuration:", className="mt-3"),
                    dcc.Markdown(f"```json\n{json.dumps(agent_details.get('config', {}), indent=2)}\n```"),
                    # Add more details as available from your API
                    html.H6("Live Metrics (Example):", className="mt-3"),
                    html.P(f"Trades Made: {agent_details.get('metrics', {}).get('trades_made', 'N/A')}"),
                    html.P(f"Current P&L: {agent_details.get('metrics', {}).get('pnl', 'N/A')}"),
                    # Add buttons for agent actions if applicable (e.g., start/stop agent, update config)
                    # dbc.Button("Stop Agent", color="warning", className="mt-2 me-2"),
                    # dbc.Button("Update Config", color="info", className="mt-2"),
                ]
                return dbc.Card(dbc.CardBody(card_content), className="mt-2 shadow-sm")
            except requests.exceptions.RequestException as e:
                return dbc.Alert(f"Error fetching agent details: {e.response.text if e.response else str(e)}", color="danger", className="mt-2")
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", className="mt-2")

        # Callback for ticker analysis
        @self.app.callback(
            Output('ticker-analysis-results', 'children'),
            [Input('analyze-ticker-button', 'n_clicks')],
            [State('ticker-input', 'value'),
             State('agent-for-ticker-analysis-dropdown', 'value'),
             State('url', 'pathname')]
        )
        def analyze_ticker(n_clicks, ticker, agent_id, pathname):
            if not n_clicks or pathname != "/agents":
                return dash.no_update
            
            if not ticker or not agent_id:
                return dbc.Alert("Please enter a ticker and select an agent for analysis.", color="warning", className="mt-2")
            
            try:
                # API endpoint for ticker analysis by a specific agent
                # This might be /agents/{agent_id}/analyze or /analysis/ticker with agent_id in payload
                # Adjust based on your API design in api_server.py
                payload = {"ticker": ticker} # Agent ID is in the URL or could be part of payload
                response = requests.post(f"{self.api_base_url}/agents/{agent_id}/analyze_ticker", json=payload)
                response.raise_for_status()
                analysis_result = response.json()
                
                # Format the result for display
                result_display = [
                    html.H5(f"Analysis for {ticker.upper()} by Agent {agent_details.get('name', agent_id) if 'agent_details' in locals() else agent_id}", className="mt-3"), # Requires agent_details to be available or fetched
                    html.P(f"Recommendation: {analysis_result.get('recommendation', 'N/A')}"),
                    html.P(f"Confidence: {analysis_result.get('confidence', 'N/A')}"),
                    html.H6("Details:", className="mt-2"),
                    dcc.Markdown(f"```json\n{json.dumps(analysis_result.get('details', {}), indent=2)}\n```")
                ]
                return dbc.Card(dbc.CardBody(result_display), className="mt-2 shadow-sm")
            except requests.exceptions.RequestException as e:
                error_message = f"API Error during ticker analysis: {e.response.text if e.response else str(e)}"
                return dbc.Alert(error_message, color="danger", className="mt-2")
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {e}", color="danger", className="mt-2")
                
        # Callback for fundamental analysis
        @self.app.callback(
            Output("fundamental-analysis-results", "children"),
            [Input("fundamental-analyze-button", "n_clicks")],
            [State("fundamental-ticker-input", "value"),
             State("url", "pathname")],
            prevent_initial_call=True
        )
        def analyze_fundamentals(n_clicks, ticker, pathname):
            if not n_clicks or not ticker or pathname != "/agents":
                raise dash.exceptions.PreventUpdate
                
            try:
                # Call the API to get fundamental analysis
                response = requests.get(f"{self.api_base_url}/fundamental-analysis/{ticker}")
                response.raise_for_status()
                analysis = response.json()
                
                # Create the layout for the analysis results
                return self._create_fundamental_analysis_layout(analysis)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"Error fetching fundamental analysis: {e.response.text if e.response else str(e)}"
                return dbc.Alert(error_msg, color="danger")
            except Exception as e:
                return dbc.Alert(f"An unexpected error occurred: {str(e)}", color="danger")

    def _create_fundamental_analysis_layout(self, analysis):
        """Creates the layout for displaying fundamental analysis results."""
        # Extract data from analysis
        ticker = analysis.get('ticker', 'N/A')
        company_name = analysis.get('company_name', 'N/A')
        current_price = analysis.get('current_price', 'N/A')
        target_price = analysis.get('target_price', 'N/A')
        recommendation = analysis.get('recommendation', 'N/A')
        fundamentals = analysis.get('fundamentals', {})
        valuation = analysis.get('valuation_metrics', {})
        analysis_data = analysis.get('analysis', {})
        
        # Determine recommendation color
        rec_color = "success" if recommendation == "BUY" else "warning" if recommendation == "HOLD" else "danger"
        
        # Create the layout
        return html.Div([
            # Header with ticker and recommendation
            dbc.Row([
                dbc.Col([
                    html.H4(f"{company_name} ({ticker})", className="mb-0"),
                    html.Small(f"Current Price: ${current_price}", className="text-muted")
                ]),
                dbc.Col([
                    html.Div([
                        html.Strong("Recommendation: "),
                        dbc.Badge(recommendation, color=rec_color, className="ms-2")
                    ], className="text-end")
                ], width="auto")
            ], className="mb-4"),
            
            # Tabs for different sections
            dbc.Tabs([
                # Fundamentals Tab
                dbc.Tab(label="Fundamentals", children=[
                    html.Div(className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Key Metrics"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.Div("P/E Ratio", className="text-muted small"),
                                                html.H5(f"{fundamentals.get('pe_ratio', 'N/A')}", 
                                                       className=("text-success" if fundamentals.get('pe_ratio', 0) < 20 else "text-danger"))
                                            ], className="text-center p-2")
                                        ], className="border-end"),
                                        dbc.Col([
                                            html.Div([
                                                html.Div("PEG Ratio", className="text-muted small"),
                                                html.H5(f"{fundamentals.get('peg_ratio', 'N/A')}",
                                                       className=("text-success" if fundamentals.get('peg_ratio', 0) < 1.5 else "text-danger"))
                                            ], className="text-center p-2")
                                        ], className="border-end"),
                                        dbc.Col([
                                            html.Div([
                                                html.Div("ROE", className="text-muted small"),
                                                html.H5(f"{fundamentals.get('roe', 0)*100:.1f}%",
                                                       className=("text-success" if fundamentals.get('roe', 0) > 0.15 else "text-danger"))
                                            ], className="text-center p-2")
                                        ], className="border-end"),
                                        dbc.Col([
                                            html.Div([
                                                html.Div("Debt/Equity", className="text-muted small"),
                                                html.H5(f"{fundamentals.get('debt_to_equity', 'N/A')}",
                                                       className=("text-success" if fundamentals.get('debt_to_equity', 0) < 1.0 else "text-danger"))
                                            ], className="text-center p-2")
                                        ])
                                    ], className="mb-3"),
                                    
                                    # Detailed metrics
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([
                                                    html.Div(["Sector"], className="text-muted small"),
                                                    html.Div(fundamentals.get('sector', 'N/A'))
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Market Cap"], className="text-muted small"),
                                                    html.Div(fundamentals.get('market_cap', 'N/A'))
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Current Ratio"], className="text-muted small"),
                                                    html.Div(fundamentals.get('current_ratio', 'N/A'),
                                                           className=("text-success" if fundamentals.get('current_ratio', 0) > 1.5 else "text-danger"))
                                                ])
                                            ], flush=True)
                                        ], md=6),
                                        dbc.Col([
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([
                                                    html.Div(["Dividend Yield"], className="text-muted small"),
                                                    html.Div(f"{fundamentals.get('dividend_yield', 0)*100:.2f}%" if fundamentals.get('dividend_yield') is not None else 'N/A')
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Revenue Growth"], className="text-muted small"),
                                                    html.Div(f"{fundamentals.get('revenue_growth', 0)*100:.1f}%" if fundamentals.get('revenue_growth') is not None else 'N/A',
                                                           className=("text-success" if fundamentals.get('revenue_growth', 0) > 0 else "text-danger"))
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Earnings Growth"], className="text-muted small"),
                                                    html.Div(f"{fundamentals.get('earnings_growth', 0)*100:.1f}%" if fundamentals.get('earnings_growth') is not None else 'N/A',
                                                           className=("text-success" if fundamentals.get('earnings_growth', 0) > 0 else "text-danger"))
                                                ])
                                            ], flush=True)
                                        ], md=6)
                                    ])
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Valuation Tab
                dbc.Tab(label="Valuation", children=[
                    html.Div(className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Valuation Metrics"),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.ListGroup([
                                                dbc.ListGroupItem([
                                                    html.Div(["Current Price"], className="text-muted small"),
                                                    html.Div(f"${current_price}", className="h5 mb-0")
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Target Price"], className="text-muted small"),
                                                    html.Div(f"${target_price}", 
                                                           className=("h5 mb-0 text-success" if target_price > current_price else "h5 mb-0 text-danger"))
                                                ]),
                                                dbc.ListGroupItem([
                                                    html.Div(["Upside/Downside"], className="text-muted small"),
                                                    html.Div(
                                                        f"{((target_price / current_price) - 1) * 100:.1f}%" if current_price and target_price else 'N/A',
                                                        className=("text-success" if target_price > current_price else "text-danger")
                                                    )
                                                ])
                                            ], flush=True)
                                        ], md=6),
                                        dbc.Col([
                                            dbc.Card([
                                                dbc.CardHeader("Valuation Methods"),
                                                dbc.ListGroup(flush=True, children=[
                                                    dbc.ListGroupItem([
                                                        html.Div(["P/E Based"], className="text-muted small"),
                                                        html.Div(f"${valuation.get('pe_valuation', 'N/A')}")
                                                    ]),
                                                    dbc.ListGroupItem([
                                                        html.Div(["DCF"], className="text-muted small"),
                                                        html.Div(f"${valuation.get('dcf_valuation', 'N/A')}")
                                                    ]),
                                                    dbc.ListGroupItem([
                                                        html.Div(["Comparables"], className="text-muted small"),
                                                        html.Div(f"${valuation.get('comparable_companies_valuation', 'N/A')}")
                                                    ])
                                                ])
                                            ])
                                        ], md=6)
                                    ])
                                ])
                            ])
                        ])
                    ])
                ]),
                
                # Analysis Tab
                dbc.Tab(label="Analysis", children=[
                    html.Div(className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Strengths"),
                                dbc.CardBody([
                                    dbc.ListGroup(flush=True, children=[
                                        dbc.ListGroupItem(html.Span(html.I(className="fas fa-check-circle text-success me-2") + item))
                                        for item in analysis_data.get('strengths', ["No strengths identified."])
                                    ])
                                ])
                            ], className="mb-3"),
                            
                            dbc.Card([
                                dbc.CardHeader("Risks"),
                                dbc.CardBody([
                                    dbc.ListGroup(flush=True, children=[
                                        dbc.ListGroupItem(html.Span(html.I(className="fas fa-exclamation-triangle text-warning me-2") + item))
                                        for item in analysis_data.get('risks', ["No specific risks identified."])
                                    ])
                                ])
                            ])
                        ], md=6),
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Summary"),
                                dbc.CardBody([
                                    html.P(analysis_data.get('summary', 'No analysis summary available.'), 
                                          className="text-justify"),
                                    html.Hr(),
                                    html.Div([
                                        dbc.Button("View Full Report", color="primary", className="me-2"),
                                        dbc.Button("Add to Watchlist", color="outline-primary")
                                    ], className="d-flex justify-content-end mt-3")
                                ])
                            ])
                        ], md=6)
                    ])
                ])
            ])
        ])

    def _setup_callbacks(self):
        """Set up general Dash callbacks for system control and UI updates."""
        # Callback to toggle backtesting settings visibility
        @self.app.callback(
            Output('backtest-settings', 'style'),
            [Input('mode-selection', 'value')]
        )
        def toggle_backtest_settings(selected_mode):
            return {'display': 'block'} if selected_mode == 'backtest' else {'display': 'none'}
            
        # Callback to handle system start/stop
        @self.app.callback(
            [
                Output('start-system-btn', 'disabled'),
                Output('stop-system-btn', 'disabled'),
                Output('system-status-text', 'children'),
                Output('system-status-text', 'className'),
                Output('system-logs-overview', 'value')
            ],
            [
                Input('start-system-btn', 'n_clicks'),
                Input('stop-system-btn', 'n_clicks')
            ],
            [
                State('mode-selection', 'value'),
                State('backtest-start-date', 'date'),
                State('backtest-end-date', 'date'),
                State('initial-capital', 'value'),
                State('system-logs-overview', 'value')
            ]
        )
        def toggle_system(start_clicks, stop_clicks, mode, start_date, end_date, capital, current_logs):
            ctx = dash.callback_context
            if not ctx.triggered:
                raise dash.exceptions.PreventUpdate
                
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'start-system-btn':
                # Start system logic
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} - System started in {mode} mode.\n"
                return True, False, "Running", "badge bg-success ms-2", log_entry + (current_logs or '')
                
            elif button_id == 'stop-system-btn':
                # Stop system logic
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} - System stopped.\n"
                return False, True, "Stopped", "badge bg-secondary ms-2", log_entry + (current_logs or '')
                
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # Callback to update system metrics periodically
        @self.app.callback(
            [
                Output('active-agents-metric', 'children'),
                Output('trades-today-metric', 'children'),
                Output('current-pnl-metric', 'children'),
                Output('system-uptime', 'children')
            ],
            [Input('interval-component', 'n_intervals')],
            [State('system-status-text', 'children')]
        )
        def update_metrics(n, status):
            if status != "Running":
                raise dash.exceptions.PreventUpdate
                
            try:
                # Simulate updating metrics (replace with actual API calls)
                active_agents = len(self.agent_manager.agents) if hasattr(self, 'agent_manager') and self.agent_manager else 0
                trades_today = 0  # Replace with actual trade count
                pnl = "$0.00"  # Replace with actual P&L calculation
                
                # Calculate uptime (simplified)
                uptime_seconds = n % (24 * 60 * 60)  # Reset daily for demo
                hours, remainder = divmod(uptime_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                return active_agents, trades_today, pnl, uptime_str
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                raise dash.exceptions.PreventUpdate
            
        # Callback to handle mode selection changes
        @self.app.callback(
            Output('mode-selection', 'disabled'),
            [Input('system-status-text', 'children')]
        )
        def disable_mode_dropdown(status):
            return status == "Running"

        @self.app.callback(
            Output("existing-models-list", "children"),
            [Input('url', 'pathname')] 
        )
        def list_existing_models_callback(pathname):
            return self.list_existing_models(pathname)

        @self.app.callback(
            [Output('model-training-status', 'children'),
             Output('model-training-history-graph', 'children'),
             Output('model-metrics-display', 'children')],
            [Input('train-model-button', 'n_clicks')],
            [State('model-selection', 'value'),
             State('training-epochs', 'value'),
             State('training-batch-size', 'value'),
             State('analysis-type', 'value'),
             State('url', 'pathname')]
        )
        def handle_train_model(n_clicks, model_name, epochs, batch_size, analysis_type, pathname):
            if not n_clicks or pathname != "/models":
                return dash.no_update, dash.no_update, dash.no_update

            if not model_name or epochs is None or batch_size is None or not analysis_type:
                return dbc.Alert("Please select a model and specify all training parameters.", color="warning"), dash.no_update, dash.no_update

            try:
                payload = {
                    'model_name': model_name,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'analysis_type': analysis_type
                }
                response = requests.post(f"{self.api_base_url}/models/train", json=payload)
                response.raise_for_status()
                data = response.json()

                status_msg = dbc.Alert(data.get("message", "Training initiated successfully."), 
                                    color="success", className="mt-2")
                
                history_graph = html.Div()
                if 'history' in data and data['history']:
                    history_df = pd.DataFrame(data['history'])
                    fig = go.Figure()
                    for col in history_df.columns:
                        if col.lower() != 'epoch':
                            fig.add_trace(go.Scatter(y=history_df[col], name=col, mode='lines+markers'))
                
                    fig.update_layout(
                        title="Training History", 
                        xaxis_title="Epoch", 
                        yaxis_title="Metric Value", 
                        margin=dict(t=30, b=10, l=10, r=10), 
                        height=300
                    )
                    history_graph = dcc.Graph(figure=fig)
            
                metrics_display = html.Div([
                    html.H6("Final Metrics:", className="mt-3"),
                    dcc.Markdown(f"```json\n{json.dumps(data.get('metrics', {}), indent=2)}\n```")
                ])
            
                return status_msg, history_graph, metrics_display
                
            except requests.exceptions.RequestException as e:
                error_detail = e.response.text if e.response else str(e)
                return dbc.Alert(
                    f"API Error during model training: {error_detail}", 
                    color="danger"
                ), dash.no_update, dash.no_update
                
            except Exception as e:
                return dbc.Alert(
                    f"An unexpected error occurred: {e}", 
                    color="danger"
                ), dash.no_update, dash.no_update

    def list_existing_models(self, pathname):
        if pathname == "/models":
            try:
                # response = requests.get(f"{self.api_base_url}/models") # Or /agents if models are part of agents
                # response.raise_for_status()
                # models = response.json()
                # model_list_items = [dbc.ListGroupItem(f"{m.get('name', m.get('id'))} - Type: {m.get('type')}") for m in models]
                # return dbc.ListGroup(model_list_items)
                return dbc.Alert("Fetching existing models... (API call pending)", color="info") # Placeholder
            except Exception as e:
                return dbc.Alert(f"Error fetching models: {e}", color="danger")
        return dash.no_update

    def run(self):
        """Run the Dash app. For production, use a proper WSGI server like Gunicorn or Waitress."""
        try:
            self.app.run_server(debug=True, port=self.dashboard_port, host='0.0.0.0')
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")
            raise

# Example usage (for direct script execution, not for production)
if __name__ == '__main__':
    # This config is a placeholder. In a real scenario, load it from a file or environment.
    # Ensure this matches your actual config structure for ports.
    from utils.config_manager import ConfigManager # Assuming ConfigManager is in utils
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if not config_path.exists():
        # Create a dummy config.yaml if it doesn't exist for standalone testing
        dummy_config_content = """
ui:
  dashboard_port: 8080
  api_port: 8081
api:
  host: "0.0.0.0"
  port: 8081        
"""
        with open(config_path, 'w') as f:
            f.write(dummy_config_content)
        print(f"Created dummy {config_path} for testing.")

    app_config = ConfigManager.load_config(config_path)

    dashboard = DashDashboard(config=app_config)
    # Get the port from the config
    port = app_config.get('ui', {}).get('dashboard_port', 8088)
    print(f"Starting dashboard on port {port}...")
    
    # To run directly for testing (without Gunicorn/Waitress):
    try:
        dashboard.app.run_server(debug=True, port=port, host='0.0.0.0')
    except Exception as e:
        print(f"Error running dashboard: {e}")
        raise